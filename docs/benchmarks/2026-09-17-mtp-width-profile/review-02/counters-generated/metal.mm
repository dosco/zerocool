#include "mtp_expert_scratch.hpp"
#include "qwen/metal.hpp"
#include "qwen_embedded.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/ps/IOPowerSources.h>
#include <CommonCrypto/CommonDigest.h>
#include <fstream>
#include <set>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <mach/mach.h>
#include <sys/sysctl.h>

namespace freellm::qwen {
namespace {
// Owner callbacks can run on I/O threads. This flag excludes only callbacks
// nested in this thread's command-group destruction from the outside total.
struct BufferCosts;
thread_local const BufferCosts* retiring_command_group=nullptr;
struct BufferCosts {
    struct Counts {
        std::atomic<uint64_t> allocations{0},allocated_bytes{0},allocation_ns{0};
        std::atomic<uint64_t> owner_releases{0},released_bytes{0},outside_release_ns{0};
    };
    std::array<Counts,6> classes;
    std::atomic<uint64_t> groups{0},group_retirement_ns{0};
    Json json() const {
        constexpr std::array<const char*,6> names={"temporary","resident","state","expert","snapshot","workspace"};
        Json rows=Json::object();
        for(size_t i=0;i<names.size();++i) {
            const auto& c=classes[i];rows[names[i]]={{"allocations",c.allocations.load()},
                {"allocated_bytes",c.allocated_bytes.load()},{"allocation_ns",c.allocation_ns.load()},
                {"owner_releases",c.owner_releases.load()},{"owner_released_bytes",c.released_bytes.load()},
                {"outside_release_ns",c.outside_release_ns.load()}};
        }
        return {{"kind","buffer_costs_v1"},{"classes",rows},{"retired_groups",groups.load()},
            {"group_retirement_ns",group_retirement_ns.load()},
            {"scope","Successful physical allocation calls; command-group destruction; last-owner callbacks outside that destruction. CPU interval sums can overlap GPU and other threads; not predicted latency savings."}};
    }
};
struct AllocationTimer {
    BufferCosts* costs;AllocationClass kind;uint64_t start=0,bytes=0;
    void begin() {if(costs) start=monotonic_ns();}
    ~AllocationTimer() {
        if(!costs || !bytes) return;
        auto& c=costs->classes.at(size_t(kind));c.allocations.fetch_add(1,std::memory_order_relaxed);
        c.allocated_bytes.fetch_add(bytes,std::memory_order_relaxed);
        c.allocation_ns.fetch_add(monotonic_ns()-start,std::memory_order_relaxed);
    }
};
struct ReleaseTimer {
    BufferCosts* costs;AllocationClass kind;uint64_t bytes,start=0;bool outside;
    ReleaseTimer(BufferCosts* c,AllocationClass k,uint64_t b):costs(c),kind(k),bytes(b),outside(c!=retiring_command_group) {if(costs && outside) start=monotonic_ns();}
    ~ReleaseTimer() {
        if(!costs) return;
        auto& c=costs->classes.at(size_t(kind));c.owner_releases.fetch_add(1,std::memory_order_relaxed);
        c.released_bytes.fetch_add(bytes,std::memory_order_relaxed);
        if(outside) c.outside_release_ns.fetch_add(monotonic_ns()-start,std::memory_order_relaxed);
    }
};
struct GroupRetirementTimer {
    BufferCosts* costs;const BufferCosts* previous=retiring_command_group;uint64_t start=0;
    explicit GroupRetirementTimer(BufferCosts* c):costs(c) {if(costs) {start=monotonic_ns();retiring_command_group=costs;}}
    ~GroupRetirementTimer() {
        if(costs) {
            retiring_command_group=previous;costs->groups.fetch_add(1,std::memory_order_relaxed);
            costs->group_retirement_ns.fetch_add(monotonic_ns()-start,std::memory_order_relaxed);
        }
    }
};
}
struct Accounting {
    std::atomic<uint64_t> live{0}, high{0};
    uint64_t limit = 22*GiB;
    std::unique_ptr<BufferCosts> costs;
};
// The last Buffer owner can disappear on an I/O thread. It queues retirement;
// only the coordinator changes Metal residency and releases the accounting.
struct ResidencyRegistry {
    struct Entry { uint64_t bytes; AllocationClass kind; };
    std::mutex mutex;
    id<MTLResidencySet> set=nil;
    std::unordered_map<void*,Entry> entries;
    std::vector<void*> retired;
    std::shared_ptr<Accounting> accounting;
    bool active=true;
    uint64_t bytes=0, commits=0, removals=0, overhead=0;
    std::string mode="off";
    void release(void* p,uint64_t charge,AllocationClass kind) {
        ReleaseTimer timer(accounting->costs.get(),kind,charge);
        std::lock_guard lock(mutex);
        if(active && entries.contains(p)) retired.push_back(p);
        else { CFRelease(p);accounting->live.fetch_sub(charge); }
    }
    void drain() {
        if(mode=="off") return;
        std::lock_guard lock(mutex);
        if(@available(macOS 15.0,*)) {
            for(auto p:retired) [set removeAllocation:(__bridge id<MTLBuffer>)p];
            if(!retired.empty()) { [set commit];++commits; }
        }
        for(auto p:retired) {
            const auto charge=entries.at(p).bytes;bytes-=charge;entries.erase(p);
            CFRelease(p);accounting->live.fetch_sub(charge);++removals;
        }
        retired.clear();
    }
    void enroll(void* p,uint64_t charge,AllocationClass kind) {
        if(mode=="off" || kind==AllocationClass::Temporary || kind==AllocationClass::Snapshot || kind==AllocationClass::Workspace ||
           (kind==AllocationClass::Expert && mode!="core-cache")) return;
        std::lock_guard lock(mutex);
        if(@available(macOS 15.0,*)) {
            if(entries.contains(p)) return;
            entries.emplace(p,Entry{charge,kind});bytes+=charge;
            [set addAllocation:(__bridge id<MTLBuffer>)p];[set commit];++commits;
            overhead=set.allocatedSize>bytes?set.allocatedSize-bytes:0;
            // This is part of the existing 1GiB bookkeeping/driver reserve.
            if(overhead>64*MiB) throw std::runtime_error("residency metadata exceeds reserved 64MiB");
        }
    }
    Json json() {
        std::lock_guard lock(mutex);
        Json classes=Json::object();
        for(const auto& [pointer,e]:entries) {
            (void)pointer;
            const auto name=e.kind==AllocationClass::Resident?"resident":e.kind==AllocationClass::State?"state":"expert";
            classes[name]=classes.value(name,uint64_t(0))+e.bytes;
        }
        return {{"mode",mode},{"registered_bytes",bytes},{"allocations",entries.size()},
            {"bytes_by_class",classes},{"set_overhead_bytes",overhead},{"commits",commits},{"removals",removals},
            {"pending_retirements",retired.size()},{"physical_residency_guaranteed",false}};
    }
    void close(id<MTLCommandQueue> queue) {
        drain();std::lock_guard lock(mutex);active=false;
        if(@available(macOS 15.0,*)) if(set) {
            [queue removeResidencySet:set];[set removeAllAllocations];[set commit];[set endResidency];set=nil;
        }
        entries.clear();bytes=0;overhead=0;
    }
};
struct Metal::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLCommandBuffer> current;
    id<MTLComputeCommandEncoder> encoder;
    NSMutableDictionary<NSString*,id<MTLComputePipelineState>>* pipelines;
    struct Pending { id<MTLCommandBuffer> cb; std::vector<Buf> buffers; std::shared_ptr<Completion> completion; Json operations;
        id<MTLCounterSampleBuffer> samples; NSUInteger sample_count; MTLTimestamp cpu_start,gpu_start; };
    id<MTLCounterSampleBuffer> samples;
    id<MTLCounterSet> timestamps;
    NSUInteger sample_count=0;
    MTLTimestamp sample_cpu_start=0,sample_gpu_start=0;
    std::vector<Buf> current_buffers;
    std::deque<Pending> pending;
    std::shared_ptr<CompletionEvents> events=std::make_shared<CompletionEvents>();
    uint64_t peak_groups=0;
    std::shared_ptr<Accounting> accounting=std::make_shared<Accounting>();
    std::shared_ptr<ResidencyRegistry> residency=std::make_shared<ResidencyRegistry>();
    struct Scratch { std::vector<Buf> buffers; size_t cursor=0; uint64_t bytes=0,capacity=0,peak=0,reuses=0,allocations=0,wait_ns=0; std::shared_ptr<Completion> last; };
    std::array<Scratch,2> scratch;
    int active_scratch=-1;
    uint64_t submissions=0, waits=0, dispatches=0, allocations=0, pool_reuses=0;
    bool retained_references=true;
    bool expanded_q8=false,expanded_scope=false;
    KernelConfig config;
    Json operations=Json::array(),profile=Json::array(),context=Json::object(),kernel_counts=Json::object();
    std::string request_phase="unspecified";
    uint64_t encode_ns=0,host_wait_ns=0,profile_entries=0,gpu_command_ns=0;
    Json operation_summary=Json::object();
    bool profile_truncated=false;
    // Two 16-token decode windows use 101,600 dispatches for this model.
    // Keep historical all-dispatch captures on their existing bound.
    uint64_t profile_limit() const {return 20000;}
    Json matrix=Json::object(),capture=Json::array();
    std::set<std::string> captured_shapes;uint64_t captured_bytes=0;
};
void KernelConfig::validate() const {
    if(q4_decode!="reference" && q4_decode!="packed-r2")
        throw std::invalid_argument("Q4 decode must be reference or packed-r2");
    if(q4_decode!="reference" && (policy!="candidate" || !shape_table.is_null()))
        throw std::invalid_argument("packed Q4 decode requires candidate mode without a shape table");
    if(profile_decode_only && (!profile || counter_profile))
        throw std::invalid_argument("decode-only profiling requires command-group profiling");
    if(route_selection!="serial" && route_selection!="simd")
        throw std::invalid_argument("route selection must be serial or simd");
    if(route_selection!="serial" && policy!="candidate")
        throw std::invalid_argument("parallel route selection requires candidate policy");
    if(attention_score_tiles!="full" && attention_score_tiles!="skip-masked")
        throw std::invalid_argument("attention score tiles must be full or skip-masked");
    if(counter_profile && !profile) throw std::invalid_argument("counter sampling requires diagnostic profiling");
    if(q8_decode_rows!=0 && q8_decode_rows!=2 && q8_decode_rows!=4 && q8_decode_rows!=8)
        throw std::invalid_argument("Q8 decode rows must be 0, 2, 4, or 8");
    if(q8_decode_rows && (policy!="candidate" || !shape_table.is_null()))
        throw std::invalid_argument("packed Q8 decode requires candidate mode without a shape table");
    if(policy!="reference" && policy!="auto" && policy!="candidate") throw std::invalid_argument("kernel policy must be reference, auto, or candidate");
    if(token_tile!=1 && token_tile!=2 && token_tile!=4 && token_tile!=8) throw std::invalid_argument("token tile must be 1, 2, 4, or 8");
    if(affine_rows!=1 && affine_rows!=2 && affine_rows!=4) throw std::invalid_argument("affine rows must be 1, 2, or 4");
    if(affine_rows!=1 && token_tile==2) throw std::invalid_argument("blocked affine requires token tile 1, 4, or 8");
    if(gdn!="original" && gdn!="precompute" && gdn!="staged") throw std::invalid_argument("GDN path must be original, precompute, or staged");
    if((gdn_rows!=4 && gdn_rows!=8) || (gdn_block!=4 && gdn_block!=8 && gdn_block!=16)) throw std::invalid_argument("unsupported GDN staging geometry");
    if(!shape_table.is_null()) {
        if(policy!="candidate" || token_tile!=1 || affine_rows!=1 || gate_pair || !shape_table.is_object() || !shape_table.contains("rules") ||
           !shape_table["rules"].is_array() || shape_table["rules"].size()>4096)
            throw std::invalid_argument("shape policy requires candidate mode and no forced tile");
        std::set<std::string> keys;
        for(const auto& rule:shape_table["rules"]) {
            if(rule.size()<8 || rule.size()>10) throw std::invalid_argument("invalid shape rule fields");
            const auto tile=rule.at("tile").get<uint32_t>();
            if(tile!=1 && tile!=2 && tile!=4 && tile!=8) throw std::invalid_argument("invalid shape tile");
            const auto rows=rule.value("output_rows",1u);
            const bool pair=rule.value("gate_pair",false);
            if((rows!=1 && rows!=2 && rows!=4) || (rows!=1 && (rule.at("fused").get<bool>() || tile==2 || (rule.at("rows")!=1 && rule.at("bits")!=8))) || (pair && (!rule.at("fused").get<bool>() || rule.at("rows")!=1))) throw std::invalid_argument("unsupported shape variant");
            for(auto name:{"K","N","rows","group","bits"}) if(!rule.at(name).is_number_unsigned() &&
                (!rule.at(name).is_number_integer() || rule.at(name).get<int64_t>()<=0)) throw std::invalid_argument("invalid shape dimension");
            for(auto name:{"K","N","rows","group","bits"}) if(!rule[name].get<uint64_t>() || rule[name].get<uint64_t>()>UINT32_MAX) throw std::invalid_argument("shape dimension out of range");
            if(!rule.at("fused").is_boolean() || !rule.at("gathered").is_boolean()) throw std::invalid_argument("invalid shape operation");
            auto key=Json::array({rule["K"],rule["N"],rule["rows"],rule["group"],rule["bits"],rule["fused"],rule["gathered"]});if(!keys.insert(key.dump()).second) throw std::invalid_argument("duplicate shape rule");
        }
    }
    if(!operator_capture.empty() && !profile) throw std::invalid_argument("operator capture is diagnostic profiling");
    if(capture_layer < -2 || capture_layer>=Layers) throw std::invalid_argument("capture layer outside model");
    if((!capture_phase.empty() || !capture_operator.empty() || capture_layer!=-2) && operator_capture.empty())
        throw std::invalid_argument("capture filters require an operator capture directory");
    if(policy!="candidate" && (token_tile!=1 || gdn!="original" || affine_rows!=1 || gate_pair)) throw std::invalid_argument("forced kernels require candidate policy");
}
Json KernelConfig::json() const {
    return {{"q4_decode",q4_decode},{"route_selection",route_selection},{"attention_score_tiles",attention_score_tiles},{"policy",policy},{"token_tile",token_tile},{"affine_rows",affine_rows},{"q8_decode_rows",q8_decode_rows},{"gate_pair",gate_pair},{"gdn",gdn},{"gdn_rows",gdn_rows},
        {"gdn_block",gdn_block},{"shape_table",shape_table},{"operator_capture",operator_capture.string()},
        {"capture_filter",{{"phase",capture_phase},{"operator",capture_operator},{"layer",capture_layer}}},
        {"profile",profile},{"profile_decode_only",profile_decode_only},{"counter_profile",counter_profile},{"automatic_rules_promoted",false}};
}
void Metal::configure(KernelConfig config) {
    config.validate();
    if(config.counter_profile && !impl_->timestamps) {
        if(![impl_->device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary])
            throw std::runtime_error("GPU stage timestamp sampling unavailable");
        for(id<MTLCounterSet> set in impl_->device.counterSets)
            if([set.name isEqualToString:MTLCommonCounterSetTimestamp]) impl_->timestamps=set;
        if(!impl_->timestamps) throw std::runtime_error("GPU timestamp counter set unavailable");
    }
    if(!config.shape_table.is_null() && (config.shape_table.at("build_fingerprint")!=BuildFingerprint ||
       (!config.artifact_revision.empty() && config.shape_table.at("artifact_revision")!=config.artifact_revision)))
        throw std::invalid_argument("shape policy build/artifact identity differs");
    impl_->config=std::move(config);
}
void Metal::request_phase(std::string phase) {impl_->request_phase=std::move(phase);}
void Metal::route(const Buf& logits,const Buf& ids,const Buf& weights,uint32_t tokens) {
    if(!tokens || tokens>8192 || !logits || !ids || !weights ||
       logits->bytes!=uint64_t(tokens)*Experts*4 || ids->bytes!=uint64_t(tokens)*TopK*4 || weights->bytes!=ids->bytes)
        throw std::invalid_argument("invalid route selection buffers");
    dispatch(impl_->config.route_selection=="simd"?"route_simd":"route",{{logits},{ids},{weights}},{tokens},tokens*32);
}
void Metal::label(std::string stage,int layer,uint32_t tokens,uint32_t offset,std::span<const uint32_t> experts) {
    impl_->expanded_scope=(stage=="gdn" || stage=="attention" || stage=="logits");
    if(impl_->config.profile) {impl_->context={{"stage",stage},{"layer",layer},{"tokens",tokens},{"offset",offset}};
        if(!experts.empty()) impl_->context["experts"]=experts;}
}
void Metal::prepare_pipelines() {
    @autoreleasepool {
        for(NSString* key in impl_->library.functionNames) if(!impl_->pipelines[key]) {
            NSError* error=nil;
            auto function=[impl_->library newFunctionWithName:key];
            auto pipeline=[impl_->device newComputePipelineStateWithFunction:function error:&error];
            if(!pipeline) throw std::runtime_error("Metal pipeline: "+std::string(error.localizedDescription.UTF8String));
            impl_->pipelines[key]=pipeline;
        }
    }
}
Json Metal::take_profile() {
    reap();auto result=Json{{"command_groups",std::move(impl_->profile)},{"truncated",impl_->profile_truncated},
        {"coverage",impl_->config.profile_decode_only?"decode-only":"all-dispatches"},{"entry_limit",impl_->profile_limit()},
        {"operation_summary",std::move(impl_->operation_summary)},
        {"timing_kind",impl_->config.counter_profile?"instrumented per-dispatch compute passes; submission boundaries preserved":"existing command groups; mixed stages are not isolated kernel costs"},
        {"counter_sample_capacity_per_group",4096},{"counter_storage_reserve_bytes",3*32768},
        {"normal_request_latency_qualified",false}};
    impl_->profile=Json::array();impl_->operation_summary=Json::object();impl_->profile_entries=0;impl_->profile_truncated=false;return result;
}
Metal::Metal() : impl_(std::make_unique<Impl>()) {
    if(const char* mode=std::getenv("FREELLM_Q8_EXPANDED")) {
        if(std::string_view(mode)!="packed") throw std::runtime_error("invalid expanded Q8 mode");
        impl_->expanded_q8=true;
    }
    impl_->residency->accounting=impl_->accounting;
    @autoreleasepool {
        impl_->device=MTLCreateSystemDefaultDevice();
        if(!impl_->device || !impl_->device.hasUnifiedMemory) throw std::runtime_error("Apple Silicon Metal device unavailable");
        impl_->queue=[impl_->device newCommandQueue];
        if(!impl_->queue) throw std::runtime_error("cannot create Metal command queue");
        NSError* error=nil;
        MTLCompileOptions* options=[MTLCompileOptions new];
        if(@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
        else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled=NO;
#pragma clang diagnostic pop
        }
        const auto source=std::string(MetalSource)+R"PACKED_Q8(
// Developer-only Q8 word loads for four tokens and one output row.
// These admitted GDN, attention and output shapes use width eight in the existing affine_multi kernel.
// Preserve its lane partition, scalar additions, SIMD reduction and BF16 round.
kernel void q8_expanded_t4_w8(device const uint* w [[buffer(0)]],
    device const ushort* s [[buffer(1)]],device const ushort* b [[buffer(2)]],
    device const float* x [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint tid [[thread_position_in_grid]]) {
    const uint K=p[0],N=p[1],row=tid/32,lane=tid%32;
    if(row>=N || p[2]!=4 || p[3]!=64 ||
       !((K==2560 && (N==10240 || N==6144 || N==12288 || N==248320)) || (K==6144 && N==2560))) return;
    float result[4]={0,0,0,0};
    for(uint base=lane*8;base<K;base+=32*8) {
        const uint words[2]={w[row*(K/4)+base/4],w[row*(K/4)+base/4+1]};
        float dot[4]={0,0,0,0},sum[4]={0,0,0,0};
        #pragma unroll
        for(uint i=0;i<8;++i) {
            const uint code=(words[i/4]>>(8*(i%4)))&255;
            #pragma unroll
            for(uint t=0;t<4;++t) {
                const float v=x[t*K+base+i];
                sum[t]+=v;dot[t]+=v*float(code);
            }
        }
        const uint g=row*(K/64)+base/64;
        const float scale=b16(s[g]),bias=b16(b[g]);
        #pragma unroll
        for(uint t=0;t<4;++t) result[t]+=scale*dot[t]+sum[t]*bias;
    }
    #pragma unroll
    for(uint t=0;t<4;++t) {
        const float value=simd_sum(result[t]);
        if(!lane) out[t*N+row]=p[4]?value:bf(value);
    }
}
// MTP's pre-FC normalization spans all four streams. The target RMS kernel
// intentionally only supports <=4096; do not silently route 10240 through it.
kernel void mtp_wide_norm(device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],device float* out [[buffer(2)]],
    constant uint* p [[buffer(3)]],uint gid [[thread_position_in_grid]]) {
    uint t=gid/32,lane=gid%32;if(t>=p[0]) return;
    float square=0;
    for(uint d=lane;d<10240;d+=32) square+=x[t*10240+d]*x[t*10240+d];
    float inv=precise::rsqrt(simd_sum(square)/10240.0f+1e-6f);
    for(uint d=lane;d<10240;d+=32) out[t*10240+d]=bf(bf(x[t*10240+d]*inv)*w[d]);
}

)PACKED_Q8";
        NSString* text=[[NSString alloc] initWithBytes:source.data() length:source.size() encoding:NSUTF8StringEncoding];
        impl_->library=[impl_->device newLibraryWithSource:text options:options error:&error];
        if(!impl_->library) throw std::runtime_error("Metal compilation: "+std::string(error.localizedDescription.UTF8String));
        impl_->pipelines=[NSMutableDictionary new];
    }
}
Metal::~Metal() { if(mtp_scratch::retained_group_owner==this) mtp_scratch::retained_group_owner=nullptr; try { finish(); } catch (...) {} impl_->residency->close(impl_->queue); }
Buf Metal::allocate(uint64_t bytes) {return allocate(bytes,AllocationClass::Temporary);}
Buf Metal::allocate(uint64_t bytes,AllocationClass kind) {
    if(size_t(kind)>size_t(AllocationClass::Workspace)) throw std::invalid_argument("invalid allocation class");
    AllocationTimer timer{impl_->accounting->costs.get(),kind};
    @autoreleasepool {
        const uint64_t charge=(checked_add(bytes,16383)/16384)*16384;
        if(!bytes) throw std::invalid_argument("empty Metal allocation");
        auto& p=*impl_;p.residency->drain();
        if(kind==AllocationClass::Temporary && p.active_scratch>=0) {
            auto& arena=p.scratch[size_t(p.active_scratch)];
            // A large buffer reused for a small request stays pinned until this
            // command group completes. That wasted capacity can strand a later
            // large request even when the requested sizes fit the workspace.
            // Exact physical-size reuse bounds the active prefix by its actual
            // requests; unused shapes remain evictable below.
            for(size_t i=arena.cursor;i<arena.buffers.size();++i) if(arena.buffers[i]->bytes==charge) {
                std::swap(arena.buffers[i],arena.buffers[arena.cursor]);
                const auto& b=arena.buffers[arena.cursor++];++p.pool_reuses;++arena.reuses;
                return std::make_shared<Buffer>(Buffer{bytes,b->data,b->owner,b->metal});
            }
            while(charge<=arena.capacity && arena.bytes>arena.capacity-charge && arena.buffers.size()>arena.cursor) {
                arena.bytes-=arena.buffers.back()->bytes;arena.buffers.pop_back();
            }
            if(charge>arena.capacity || arena.bytes>arena.capacity-charge)
                throw std::runtime_error("temporary workspace capacity exceeded");
            // Bypass this pool during the physical allocation.
            auto b=allocate(charge,AllocationClass::Workspace);
            arena.bytes+=charge;arena.buffers.insert(arena.buffers.begin()+arena.cursor,b);++arena.cursor;
            arena.peak=std::max(arena.peak,arena.bytes);++arena.allocations;
            return std::make_shared<Buffer>(Buffer{bytes,b->data,b->owner,b->metal});
        }
        auto a=impl_->accounting;
        if(!bytes || charge>a->limit || a->live.load()>a->limit-charge)
            throw std::runtime_error("engine memory budget exceeded");
        timer.begin();
        id<MTLBuffer> b=[impl_->device newBufferWithLength:charge options:MTLResourceStorageModeShared];
        if(!b) throw std::runtime_error("Metal allocation failed within engine budget");
        const uint64_t actual=b.allocatedSize;++p.allocations;
        if(actual>a->limit || a->live.load()>a->limit-actual) throw std::runtime_error("actual Metal allocation exceeds budget");
        const auto live=a->live.fetch_add(actual)+actual;
        a->high.store(std::max(live,a->high.load()));
        void* retained=(__bridge_retained void*)b;
        auto registry=p.residency;
        auto owner=std::shared_ptr<void>(retained,[registry,actual,kind](void* pointer){registry->release(pointer,actual,kind);});
        registry->enroll(retained,actual,kind);
        auto result=std::make_shared<Buffer>(Buffer{bytes,static_cast<std::byte*>(b.contents),std::move(owner),retained});
        timer.bytes=actual;return result;
    }
}
Buf Metal::zeros(uint64_t floats) {
    return zeros(floats,AllocationClass::Temporary);
}
Buf Metal::zeros(uint64_t floats,AllocationClass kind) {auto b=allocate(checked_mul(floats,4),kind);std::memset(b->data,0,b->bytes);return b;}
void Metal::residency(const std::string& mode) {
    if(mode!="off" && mode!="core" && mode!="core-cache") throw std::invalid_argument("invalid residency mode");
    if(allocated() || impl_->residency->mode!="off") throw std::logic_error("configure residency before allocating buffers");
    if(mode=="off") return;
    if(@available(macOS 15.0,*)) {
        NSError* error=nil;auto descriptor=[MTLResidencySetDescriptor new];
        auto set=[impl_->device newResidencySetWithDescriptor:descriptor error:&error];
        if(!set) throw std::runtime_error("Metal residency unavailable: "+std::string(error?error.localizedDescription.UTF8String:"unsupported device"));
        impl_->residency->set=set;impl_->residency->mode=mode;
        [set requestResidency];[impl_->queue addResidencySet:set];
    } else throw std::runtime_error("explicit residency requires macOS 15 or later");
}
void Metal::wait(const std::shared_ptr<Completion>& completion) {
    if(!completion) return;
    const auto started=monotonic_ns();const bool blocked=!completion->done.load(std::memory_order_acquire);
    while(!completion->done.load(std::memory_order_acquire)) {
        const auto ticket=impl_->events->ticket();
        if(!completion->done.load(std::memory_order_acquire)) impl_->events->wait(ticket);
    }
    if(blocked) {impl_->host_wait_ns+=monotonic_ns()-started;++impl_->waits;}
    reap();if(!completion->error.empty()) throw std::runtime_error("Metal execution: "+completion->error);
}
void Metal::begin_scratch(size_t slot,uint64_t capacity) {
    if(slot>=2 || impl_->active_scratch>=0 || !capacity) throw std::invalid_argument("invalid scratch workspace");
    auto& arena=impl_->scratch[slot];const auto start=monotonic_ns();wait(arena.last);arena.wait_ns+=monotonic_ns()-start;
    if(arena.bytes>capacity) throw std::runtime_error("scratch pool exceeds requested capacity");
    arena.cursor=0;arena.capacity=capacity;impl_->active_scratch=int(slot);
}
void Metal::end_scratch() {
    if(impl_->active_scratch<0) return;
    const auto slot=size_t(impl_->active_scratch);impl_->active_scratch=-1;
    auto completion=submit();
    // A CPU visibility boundary may have submitted the last work already.
    if(!completion && !impl_->pending.empty()) completion=impl_->pending.back().completion;
    impl_->scratch[slot].last=std::move(completion);
}
void Metal::release_scratch() {
    std::exception_ptr failure;
    try {end_scratch();} catch(...) {failure=std::current_exception();}
    try {finish();} catch(...) {if(!failure) failure=std::current_exception();}
    for(auto& arena:impl_->scratch) {
        arena.buffers.clear();arena.cursor=0;arena.bytes=0;arena.capacity=0;arena.last.reset();
    }
    impl_->residency->drain();
    if(failure) std::rethrow_exception(failure);
}
Buf Metal::upload(std::span<const float> values) {
    auto b=allocate(values.size_bytes()); std::memcpy(b->data,values.data(),values.size_bytes()); return b;
}
void Metal::buffer_diagnostics(bool enabled) {
    if(impl_->allocations) throw std::logic_error("configure buffer diagnostics before allocations");
    impl_->accounting->costs=enabled?std::make_unique<BufferCosts>():nullptr;
}
void Metal::budget(uint64_t bytes) {
    if(!bytes || bytes>22*GiB || bytes<allocated()) throw std::invalid_argument("invalid Metal budget");
    impl_->accounting->limit=bytes;
}
void Metal::copy(const Buf& source,uint64_t from,const Buf& destination,uint64_t to,uint64_t bytes) {
    if(!source || !destination || !bytes || (from|to|bytes)%4 || bytes/4>UINT32_MAX ||
       from>source->bytes || bytes>source->bytes-from || to>destination->bytes || bytes>destination->bytes-to ||
       (source==destination && from<to+bytes && to<from+bytes))
        throw std::invalid_argument("invalid or overlapping GPU copy");
    dispatch("copy_words",{{source,from},{destination,to}},{uint32_t(bytes/4)},uint32_t(bytes/4));
}
Buf Metal::slice(const Buf& source,uint64_t offset,uint64_t bytes) {
    if(!source || offset>source->bytes || bytes>source->bytes-offset) throw std::invalid_argument("GPU slice bounds");
    auto out=allocate(bytes);copy(source,offset,out,0,bytes);return out;
}
void Metal::dispatch(const std::string& name,std::initializer_list<Binding> buffers,
                     std::initializer_list<uint32_t> params,uint32_t x,uint32_t y,uint32_t z,
                     uint32_t tx,uint32_t ty,uint32_t tz) {
    @autoreleasepool {
        const auto encoded_at=monotonic_ns();
        if(!x || !y || !z || !tx || !ty || !tz) throw std::invalid_argument("empty Metal dispatch");
        auto& p=*impl_;
        NSString* key=[NSString stringWithUTF8String:name.c_str()];
        id<MTLComputePipelineState> pipeline=p.pipelines[key];
        if(!pipeline) {
            NSError* error=nil;
            id<MTLFunction> fn=[p.library newFunctionWithName:key];
            if(!fn) throw std::runtime_error("missing Metal kernel: "+name);
            pipeline=[p.device newComputePipelineStateWithFunction:fn error:&error];
            if(!pipeline) throw std::runtime_error("Metal pipeline: "+std::string(error.localizedDescription.UTF8String));
            p.pipelines[key]=pipeline;
        }
        if(uint64_t(tx)*ty*tz>pipeline.maxTotalThreadsPerThreadgroup) throw std::runtime_error("threadgroup exceeds device limit");
        if(pipeline.staticThreadgroupMemoryLength>p.device.maxThreadgroupMemoryLength) throw std::runtime_error("threadgroup memory exceeds device limit");
        if(!p.current) p.current=p.retained_references?[p.queue commandBuffer]:[p.queue commandBufferWithUnretainedReferences];
        if(p.config.counter_profile) {
            if(p.encoder) {[p.encoder endEncoding];p.encoder=nil;}
            if(!p.samples) {
                auto desc=[MTLCounterSampleBufferDescriptor new];desc.counterSet=p.timestamps;
                desc.storageMode=MTLStorageModeShared;desc.sampleCount=4096;
                NSError* error=nil;p.samples=[p.device newCounterSampleBufferWithDescriptor:desc error:&error];
                if(!p.samples) throw std::runtime_error("GPU timestamp allocation failed");
                [p.device sampleTimestamps:&p.sample_cpu_start gpuTimestamp:&p.sample_gpu_start];
            }
            if(p.sample_count>4094 || p.profile_entries>=p.profile_limit())
                throw std::runtime_error("dispatch profile capacity exceeded; use a shorter diagnostic");
            auto desc=[MTLComputePassDescriptor computePassDescriptor];
            desc.sampleBufferAttachments[0].sampleBuffer=p.samples;
            desc.sampleBufferAttachments[0].startOfEncoderSampleIndex=p.sample_count;
            desc.sampleBufferAttachments[0].endOfEncoderSampleIndex=p.sample_count+1;
            p.encoder=[p.current computeCommandEncoderWithDescriptor:desc];
        }
        if(!p.encoder) p.encoder=[p.current computeCommandEncoder];
        [p.encoder setComputePipelineState:pipeline];
        NSUInteger i=0;
        for(const auto& b:buffers) {
            if(!b.buffer || !b.buffer->metal || b.offset>=b.buffer->bytes)
                throw std::invalid_argument("invalid Metal buffer binding");
            [p.encoder setBuffer:(__bridge id<MTLBuffer>)b.buffer->metal offset:b.offset atIndex:i++];
            p.current_buffers.push_back(b.buffer);
        }
        if(params.size()) [p.encoder setBytes:params.begin() length:params.size()*4 atIndex:i];
        [p.encoder dispatchThreads:MTLSizeMake(x,y,z) threadsPerThreadgroup:MTLSizeMake(tx,ty,tz)];
        ++p.dispatches;
        p.kernel_counts[name]=p.kernel_counts.value(name,uint64_t(0))+1;
        const auto duration=monotonic_ns()-encoded_at;p.encode_ns+=duration;
        if(p.config.profile && (!p.config.profile_decode_only || p.request_phase=="decode")) {
            auto family=p.context;family.erase("offset");
            const auto key=p.request_phase+"/"+family.dump()+"/"+name+"/"+p.matrix.dump();
            auto& summary=p.operation_summary[key];
            if(summary.is_null()) summary={{"count",0},{"cpu_encode_ns",0}};
            summary["count"]=summary["count"].get<uint64_t>()+1;
            summary["cpu_encode_ns"]=summary["cpu_encode_ns"].get<uint64_t>()+duration;
            if(p.profile_entries<p.profile_limit()) {
                auto row=p.context;row["kernel"]=name;row["request_phase"]=p.request_phase;
                if(!p.matrix.empty()) row["matrix"]=p.matrix;
                row["encoded_at_ns"]=encoded_at;row["encode_ns"]=duration;
                if(p.config.counter_profile) row["counter_index"]=p.sample_count;
                p.operations.push_back(std::move(row));++p.profile_entries;
            } else p.profile_truncated=true;
        }
        p.matrix=Json::object();
        if(p.config.counter_profile) {[p.encoder endEncoding];p.encoder=nil;p.sample_count+=2;}
    }
}
void Metal::completion_events(std::shared_ptr<CompletionEvents> events) { impl_->events=std::move(events); }
void Metal::command_references(bool retained) {
    if(impl_->current || !impl_->pending.empty()) throw std::logic_error("command ownership requires a drained boundary");
    impl_->retained_references=retained;
}
void Metal::reap() {
    auto& p=*impl_;std::string failure;
    while(!p.pending.empty() && p.pending.front().completion->done.load(std::memory_order_acquire)) {
        const auto& completed=*p.pending.front().completion;
        p.gpu_command_ns+=uint64_t(std::max(0.0,completed.gpu_end-completed.gpu_start)*1e9);
        if(!p.pending.front().completion->error.empty()) failure=p.pending.front().completion->error;
        if(p.config.profile && !p.pending.front().operations.empty()) {
            auto& entry=p.pending.front();
            if(entry.samples) {
                MTLTimestamp cpu_end=0,gpu_end=0;[p.device sampleTimestamps:&cpu_end gpuTimestamp:&gpu_end];
                NSData* data=[entry.samples resolveCounterRange:NSMakeRange(0,entry.sample_count)];
                if(!data || data.length!=entry.sample_count*sizeof(MTLCounterResultTimestamp) ||
                   cpu_end<=entry.cpu_start || gpu_end<=entry.gpu_start)
                    throw std::runtime_error("invalid resolved GPU timestamp data");
                const auto* samples=static_cast<const MTLCounterResultTimestamp*>(data.bytes);
                const double scale=double(cpu_end-entry.cpu_start)/double(gpu_end-entry.gpu_start);
                for(auto& op:entry.operations) {
                    const auto i=op.at("counter_index").get<size_t>();
                    if(i+1>=entry.sample_count || samples[i].timestamp==MTLCounterErrorValue ||
                       samples[i+1].timestamp==MTLCounterErrorValue || samples[i+1].timestamp<samples[i].timestamp)
                        throw std::runtime_error("GPU dispatch timestamp missing or reversed");
                    op["gpu_pass_ns"]=uint64_t(double(samples[i+1].timestamp-samples[i].timestamp)*scale);
                    op["gpu_begin_ticks"]=samples[i].timestamp;op["gpu_end_ticks"]=samples[i+1].timestamp;
                }
            }
            const auto& c=*p.pending.front().completion;
            p.profile.push_back({{"submitted_ns",c.submitted_ns},{"completed_ns",c.completed_ns},
                {"commit_returned_ns",c.commit_returned_ns},
                {"driver_start_seconds",c.driver_start},{"driver_end_seconds",c.driver_end},
                {"gpu_start_seconds",c.gpu_start},{"gpu_end_seconds",c.gpu_end},
                {"operations",std::move(p.pending.front().operations)}});
        }
        {
            GroupRetirementTimer timer(p.accounting->costs.get());
            p.pending.pop_front(); // includes both engine owners and Metal's command-buffer references
        }
    }
    p.residency->drain();
    if(!failure.empty()) throw std::runtime_error("Metal execution: "+failure);
}
std::shared_ptr<Metal::Completion> Metal::submit() {
    // Ending/committing an encoder can create autoreleased Metal objects that
    // retain its buffers, especially with validation enabled. Plain C++ callers
    // have no outer pool. Drain these objects at submission; pending still owns
    // every command buffer and allocation until GPU completion is reaped.
    @autoreleasepool {
    auto& p=*impl_; if(!p.current) {reap();return {};}
    reap();
    // Bound all submission paths, including attention and diagnostics.
    if(p.pending.size()>=2) {const auto t=monotonic_ns();[p.pending.front().cb waitUntilCompleted];p.host_wait_ns+=monotonic_ns()-t;++p.waits;reap();}
    if(p.encoder) { [p.encoder endEncoding]; p.encoder=nil; }
    auto completion=std::make_shared<Completion>();
    const auto events=p.events;
    [p.current addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        completion->completed_ns=monotonic_ns();
        completion->gpu_start=cb.GPUStartTime;completion->gpu_end=cb.GPUEndTime;
        completion->driver_start=cb.kernelStartTime;completion->driver_end=cb.kernelEndTime;
        if(cb.status==MTLCommandBufferStatusError) completion->error=cb.error.localizedDescription.UTF8String;
        mtp_scratch::completion_gate();
        completion->done.store(true,std::memory_order_release);events->publish();
    }];
    p.pending.push_back({p.current,std::move(p.current_buffers),completion,std::move(p.operations),
        p.samples,p.sample_count,p.sample_cpu_start,p.sample_gpu_start});
    p.samples=nil;p.sample_count=0;
    p.operations=Json::array();
    completion->submitted_ns=monotonic_ns();
    [p.current commit];
    // Coordinator-owned; the callback never reads this field. It can finish
    // before commit returns, so this timestamp need not precede GPU completion.
    completion->commit_returned_ns=monotonic_ns();
    p.peak_groups=std::max<uint64_t>(p.peak_groups,p.pending.size());
    p.current_buffers.clear(); p.current=nil; ++p.submissions;
    return completion;
    }
}
void Metal::finish() {
    // Even if a previous command failed, drain every outstanding user.
    std::exception_ptr failure;
    try {submit();} catch(...) {failure=std::current_exception();}
    auto& p=*impl_;
    for(auto& b:p.pending) {const auto t=monotonic_ns();[b.cb waitUntilCompleted];p.host_wait_ns+=monotonic_ns()-t;++p.waits;}
    try {reap();} catch(...) {if(!failure) failure=std::current_exception();}
    if(failure) std::rethrow_exception(failure);
}
namespace {
void validate_affine(const Linear& l) {
    if(!l.input || !l.output || (l.bits!=4 && l.bits!=8) ||
       (l.group!=32 && l.group!=64) || l.input%l.group)
        throw std::invalid_argument("unsupported affine matrix geometry");
    auto covers=[](const Binding& b,uint64_t bytes) {
        return b.buffer && b.offset<=b.buffer->bytes && bytes<=b.buffer->bytes-b.offset;
    };
    const auto weights=checked_mul(l.input,l.output)/(8/l.bits);
    const auto metadata=checked_mul(checked_mul(l.input/l.group,l.output),2);
    if(!covers(l.weight,weights) || !covers(l.scales,metadata) || !covers(l.biases,metadata))
        throw std::invalid_argument("affine tensor binding is truncated");
}
}
Metal::LinearPolicy Metal::linear_policy(const Linear& l,uint32_t tokens,bool fused,bool gathered) const {
    const auto& config=impl_->config;LinearPolicy p;
    if(!config.shape_table.is_null()) {
        for(const auto& r:config.shape_table.at("rules")) if(l.quantized && r["K"]==l.input && r["N"]==l.output &&
            r["rows"]==tokens && r["group"]==l.group && r["bits"]==l.bits && r["fused"]==fused && r["gathered"]==gathered) {
            p={r["tile"].get<uint32_t>(),r.value("output_rows",1u),r.value("gate_pair",false)};break;
        }
    } else p={config.token_tile,config.affine_rows,config.gate_pair};
    if(tokens==1) {p.tile=1;p.rows=std::min(p.rows,2u);}
    else {p.pair=false;if(l.bits!=8 || p.tile<4) p.rows=1;}
    if(fused) p.rows=1;
    return p;
}

// Source-copy replacement of Metal::capture_linear. Capture six tiny inputs,
// hashing existing GPU-visible weight bytes without allocating weight copies.
void Metal::capture_linear(const Linear& l,const Linear* up,const Buf& x,uint32_t tokens,const Buf& rows) {
    auto& p=*impl_;const auto& dir=p.config.operator_capture;
    if(dir.empty() || !l.quantized || l.bits!=8 || l.group!=64 || up || rows || tokens!=4 ||
       p.request_phase!="decode" || p.context.value("offset",0u)!=72) return;
    const auto stage=p.context.value("stage","");const auto layer=p.context.value("layer",-2);
    const bool wanted=(stage=="gdn" && layer==0 &&
        ((l.input==2560 && (l.output==10240 || l.output==6144)) || (l.input==6144 && l.output==2560))) ||
        (stage=="attention" && layer==3 && ((l.input==2560 && l.output==12288) || (l.input==6144 && l.output==2560))) ||
        (stage=="logits" && layer==-1 && l.input==2560 && l.output==248320);
    if(!wanted) return;
    Json matrix={{"K",l.input},{"N",l.output},{"rows",tokens},{"group",l.group},{"bits",l.bits},
        {"fused",false},{"gathered",false}};
    const auto key=stage+"/"+std::to_string(layer)+"/"+matrix.dump();
    if(p.captured_shapes.contains(key)) return;
    const uint64_t input_bytes=uint64_t(tokens)*l.input*4;
    if(p.capture.size()>=6 || x->bytes!=input_bytes || input_bytes>MiB-p.captured_bytes)
        throw std::runtime_error("expanded capture exceeded input bound");
    finish();
    if(p.capture.empty()) {
        if(std::filesystem::exists(dir) && !std::filesystem::is_empty(dir)) throw std::runtime_error("expanded input directory must be new");
        std::filesystem::create_directories(dir);
    }
    auto digest=[](const Binding& binding,uint64_t bytes) {
        if(!binding.buffer || binding.offset>binding.buffer->bytes || bytes>binding.buffer->bytes-binding.offset || bytes>UINT32_MAX)
            throw std::runtime_error("expanded capture tensor bounds");
        unsigned char hash[CC_SHA256_DIGEST_LENGTH];CC_SHA256(binding.buffer->data+binding.offset,CC_LONG(bytes),hash);
        std::string hex;constexpr char digits[]="0123456789abcdef";
        for(auto c:hash){hex+=digits[c>>4];hex+=digits[c&15];}return hex;
    };
    const auto filename=std::to_string(p.capture.size())+"-x.bin";
    std::ofstream stream(dir/filename,std::ios::binary);
    stream.write(reinterpret_cast<const char*>(x->data),std::streamsize(input_bytes));stream.close();
    if(!stream) throw std::runtime_error("cannot save expanded input");
    const uint64_t weight_bytes=uint64_t(l.input)*l.output,meta_bytes=weight_bytes/64*2;
    Json tensors={{"w",{{"bytes",weight_bytes},{"sha256",digest(l.weight,weight_bytes)}}},
        {"s",{{"bytes",meta_bytes},{"sha256",digest(l.scales,meta_bytes)}}},
        {"b",{{"bytes",meta_bytes},{"sha256",digest(l.biases,meta_bytes)}}},
        {"x",{{"file",filename},{"bytes",input_bytes},{"sha256",digest({x},input_bytes)}}}};
    p.capture.push_back({{"matrix",matrix},{"phase",p.request_phase},{"context",p.context},{"tensors",tensors}});
    p.captured_shapes.insert(key);p.captured_bytes+=input_bytes;
    Json manifest={{"kind","q8_expanded_inputs_v1"},{"build_fingerprint",BuildFingerprint},
        {"artifact_revision",p.config.artifact_revision},{"bytes",p.captured_bytes},{"byte_limit",MiB},{"case_limit",6},
        {"weight_payloads_copied",false},{"normal_request_latency_qualified",false},{"cases",p.capture}};
    std::ofstream file(dir/"manifest.json");file<<manifest.dump(2)<<'\n';
    if(!file) throw std::runtime_error("cannot save expanded input manifest");
}

Buf Metal::linear(const Linear& l,const Buf& x,uint32_t tokens,bool float_output) {
    auto out=allocate(checked_mul(checked_mul(tokens,l.output),4));
    linear_into(l,x,tokens,{out},float_output);return out;
}
Json Metal::reference_probe(const Linear& l,const std::atomic<bool>* cancel) {
    if(!l.quantized || l.bits!=8 || l.group!=64 || !l.input || l.input>2560 || !l.output || l.output>6144)
        throw std::invalid_argument("GPU reference requires bounded affine Q8 group64");
    validate_affine(l);
    if(impl_->active_scratch>=0) throw std::logic_error("GPU reference cannot use a live scratch pool");
    auto check=[&] {if(cancel && cancel->load()) throw std::runtime_error("GPU reference cancelled");};
    check();finish();
    const auto baseline=allocated();
    const auto before=process_memory();
    Buf x,out;Json samples=Json::array();std::string checksum;
    uint64_t workspace=0;
    try {
        x=allocate(uint64_t(l.input)*4);out=allocate(uint64_t(l.output)*4);
        workspace=allocated()-baseline;
        if(workspace>48*1024) throw std::logic_error("GPU reference scratch exceeds 48KiB");
        for(uint32_t i=0;i<l.input;++i) x->floats()[i]=float(int(i%17)-8)/16;
        for(int sample=0;sample<4;++sample) {
            check();const auto memory_before=process_memory();const auto start=monotonic_ns();
            const int dispatches=sample?8:1;
            for(int i=0;i<dispatches;++i)
                dispatch("q8_mm",{l.weight,l.scales,l.biases,{x},{out}},
                    {l.input,l.output,1,l.group,0},32*l.output);
            auto completion=submit();wait(completion);reap();
            const auto end=monotonic_ns();
            for(uint32_t i=0;i<l.output;++i) if(!std::isfinite(out->floats()[i]))
                throw std::runtime_error("nonfinite GPU reference output");
            unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(out->data,CC_LONG(l.output*4),digest);
            std::string hash;constexpr char digits[]="0123456789abcdef";
            for(auto c:digest) {hash+=digits[c>>4];hash+=digits[c&15];}
            if(!checksum.empty() && hash!=checksum) throw std::runtime_error("GPU reference output changed");
            checksum=hash;
            const double gs=completion->gpu_start,ge=completion->gpu_end;
            const bool valid=std::isfinite(gs) && std::isfinite(ge) && gs>0 && ge>gs;
            const double submit_seconds=double(completion->submitted_ns)/1e9;
            const bool ordered=valid && gs>=submit_seconds && ge<=double(completion->completed_ns)/1e9;
            samples.push_back({{"sample",sample},{"dispatches",dispatches},{"wall_ns",end-start},
                {"submitted_ns",completion->submitted_ns},{"completed_ns",completion->completed_ns},
                {"gpu_start_seconds",std::isfinite(gs)?Json(gs):Json(nullptr)},
                {"gpu_end_seconds",std::isfinite(ge)?Json(ge):Json(nullptr)},
                {"gpu_ns",valid?Json((ge-gs)*1e9):Json(nullptr)},
                {"submission_delay_ns",ordered?Json((gs-submit_seconds)*1e9):Json(nullptr)},
                {"output_sha256",hash},{"memory_before",memory_before},{"memory_after",process_memory()}});
        }
        finish();x.reset();out.reset();reap();check();
    } catch(...) {
        const auto error=std::current_exception();try {finish();} catch(...) {}
        x.reset();out.reset();std::rethrow_exception(error);
    }
    if(allocated()!=baseline) throw std::logic_error("GPU reference retained allocations");
    Json median=nullptr;std::vector<double> warm;
    for(size_t i=1;i<samples.size();++i) if(samples[i]["gpu_ns"].is_number()) warm.push_back(samples[i]["gpu_ns"].get<double>()/8);
    if(warm.size()==3) {std::sort(warm.begin(),warm.end());median=warm[1];}
    return {{"kernel","q8_mm"},{"input",l.input},{"output",l.output},{"group",l.group},
        {"samples",samples},{"warm_gpu_ns_per_dispatch",median},{"temporary_bytes",workspace},
        {"scratch_accounting","existing transient scratch; disjoint from forward temporaries"},
        {"live_bytes_before",baseline},{"live_bytes_after",allocated()},
        {"live_command_groups",impl_->pending.size()},{"memory_before",before},{"memory_after",process_memory()}};
}
void Metal::linear_into(const Linear& l,const Buf& x,uint32_t tokens,Binding out,bool float_output) {
    if(!tokens || !l.input || !l.output || !x || x->bytes<checked_mul(checked_mul(tokens,l.input),4))
        throw std::invalid_argument("linear input shape");
    if(!l.weight.buffer) throw std::invalid_argument("missing linear weights");
    if(l.quantized) validate_affine(l);
    capture_linear(l,nullptr,x,tokens);
    if(impl_->config.profile) impl_->matrix={{"K",l.input},{"N",l.output},{"rows",tokens},
        {"quantized",l.quantized},{"bits",l.quantized?l.bits:(l.dtype==1?32u:16u)},
        {"format",l.quantized?"affine":l.dtype==1?"F32":l.dtype==2?"F16":"BF16"},
        {"group",l.quantized?l.group:0u},{"fused",false}};
    const auto bytes=checked_mul(checked_mul(tokens,l.output),4);
    if(!out.buffer || out.offset%4 || out.offset>out.buffer->bytes || bytes>out.buffer->bytes-out.offset ||
       out.buffer->metal==x->metal || out.buffer->metal==l.weight.buffer->metal)
        throw std::invalid_argument("invalid or overlapping linear output");
    const auto policy=linear_policy(l,tokens,false,false);const uint32_t tile=policy.tile;
    const auto suffix=policy.rows>1?"_r"+std::to_string(policy.rows)+"_t"+std::to_string(tile):(tile==1?std::string{}:"_t"+std::to_string(tile));
    if(impl_->expanded_q8 && impl_->expanded_scope && impl_->request_phase=="decode" && tokens==4 &&
       tile==4 && policy.rows==1 && l.quantized && l.bits==8 && l.group==64 && l.weight.offset%4==0 &&
       ((l.input==2560 && (l.output==10240 || l.output==6144 || l.output==12288 || l.output==248320)) ||
        (l.input==6144 && l.output==2560))) {
        dispatch("q8_expanded_t4_w8",{l.weight,l.scales,l.biases,{x},out},
            {l.input,l.output,tokens,l.group,uint32_t(float_output)},l.output*32);
    }
    else if(impl_->config.q4_decode=="packed-r2" && impl_->request_phase=="decode" && tokens==1 &&
       l.quantized && l.bits==4 && l.group==64 && l.input==640 && l.output==2560 && l.weight.offset%4==0) {
        dispatch("q4_down_packed_r2",{l.weight,l.scales,l.biases,{x},out},
            {l.input,l.output,tokens,l.group,uint32_t(float_output)},32*(l.output/2));
    }
    else if(l.quantized && l.bits==8 && tokens==1 && impl_->config.q8_decode_rows && l.weight.offset%4==0) {
        const auto rows=impl_->config.q8_decode_rows;
        const auto width=l.input%512==0 && l.output%8==0?8u:4u;
        dispatch("q8_mv_packed_r"+std::to_string(rows)+"_w"+std::to_string(width),{l.weight,l.scales,l.biases,{x},out},
            {l.input,l.output,l.group,uint32_t(float_output)},32*((l.output+rows-1)/rows));
    }
    else if(l.quantized) dispatch(std::string(l.bits==8?"q8_mm":"q4_mm")+suffix,{l.weight,l.scales,l.biases,{x},out},
        {l.input,l.output,tokens,l.group,uint32_t(float_output)},32*((l.output+policy.rows-1)/policy.rows),(tokens+tile-1)/tile);
    else if(float_output && l.dtype==0 && l.input==Hidden && l.output==Experts) {
        auto partial=zeros(uint64_t(tokens)*Experts*16);
        dispatch("router_mm",{l.weight,{x},{partial}},{tokens},(Experts/8)*32,(tokens+7)/8,16);
        dispatch("router_accumulate",{{partial},out},{tokens},tokens*Experts);
    }
    else dispatch("plain_mm",{l.weight,{x},out},
        {l.input,l.output,tokens,l.dtype,uint32_t(float_output)},32*l.output,tokens);
}
void Metal::sparse_select(const Buf& scores,const Buf& mask,const Buf& status,
                          uint32_t tokens,uint32_t offset,uint32_t length) {
    if(!tokens || tokens>256 || uint64_t(offset)+tokens!=length || length>8192 || length<4 ||
       !scores || scores->bytes!=uint64_t(tokens)*(length/4)*4 || !mask || mask->bytes<uint64_t(tokens)*length ||
       !status || status->bytes<4)
        throw std::invalid_argument("invalid GPU sparse selection geometry");
    dispatch("sparse_select",{{scores},{mask},{status}},{tokens,offset,length},256,tokens,1,256);
}
void Metal::attention_scores(const Buf& q,const Buf& keys,const Buf& mask,const Buf& scores,
                             uint32_t tokens,uint32_t offset,uint32_t length,bool sparse) {
    if(!tokens || tokens>256 || uint64_t(offset)+tokens!=length || length>8192 ||
       !q || q->bytes<uint64_t(tokens)*6144*4 || !keys || keys->bytes<uint64_t(length)*512*4 ||
       !mask || (sparse && mask->bytes<uint64_t(tokens)*length) || !scores || scores->bytes<uint64_t(tokens)*24*length*4)
        throw std::invalid_argument("invalid attention score geometry");
    dispatch(sparse && impl_->config.attention_score_tiles=="skip-masked"?"attention_scores_skip_masked":"attention_scores",
        {{q},{keys},{mask},{scores}},{tokens,offset,length,uint32_t(sparse)},((length+7)/8)*32,(tokens+7)/8,24);
}
Buf Metal::gated_linear(const Linear& gate,const Linear& up,const Buf& x,uint32_t tokens,const Buf& rows) {
    if(!gate.quantized || !up.quantized || gate.input!=up.input || gate.output!=up.output || gate.group!=up.group || gate.bits!=up.bits)
        throw std::invalid_argument("fused gate/up requires matching affine matrices");
    validate_affine(gate);validate_affine(up);
    if(!tokens || !x || x->bytes<gate.input*4ull ||
       (rows ? rows->bytes<tokens*4ull : x->bytes<uint64_t(tokens)*gate.input*4))
        throw std::invalid_argument("fused gate/up input shape");
    if(rows) for(uint32_t t=0;t<tokens;++t) {
        const auto row=reinterpret_cast<const int*>(rows->data)[t];
        if(row<0 || checked_mul(uint64_t(row)+1,gate.input*4ull)>x->bytes)
            throw std::invalid_argument("fused gate/up row outside input");
    }
    capture_linear(gate,&up,x,tokens,rows);
    auto out=allocate(uint64_t(tokens)*gate.output*4);
    if(impl_->config.profile) impl_->matrix={{"K",gate.input},{"N",gate.output},{"rows",tokens},{"quantized",true},
        {"format","affine"},{"bits",gate.bits},{"group",gate.group},{"fused",true},{"gathered",bool(rows)}};
    const auto policy=linear_policy(gate,tokens,true,bool(rows));const uint32_t tile=policy.tile;
    if(impl_->config.q4_decode=="packed-r2" && impl_->request_phase=="decode" && tokens==1 && !rows &&
       gate.bits==4 && gate.group==64 && gate.input==2560 && gate.output==640 &&
       gate.weight.offset%4==0 && up.weight.offset%4==0) {
        dispatch("q4_gate_up_packed_r2",{gate.weight,gate.scales,gate.biases,up.weight,up.scales,up.biases,{x},{x},{out}},
            {gate.input,gate.output,tokens,gate.group,0},32*(gate.output/2));
        return out;
    }
    dispatch(std::string(gate.bits==8?"q8_gate_up":"q4_gate_up")+(policy.pair?"_pair":tile==1?"":"_t"+std::to_string(tile)),{gate.weight,gate.scales,gate.biases,up.weight,up.scales,up.biases,{x},{rows?rows:x},{out}},
        {gate.input,gate.output,tokens,gate.group,uint32_t(bool(rows))},gate.output*32,(tokens+tile-1)/tile);
    return out;
}
void Metal::grouped_experts(std::span<const Buf> records,std::span<const uint32_t> positions,
                           const Buf& x,const Buf& out,const Buf& scratch) {
    const auto n=records.size();
    if(!n || n>8 || positions.size()!=n || !x || x->bytes<Hidden*4ull ||
       !out || out->bytes<TopK*Hidden*4ull || !scratch || scratch->bytes<n*Intermediate*4 ||
       x->metal==out->metal || x->metal==scratch->metal || out->metal==scratch->metal)
        throw std::invalid_argument("invalid grouped expert geometry");
    std::array<Buf,8> r;std::array<uint32_t,8> pos{};uint32_t seen=0;
    for(size_t i=0;i<n;++i) {
        if(!records[i] || records[i]->bytes<ExpertBytes || positions[i]>=TopK || (seen&(1u<<positions[i])) ||
           records[i]->metal==out->metal || records[i]->metal==scratch->metal)
            throw std::invalid_argument("invalid grouped expert record or destination");
        seen|=1u<<positions[i];r[i]=records[i];pos[i]=positions[i];
    }
    for(size_t i=n;i<8;++i) r[i]=r[0];
    const auto g=expert_linear(r[0],0),u=expert_linear(r[0],1),d=expert_linear(r[0],2);
    if(!impl_->config.operator_capture.empty()) for(size_t i=0;i<n;++i) {
        auto gate=expert_linear(r[i],0),up=expert_linear(r[i],1);capture_linear(gate,&up,x,1);
    }
    if(impl_->config.profile) impl_->matrix={{"K",Hidden},{"N",Intermediate},{"rows",1},{"bits",4},{"group",64},
        {"quantized",true},{"format","affine"},{"fused",true},{"ready_group",n},{"positions",positions}};
    dispatch("q4_expert_gate_group",{{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]},{x},{scratch}},
        {uint32_t(n),uint32_t(g.weight.offset),uint32_t(g.scales.offset),uint32_t(g.biases.offset),
         uint32_t(u.weight.offset),uint32_t(u.scales.offset),uint32_t(u.biases.offset),uint32_t(linear_policy(g,1,true,false).pair)},Intermediate*32,uint32_t(n));
    if(!impl_->config.operator_capture.empty()) for(size_t i=0;i<n;++i)
        capture_linear(expert_linear(r[i],2),nullptr,slice(scratch,i*Intermediate*4,Intermediate*4),1);
    if(impl_->config.profile) impl_->matrix={{"K",Intermediate},{"N",Hidden},{"rows",1},{"bits",4},{"group",64},
        {"quantized",true},{"format","affine"},{"fused",false},{"ready_group",n},{"positions",positions}};
    dispatch("q4_expert_down_group",{{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]},{scratch},{out}},
        {uint32_t(n),uint32_t(d.weight.offset),uint32_t(d.scales.offset),uint32_t(d.biases.offset),
         pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6],pos[7],linear_policy(d,1,false,false).rows},32*((Hidden+linear_policy(d,1,false,false).rows-1)/linear_policy(d,1,false,false).rows),uint32_t(n));
}
Buf Metal::gdn_scan(const Buf& qkv,const Buf& a,const Buf& b,const Buf& alog,const Buf& dt,
    const Buf& state,uint32_t tokens,uint32_t ad,uint32_t dd) {
    auto out=allocate(uint64_t(tokens)*6144*4);
    if(impl_->config.gdn=="original")
        dispatch("gdn_scan",{{qkv},{a},{b},{alog},{dt},{state},{out}},{tokens,ad,dd},32,128,48,32,4,1);
    else {
        auto gates=allocate(uint64_t(tokens)*48*8);
        dispatch("gdn_prepare",{{a},{b},{alog},{dt},{gates}},{tokens,ad,dd},tokens*48);
        const auto& c=impl_->config;
        const auto name=c.gdn=="precompute"?std::string("gdn_scan_prepared"):
            "gdn_scan_r"+std::to_string(c.gdn_rows)+"_b"+std::to_string(c.gdn_block);
        dispatch(name,{{qkv},{gates},{state},{out}},{tokens},32,128,48,32,c.gdn=="precompute"?4:c.gdn_rows,1);
    }
    return out;
}
Buf Metal::embedding(const Linear& l,std::span<const int> ids,uint32_t copies) {
    if(!l.quantized || ids.empty() || ids.size()>8192 || !copies || copies>4)
        throw std::invalid_argument("embedding geometry");
    validate_affine(l);
    for(auto id:ids) if(id<0 || uint64_t(id)>=l.output) throw std::out_of_range("embedding token outside vocabulary");
    auto tokens=allocate(ids.size_bytes());std::memcpy(tokens->data,ids.data(),ids.size_bytes());
    auto out=allocate(checked_mul(checked_mul(ids.size(),l.input),copies*4ull));
    dispatch("affine_embedding",{l.weight,l.scales,l.biases,{tokens},{out}},
        {l.input,l.input*copies,uint32_t(ids.size()),l.group,l.bits},l.input*copies,uint32_t(ids.size()));
    return out;
}
uint64_t Metal::recommended() const { return impl_->device.recommendedMaxWorkingSetSize; }
uint64_t Metal::physical() const {
    uint64_t bytes=0; size_t len=sizeof(bytes);
    if(sysctlbyname("hw.memsize",&bytes,&len,nullptr,0)) throw std::runtime_error("cannot read physical memory");
    return bytes;
}
uint64_t Metal::allocated() const { return impl_->accounting->live.load(); }
uint64_t Metal::peak() const { return impl_->accounting->high.load(); }
std::string Metal::device_name() const { return impl_->device.name.UTF8String; }
Json Metal::timing_counters() const {
    return {{"buffer_costs",impl_->accounting->costs?impl_->accounting->costs->json():Json(nullptr)},
        {"cpu_encode_ns",impl_->encode_ns},{"cpu_gpu_wait_ns",impl_->host_wait_ns},
        {"gpu_command_ns",impl_->gpu_command_ns},{"submissions",impl_->submissions},{"allocation_count",impl_->allocations},
        {"live_command_groups",impl_->pending.size()},{"live_buffer_bytes",allocated()}};
}
Json Metal::statistics() const {
    Json pools=Json::array();
    for(const auto& a:impl_->scratch) {
        uint64_t unused=0;for(size_t i=a.cursor;i<a.buffers.size();++i) unused+=a.buffers[i]->bytes;
        pools.push_back({{"capacity_bytes",a.capacity},{"allocated_bytes",a.bytes},{"peak_bytes",a.peak},
            {"unused_retained_bytes",unused},{"reuses",a.reuses},{"allocation_count",a.allocations},{"wait_ns",a.wait_ns}});
    }
    return {{"buffer_costs",impl_->accounting->costs?impl_->accounting->costs->json():Json(nullptr)},
        {"device",device_name()},{"physical_bytes",physical()},{"recommended_bytes",recommended()},
        {"build_fingerprint",BuildFingerprint},{"q4_arithmetic","MLX 0.31.1 GEMV BF16 bias sums"},
        {"q8_arithmetic","MLX 0.31.1 fixed per-token QMV, FP32 input bias sums"},
        {"router_arithmetic","M1 SIMD 8x8, sixteen fixed K partitions"},
        {"expert_reduction","eight partial sums in selected-expert order"},
        {"attention_arithmetic","BF16 scores and probabilities, fixed across chunks"},
        {"live_buffer_bytes",allocated()},{"peak_buffer_bytes",peak()},{"scratch_pools",pools},
        {"active_scratch_slot",impl_->active_scratch},
        {"live_command_groups",impl_->pending.size()},{"peak_command_groups",impl_->peak_groups},
        {"residency",impl_->residency->json()},{"kernels",impl_->config.json()},{"kernel_dispatches",impl_->kernel_counts},
        {"cpu_encode_ns",impl_->encode_ns},{"cpu_gpu_wait_ns",impl_->host_wait_ns},
        {"gpu_command_ns",impl_->gpu_command_ns},
        {"allocation_count",impl_->allocations},{"scratch_reuses",impl_->pool_reuses},{"dispatches",impl_->dispatches},{"submissions",impl_->submissions},{"waits",impl_->waits}};
}
Json Metal::memory_counters() const {
    auto result=timing_counters();
    result["peak_buffer_bytes"]=peak();
    result["device_allocated_bytes"]=impl_->device.currentAllocatedSize;
    result["encoded_buffer_references"]=impl_->current_buffers.size();
    result["scratch_bytes"]=impl_->scratch[0].bytes+impl_->scratch[1].bytes;
    result["active_scratch_slot"]=impl_->active_scratch;
    result["scope"]="Engine charges and device resource sizes; not physical residency. Observation does not drain pending users.";
    return result;
}
Json process_memory() {
    task_vm_info_data_t info{}; mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    Json j={{"physical_footprint_bytes",nullptr},{"resident_bytes",nullptr},{"compressed_bytes",nullptr},
        {"compressed_peak_bytes",nullptr},{"decompressions",nullptr},{"page_faults",nullptr},{"pageins",nullptr},
        {"physical_footprint_peak_bytes",nullptr},{"resident_peak_bytes",nullptr}};
    if(task_info(mach_task_self(),TASK_VM_INFO,reinterpret_cast<task_info_t>(&info),&count)==KERN_SUCCESS) {
        if(count>=TASK_VM_INFO_REV0_COUNT) {
            j["resident_bytes"]=info.resident_size;j["compressed_bytes"]=info.compressed;
            j["compressed_peak_bytes"]=info.compressed_peak;
            j["resident_peak_bytes"]=info.resident_size_peak;
        }
        if(count>=TASK_VM_INFO_REV1_COUNT) j["physical_footprint_bytes"]=info.phys_footprint;
        if(count>=TASK_VM_INFO_REV3_COUNT && info.ledger_phys_footprint_peak>=0)
            j["physical_footprint_peak_bytes"]=info.ledger_phys_footprint_peak;
        if(count>=TASK_VM_INFO_REV5_COUNT) j["decompressions"]=uint32_t(info.decompressions);
    }
    task_events_info_data_t events{};count=TASK_EVENTS_INFO_COUNT;
    if(task_info(mach_task_self(),TASK_EVENTS_INFO,reinterpret_cast<task_info_t>(&events),&count)==KERN_SUCCESS && count>=TASK_EVENTS_INFO_COUNT) {
        j["page_faults"]=events.faults;j["pageins"]=events.pageins;
    }
    xsw_usage swap{}; size_t size=sizeof(swap);
    if(!sysctlbyname("vm.swapusage",&swap,&size,nullptr,0)) j["system_swap_used_bytes"]=swap.xsu_used;
    j["reclaimable_bytes"]=available_memory();
    return j;
}
Json host_conditions() {
    @autoreleasepool {
        Json result={{"monotonic_ns",monotonic_ns()},{"thermal_state",nullptr},{"low_power_mode",nullptr},
            {"power_source",nullptr},{"source","NSProcessInfo and IOPowerSources"},
            {"limitation","OS may report nominal/false when thermal or low-power status is unsupported or unknown; no frequency or wattage measurement"}};
        const auto process=[NSProcessInfo processInfo];
        result["thermal_state"]=int(process.thermalState);
        if(@available(macOS 12.0,*)) result["low_power_mode"]=bool(process.lowPowerModeEnabled);
        CFTypeRef info=IOPSCopyPowerSourcesInfo();
        if(info) {
            CFStringRef type=IOPSGetProvidingPowerSourceType(info);
            if(type) result["power_source"]=std::string([(__bridge NSString*)type UTF8String]);
            CFRelease(info);
        }
        return result;
    }
}
Json disk_counters() {
    @autoreleasepool {
        io_iterator_t iterator=0;
        if(IOServiceGetMatchingServices(kIOMainPortDefault,IOServiceMatching("IOBlockStorageDriver"),&iterator)!=KERN_SUCCESS)
            return nullptr;
        Json drives=Json::array();
        while(auto service=IOIteratorNext(iterator)) {
            CFTypeRef raw=IORegistryEntryCreateCFProperty(service,CFSTR("Statistics"),kCFAllocatorDefault,0);
            CFTypeRef protocol=IORegistryEntrySearchCFProperty(service,kIOServicePlane,CFSTR("Protocol Characteristics"),kCFAllocatorDefault,kIORegistryIterateRecursively|kIORegistryIterateParents);
            NSDictionary* stats=raw && CFGetTypeID(raw)==CFDictionaryGetTypeID()?(__bridge NSDictionary*)raw:nil;
            NSDictionary* props=protocol && CFGetTypeID(protocol)==CFDictionaryGetTypeID()?(__bridge NSDictionary*)protocol:nil;
            if([props[@"Physical Interconnect Location"] isEqual:@"Internal"] && stats[@"Bytes (Read)"]) {
                uint64_t registry_id=0;IORegistryEntryGetRegistryEntryID(service,&registry_id);
                drives.push_back({{"registry_id",registry_id},{"read_bytes",[stats[@"Bytes (Read)"] unsignedLongLongValue]},
                    {"write_bytes",[stats[@"Bytes (Write)"] unsignedLongLongValue]}});
            }
            if(raw) CFRelease(raw);if(protocol) CFRelease(protocol);IOObjectRelease(service);
        }
        IOObjectRelease(iterator);
        if(drives.empty()) return nullptr;
        return {{"scope","internal device counters, including other processes"},{"devices",drives}};
    }
}
uint64_t available_memory() {
    vm_statistics64_data_t info{}; mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
    const auto host=mach_host_self();
    const auto status=host_statistics64(host,HOST_VM_INFO64,reinterpret_cast<host_info64_t>(&info),&count);
    mach_port_deallocate(mach_task_self(),host);
    if(status!=KERN_SUCCESS) throw std::runtime_error("cannot measure current memory availability");
    return (uint64_t(info.free_count)+info.purgeable_count+info.external_page_count)*vm_page_size;
}
Json system_memory() {
    vm_statistics64_data_t info{};mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
    const auto host=mach_host_self();
    const auto status=host_statistics64(host,HOST_VM_INFO64,reinterpret_cast<host_info64_t>(&info),&count);
    mach_port_deallocate(mach_task_self(),host);
    if(status!=KERN_SUCCESS) return nullptr;
    return {{"free_bytes",uint64_t(info.free_count)*vm_page_size},
        {"file_backed_bytes",uint64_t(info.external_page_count)*vm_page_size},
        {"anonymous_bytes",uint64_t(info.internal_page_count)*vm_page_size},
        {"wired_bytes",uint64_t(info.wire_count)*vm_page_size},
        {"compressor_bytes",uint64_t(info.compressor_page_count)*vm_page_size},
        {"purgeable_bytes",uint64_t(info.purgeable_count)*vm_page_size},
        {"scope","System-wide overlapping VM categories; includes other processes. Not additive to process footprint."}};
}
Resident::Resident(const Checkpoint& cp,Metal& gpu,int layers,bool streaming) : cp_(cp),gpu_(gpu),streaming_(streaming) {
    std::vector<std::string> keys;
    for(const auto& [key,r]:cp.tensors) { (void)r; if(is_resident(key,layers) && (!streaming || !key.starts_with("model.layers."))) keys.push_back(key); }
    std::sort(keys.begin(),keys.end(),[&](const auto& a,const auto& b){
        const auto& x=cp.at(a); const auto& y=cp.at(b);
        return x.file->name()==y.file->name() ? x.offset<y.offset : x.file->name()<y.file->name();
    });
    for(const auto& key:keys) tensors_[key]=cp.load(key,[&](uint64_t n){return gpu.allocate(n,AllocationClass::Resident);});
}
void Resident::activate_layer(int layer) {
    if(!streaming_) return;
    gpu_.finish();
    std::erase_if(tensors_,[](const auto& pair){return pair.first.starts_with("model.layers.");});
    const auto prefix="model.layers."+std::to_string(layer)+".";
    std::vector<std::string> keys;
    for(const auto& [key,r]:cp_.tensors) { (void)r;if(key.starts_with(prefix) && is_resident(key)) keys.push_back(key); }
    std::sort(keys.begin(),keys.end(),[&](const auto& a,const auto& b){
        const auto& x=cp_.at(a);const auto& y=cp_.at(b);
        return x.file->name()==y.file->name()?x.offset<y.offset:x.file->name()<y.file->name();
    });
    for(const auto& key:keys) tensors_[key]=cp_.load(key,[&](uint64_t bytes){return gpu_.allocate(bytes,AllocationClass::Resident);});
}
const Buf& Resident::at(const std::string& name) const {
    auto i=tensors_.find(name); if(i==tensors_.end()) throw std::runtime_error("missing resident tensor: "+name); return i->second;
}
uint32_t Resident::dtype(const std::string& name) const {
    const auto& d=cp_.at(name).dtype;
    if(d=="BF16") return 0; if(d=="F32") return 1; if(d=="F16") return 2;
    throw std::runtime_error("unsupported arithmetic dtype: "+name);
}
Linear Resident::linear(const std::string& name) const {
    const auto key=name+".weight"; const auto& r=cp_.at(key);
    if(r.shape.size()!=2) throw std::runtime_error("linear weight must be a matrix: "+name);
    Linear l; l.weight={at(key)}; l.output=uint32_t(r.shape[0]);
    l.quantized=cp_.contains(name+".scales");
    if(l.quantized) {
        const auto& q=cp_.config.at("quantization");const auto format=q.value(name,Json::object());
        l.bits=format.value("bits",q.at("bits").get<uint32_t>());
        l.group=format.value("group_size",q.at("group_size").get<uint32_t>());
        if((l.bits!=4 && l.bits!=8) || l.group!=64) throw std::runtime_error("unsupported resident affine format");
    }
    l.input=uint32_t(r.shape[1])*(l.quantized?32/l.bits:1);
    if(l.quantized) {
        if(r.dtype!="U32") throw std::runtime_error("quantized weight must be U32");
        l.scales={at(name+".scales")}; l.biases={at(name+".biases")};
        const auto& s=cp_.at(name+".scales"); const auto& b=cp_.at(name+".biases");
        if(s.dtype!="BF16" || b.dtype!="BF16" || s.shape!=std::vector<uint64_t>{l.output,l.input/l.group} || b.shape!=s.shape)
            throw std::runtime_error("invalid affine scales/biases: "+name);
    } else l.dtype=dtype(key);
    return l;
}
Linear expert_linear(const Buf& record,int projection) {
    if(projection<0 || projection>2 || record->bytes<ExpertBytes) throw std::invalid_argument("invalid expert projection");
    const uint64_t offset=uint64_t(projection)*921600;
    return {{record,offset},{record,offset+819200},{record,offset+870400},
        uint32_t(projection==2?Intermediate:Hidden),uint32_t(projection==2?Hidden:Intermediate),64,0,true};
}
} // namespace freellm::qwen
