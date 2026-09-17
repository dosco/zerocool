// Standalone complete-block screen; production shaders and native build stay unchanged.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
static NSString* digest(const void* p,size_t n) {
    check(n<=UINT32_MAX,"hash size exceeds bound");unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256(p,CC_LONG(n),d);
    std::string s;for(auto v:d){s+="0123456789abcdef"[v>>4];s+="0123456789abcdef"[v&15];}return @(s.c_str());
}
static NSDictionary* memory() {
    task_vm_info_data_t v{};mach_msg_type_number_t n=TASK_VM_INFO_COUNT;
    check(task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&v,&n)==KERN_SUCCESS && n>=TASK_VM_INFO_REV5_COUNT,"memory counters unavailable");
    xsw_usage s{};size_t size=sizeof(s);check(!sysctlbyname("vm.swapusage",&s,&size,nullptr,0),"swap counter unavailable");
    return @{@"physical_footprint_bytes":@(v.phys_footprint),@"physical_footprint_peak_bytes":@(v.ledger_phys_footprint_peak),
        @"compressed_bytes":@(v.compressed),@"compressed_peak_bytes":@(v.compressed_peak),
        @"decompressions":@(uint32_t(v.decompressions)),@"system_swap_used_bytes":@(s.xsu_used)};
}
static std::string readText(const char* p) {std::ifstream f(p);check(bool(f),"missing shader");return {(std::istreambuf_iterator<char>(f)),{}};}
struct Buffer {id<MTLBuffer> metal;size_t bytes;void* data()const{return (char*)metal.contents+256;}};
struct Work {Buffer norm,down,low,up,mix,out,projected_inject,injection;};
struct Fixture {Buffer x,norm,dw,ds,db,uw,us,ub,inject;NSData* output;NSData* injection;NSDictionary* origin;Work ref,candidate;};
int main(int argc,const char** argv) {
 @autoreleasepool {try {
    check(argc==6,"usage: probe_hyper_fused REFERENCE_SHADER CANDIDATE_SHADER FIXTURE_DIR REPORT timing|validate");
    bool validation=std::string(argv[5])=="validate";
    check(validation || std::string(argv[5])=="timing","invalid mode");
    check(validation?(getenv("MTL_DEBUG_LAYER") && getenv("MTL_SHADER_VALIDATION")):(!getenv("MTL_DEBUG_LAYER") && !getenv("MTL_SHADER_VALIDATION")),"validation environment differs");
    check(![[NSFileManager defaultManager] fileExistsAtPath:@(argv[4])],"report exists");
    auto baseline=memory();std::string reference=readText(argv[1]),candidate=readText(argv[2]),source=reference+"\n"+candidate;
    auto device=MTLCreateSystemDefaultDevice();check(device && device.hasUnifiedMemory,"unified Metal unavailable");
    auto queue=[device newCommandQueue];auto options=[MTLCompileOptions new];
    if(@available(macOS 15.0,*))options.mathMode=MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled=NO;
#pragma clang diagnostic pop
    }
    NSError* error=nil;auto library=[device newLibraryWithSource:@(source.c_str()) options:options error:&error];
    if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
    std::map<std::string,id<MTLComputePipelineState>> pipelines;
    for(const char* name:{"norm","q8_mv_packed_r2_w8","q8_mv_packed_r2_w4","unary","hc_mix","plain_mm","probe_hyper_up_mix_w4"}) {
        auto p=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@(name)] error:&error];
        check(p!=nil,"pipeline compilation failed");pipelines[name]=p;
    }
    std::vector<Buffer> allocations;uint64_t bytes=0;
    auto allocate=[&](size_t size) {
        check(size && bytes+size+512<=256*1024*1024,"shared buffer admission exceeds 256MiB");
        auto m=[device newBufferWithLength:size+512 options:MTLResourceStorageModeShared];check(m!=nil,"allocation failed");
        bytes+=m.allocatedSize;check(bytes<=256*1024*1024,"actual shared allocations exceed 256MiB");
        std::memset(m.contents,0xa5,size+512);Buffer b{m,size};allocations.push_back(b);return b;
    };
    NSString* directory=@(argv[3]);auto manifestData=[NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"manifest.json"]];
    check(manifestData!=nil,"missing fixture manifest");NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:manifestData options:0 error:&error];
    check([manifest isKindOfClass:[NSDictionary class]] && [manifest[@"kind"] isEqual:@"prepared_hyper_fixtures_v1"] &&
        [manifest[@"artifact_revision"] isEqual:@"b2c422f3c643e36f04227a64d61796b44a4b1029"] && [manifest[@"cases"] count]==4,"fixture identity differs");
    auto read=[&](NSDictionary* e,size_t size) {
        NSString* file=e[@"file"];check(file.length && [file isEqual:file.lastPathComponent] && ![file isEqual:@".."] &&
            [e[@"bytes"] unsignedLongLongValue]==size,"fixture path or bytes differ");
        auto d=[NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:file]];
        check(d && d.length==size && [digest(d.bytes,d.length) isEqual:e[@"sha256"]],"fixture bytes changed");return d;
    };
    auto payload=[&](NSDictionary* e,size_t n) {auto d=read(e,n);auto b=allocate(n);std::memcpy(b.data(),d.bytes,n);return b;};
    auto work=[&](bool fused) {return Work{allocate(40960),allocate(1280),allocate(1280),
        fused?Buffer{}:allocate(40960),fused?Buffer{}:allocate(40960),allocate(10240),allocate(16),allocate(16)};};
    std::vector<Fixture> fixtures;NSMutableArray* immutable=[NSMutableArray new];
    for(NSDictionary* c in manifest[@"cases"]) {
        NSDictionary* t=c[@"tensors"];Fixture f{payload(c[@"input"],40960),payload(t[@"norm"],20480),
            payload(t[@"down_w"],3276800),payload(t[@"down_s"],102400),payload(t[@"down_b"],102400),
            payload(t[@"up_w"],3276800),payload(t[@"up_s"],102400),payload(t[@"up_b"],102400),
            payload(t[@"inject"],81920),read(c[@"native_output"],10240),read(c[@"native_injection"],16),c,work(false),work(true)};
        for(auto b:{f.x,f.norm,f.dw,f.ds,f.db,f.uw,f.us,f.ub,f.inject})[immutable addObject:digest(b.data(),b.bytes)];
        fixtures.push_back(std::move(f));
    }
    auto dispatch=[&](id<MTLComputeCommandEncoder> e,const char* name,std::initializer_list<Buffer> bs,std::initializer_list<uint32_t> ps,uint32_t threads) {
        [e setComputePipelineState:pipelines.at(name)];NSUInteger i=0;
        for(auto b:bs){[e setBuffer:b.metal offset:256 atIndex:i++];}
        if(ps.size())[e setBytes:ps.begin() length:ps.size()*4 atIndex:i];
        [e dispatchThreads:MTLSizeMake(threads,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
    };
    auto encode=[&](id<MTLComputeCommandEncoder> e,Fixture& f,bool changed,bool tail=false) {
        auto& w=changed?f.candidate:f.ref;
        if(!tail) {
            dispatch(e,"norm",{f.x,f.norm,w.norm},{2560,10240,1,0,1},128);
            dispatch(e,"q8_mv_packed_r2_w8",{f.dw,f.ds,f.db,w.norm,w.down},{10240,320,64,0},32*160);
            dispatch(e,"unary",{w.down,w.low},{320,2},320);
        }
        if(changed)dispatch(e,"probe_hyper_up_mix_w4",{f.uw,f.us,f.ub,w.low,w.norm,w.out},{},32*2560);
        else {
            dispatch(e,"q8_mv_packed_r2_w4",{f.uw,f.us,f.ub,w.low,w.up},{320,10240,64,0},32*5120);
            dispatch(e,"unary",{w.up,w.mix},{10240,1},10240);
            dispatch(e,"hc_mix",{w.norm,w.mix,w.out},{1},2560);
        }
        if(!tail) {
            dispatch(e,"plain_mm",{f.inject,w.norm,w.projected_inject},{10240,4,1,0,0},128);
            dispatch(e,"unary",{w.projected_inject,w.injection},{4,3},4);
        }
    };
    const auto began=std::chrono::steady_clock::now();
    auto execute=[&](size_t index,bool changed,uint32_t repeats,bool tail=false) {
      @autoreleasepool {
        check(std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()<30,"GPU stage exceeds 30 seconds");
        auto start=std::chrono::steady_clock::now();auto cmd=[queue commandBuffer];auto e=[cmd computeCommandEncoder];
        for(uint32_t r=0;r<repeats;++r) {
            if(index<4)encode(e,fixtures[index],changed,tail);
            else for(size_t i:{0,1,0,1,0,1,2,3})encode(e,fixtures[i],changed);
        }
        [e endEncoding];const auto encoded=std::chrono::steady_clock::now();[cmd commit];[cmd waitUntilCompleted];
        check(cmd.status==MTLCommandBufferStatusCompleted,"GPU command failed");
        auto now=std::chrono::steady_clock::now();double gpu=(cmd.GPUEndTime-cmd.GPUStartTime)*1e6/repeats;
        check(gpu>0 && std::isfinite(gpu),"missing GPU timing");
        return @{@"wall_us":@(std::chrono::duration<double,std::micro>(now-start).count()/repeats),
            @"encode_us":@(std::chrono::duration<double,std::micro>(encoded-start).count()/repeats),@"gpu_us":@(gpu)};
      }
    };
    auto exact=[&](Fixture& f,bool native) {
        check(!std::memcmp(f.ref.out.data(),f.candidate.out.data(),10240),"fused complete output differs");
        check(!std::memcmp(f.ref.injection.data(),f.candidate.injection.data(),16),"injection differs");
        if(native)check(!std::memcmp(f.ref.out.data(),f.output.bytes,10240) && !std::memcmp(f.ref.injection.data(),f.injection.bytes,16),"standalone reference differs from captured native block");
        for(uint32_t i=0;i<4;++i)check(std::isfinite(((float*)f.ref.injection.data())[i]),"nonfinite injection");
        for(uint32_t i=0;i<2560;++i)check(std::isfinite(((float*)f.ref.out.data())[i]),"nonfinite real output");
    };
    auto intact=[&] {
        size_t index=0;for(auto& f:fixtures)for(auto b:{f.x,f.norm,f.dw,f.ds,f.db,f.uw,f.us,f.ub,f.inject})
            check([digest(b.data(),b.bytes) isEqual:immutable[index++]],"input payload modified");
        for(auto b:allocations)for(size_t i=0;i<256;++i)check(((unsigned char*)b.metal.contents)[i]==0xa5 &&
            ((unsigned char*)b.metal.contents)[256+b.bytes+i]==0xa5,"neighbor guard modified");
    };
    NSMutableArray* cases=[NSMutableArray new];
    for(size_t i=0;i<4;++i) {
        execute(i,false,1);execute(i,true,1);auto& f=fixtures[i];exact(f,true);
        [cases addObject:@{@"origin":f.origin,@"exact":@YES,@"native_reference_exact":@YES,
            @"output_sha256":digest(f.ref.out.data(),10240),@"injection_sha256":digest(f.ref.injection.data(),16)}];
    }
    intact();
    // Supplemental synthetic tail inputs exercise signed zero, BF16 ties,
    // tiny numbers, cancellation and sigmoid saturation. They are not real fixtures.
    uint32_t edgeCases=0;
    for(size_t i=0;i<4;++i)for(uint32_t pattern=0;pattern<3;++pattern) {
        auto& f=fixtures[i];const float values[]={0.0f,-0.0f,0x1p-126f,-0x1p-126f,0x1p-133f,-0x1p-133f,
            1.00390625f,-1.00390625f,16.f,-16.f,80.f,-80.f};
        for(auto w:{f.ref,f.candidate}) {
            for(size_t k=0;k<320;++k)((float*)w.low.data())[k]=pattern==0?0.f:values[(k+pattern)%12];
            for(size_t k=0;k<10240;++k)((float*)w.norm.data())[k]=values[(k*7+pattern)%12];
        }
        execute(i,false,1,true);execute(i,true,1,true);exact(f,false);++edgeCases;
    }
    intact();
    // Reestablish real inputs before timing, warm both complete graphs equally.
    execute(4,false,2);execute(4,true,2);for(auto& f:fixtures)exact(f,true);
    NSMutableArray* pairs=[NSMutableArray new];auto timingBefore=memory();
    if(!validation)for(uint32_t p=0;p<5;++p) {
        printf("complete-block pair %u/5\n",p+1);fflush(stdout);
        for(size_t i=0;i<5;++i) {
            NSMutableArray* arms=[NSMutableArray new];
            for(uint32_t a=0;a<2;++a) {
                bool changed=(p+a)%2;execute(i,changed,2);auto before=memory();auto sample=execute(i,changed,32);
                [arms addObject:@{@"candidate":@(changed),@"sample":sample,@"memory_before":before,@"memory_after":memory()}];
            }
            for(auto& f:fixtures)exact(f,true);
            [pairs addObject:@{@"pair":@(p),@"case":@(i),@"arms":arms}];
        }
    }
    auto timingAfter=memory();intact();
    auto report=@{@"kind":@"hyper_fusion_probe_v1",@"complete":@YES,@"validation":@(validation),@"device":device.name,
        @"reference_shader_sha256":digest(reference.data(),reference.size()),@"candidate_shader_sha256":digest(candidate.data(),candidate.size()),
        @"fixture_manifest_sha256":digest(manifestData.bytes,manifestData.length),@"max_shared_buffer_bytes":@(bytes),
        @"memory_before":baseline,@"memory_after":memory(),@"timing_memory_before":timingBefore,@"timing_memory_after":timingAfter,
        @"gpu_stage_elapsed_seconds":@(std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()),
        @"cases":cases,@"pairs":pairs,@"dispatch_repeats":@32,@"reference_dispatches_per_block":@8,@"candidate_dispatches_per_block":@6,
        @"edge_cases":@(edgeCases),@"neighbor_guards_intact":@YES,@"inputs_unchanged":@YES,
        @"normal_request_latency_qualified":@NO,@"production_promoted":@NO};
    auto data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
    check(data && [data writeToFile:@(argv[4]) options:NSDataWritingAtomic error:&error],"cannot save report");return 0;
 }catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 2;}}
}
