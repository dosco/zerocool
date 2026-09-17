// Isolated real-input Q8 experiment. Does not change the native engine.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
static NSDictionary* memory() {
    task_vm_info_data_t info{}; mach_msg_type_number_t n=TASK_VM_INFO_COUNT;
    check(task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&info,&n)==KERN_SUCCESS &&
        n>=TASK_VM_INFO_REV5_COUNT,"memory counters unavailable");
    xsw_usage swap{};size_t bytes=sizeof(swap);
    check(!sysctlbyname("vm.swapusage",&swap,&bytes,nullptr,0),"swap counter unavailable");
    return @{@"physical_footprint_bytes":@(info.phys_footprint),@"compressed_bytes":@(info.compressed),
        @"decompressions":@(uint32_t(info.decompressions)),@"system_swap_used_bytes":@(swap.xsu_used)};
}
static NSString* digest(const void* data, size_t size) {
    check(size<=UINT32_MAX,"hash input too large");
    unsigned char bytes[CC_SHA256_DIGEST_LENGTH];CC_SHA256(data,CC_LONG(size),bytes);
    std::string out; for (auto b:bytes) {out+="0123456789abcdef"[b>>4];out+="0123456789abcdef"[b&15];}
    return @(out.c_str());
}
static NSDictionary* json(NSString* path) {
    auto bytes=[NSData dataWithContentsOfFile:path];check(bytes!=nil,"missing fixture manifest");
    id value=[NSJSONSerialization JSONObjectWithData:bytes options:0 error:nil];
    check([value isKindOfClass:[NSDictionary class]],"invalid manifest JSON");return value;
}
struct Fixture {
    uint32_t K,N,width;
    id<MTLBuffer> w,s,b,x,control,out;
    NSDictionary* origin;
};

int main(int argc, const char** argv) {
    @autoreleasepool {
        try {
            check(argc==6,"usage: probe_q8_lookahead SHADER GDN_FIXTURES ATTENTION_FIXTURES REPORT timing|validate");
            const bool validate=std::string(argv[5])=="validate";
            check(validate || std::string(argv[5])=="timing","invalid mode");
            check(validate==(getenv("MTL_DEBUG_LAYER")!=nullptr && getenv("MTL_SHADER_VALIDATION")!=nullptr),
                "validation mode differs from environment");
            if (!validate) check(!getenv("MTL_DEBUG_LAYER") && !getenv("MTL_SHADER_VALIDATION"),"timing validation enabled");
            check(![[NSFileManager defaultManager] fileExistsAtPath:@(argv[4])],"report already exists");
            std::ifstream stream(argv[1]);check(bool(stream),"missing shader");
            std::string source((std::istreambuf_iterator<char>(stream)),{});
            // Two consecutive original iterations are loaded before their arithmetic.
            // The dot, bias sum, outer accumulation and SIMD reduction orders stay fixed.
            source+=R"(
template<uint Width>
inline void q8_lookahead(device const uint* w,device const ushort* s,device const ushort* b,
    device const float* x,device float* out,constant uint* p,uint tid) {
    uint row0=(tid/32)*2,lane=tid%32,K=p[0],N=p[1],G=p[2];if(row0>=N)return;
    float result[2]={0,0};
    for(uint base0=lane*Width;base0<K;base0+=64*Width) {
        uint words[2][2][Width/4];float values[2][Width],scales[2][2],biases[2][2];
        #pragma unroll
        for(uint c=0;c<2;++c) {
            uint base=base0+c*32*Width;
            if(base<K) {
                #pragma unroll
                for(uint i=0;i<Width;++i) values[c][i]=x[base+i];
                #pragma unroll
                for(uint r=0;r<2;++r) if(row0+r<N) {
                    #pragma unroll
                    for(uint i=0;i<Width/4;++i) words[c][r][i]=w[(row0+r)*(K/4)+base/4+i];
                    uint g=(row0+r)*(K/G)+base/G;
                    scales[c][r]=b16(s[g]);biases[c][r]=b16(b[g]);
                }
            }
        }
        #pragma unroll
        for(uint c=0;c<2;++c) if(base0+c*32*Width<K) {
            float dot[2]={0,0},sum=0;
            #pragma unroll
            for(uint i=0;i<Width;++i) {
                float v=values[c][i];sum+=v;
                #pragma unroll
                for(uint r=0;r<2;++r) if(row0+r<N) dot[r]+=v*float((words[c][r][i/4]>>(8*(i%4)))&255);
            }
            #pragma unroll
            for(uint r=0;r<2;++r) if(row0+r<N) result[r]+=scales[c][r]*dot[r]+sum*biases[c][r];
        }
    }
    #pragma unroll
    for(uint r=0;r<2;++r) {
        float value=simd_sum(result[r]);if(!lane && row0+r<N)out[row0+r]=p[3]?value:bf(value);
    }
}
kernel void q8_lookahead_w8(device const uint* w [[buffer(0)]],device const ushort* s [[buffer(1)]],
    device const ushort* b [[buffer(2)]],device const float* x [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint tid [[thread_position_in_grid]]) {q8_lookahead<8>(w,s,b,x,out,p,tid);}
kernel void q8_lookahead_w4(device const uint* w [[buffer(0)]],device const ushort* s [[buffer(1)]],
    device const ushort* b [[buffer(2)]],device const float* x [[buffer(3)]],device float* out [[buffer(4)]],
    constant uint* p [[buffer(5)]],uint tid [[thread_position_in_grid]]) {q8_lookahead<4>(w,s,b,x,out,p,tid);}
)";
            auto device=MTLCreateSystemDefaultDevice();check(device && device.hasUnifiedMemory,"unified Metal unavailable");
            auto queue=[device newCommandQueue];auto options=[MTLCompileOptions new];
            if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
            else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled=NO;
#pragma clang diagnostic pop
            }
            NSError* error=nil;
            auto library=[device newLibraryWithSource:@(source.c_str()) options:options error:&error];
            if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
            id<MTLComputePipelineState> reference[2],candidate[2];
            for (uint32_t i=0;i<2;++i) {
                reference[i]=[device newComputePipelineStateWithFunction:[library newFunctionWithName:
                    [NSString stringWithFormat:@"q8_mv_packed_r2_w%u",i?8:4]] error:&error];
                candidate[i]=[device newComputePipelineStateWithFunction:[library newFunctionWithName:
                    [NSString stringWithFormat:@"q8_lookahead_w%u",i?8:4]] error:&error];
                check(reference[i] && candidate[i],"pipeline compilation failed");
            }
            std::vector<Fixture> fixtures;uint64_t allocated=0;
            for(uint32_t directory=2;directory<=3;++directory) {
                NSString* path=@(argv[directory]);auto manifest=json([path stringByAppendingPathComponent:@"manifest.json"]);
                check([manifest[@"kind"] isEqual:@"captured_affine_operators"] &&
                    [manifest[@"artifact_revision"] isEqual:@"b2c422f3c643e36f04227a64d61796b44a4b1029"] &&
                    [manifest[@"cases"] count]==3,"wrong real-model fixture identity");
                for(NSDictionary* c in manifest[@"cases"]) {
                    NSDictionary* m=c[@"matrix"];uint32_t K=[m[@"K"] unsignedIntValue],N=[m[@"N"] unsignedIntValue];
                    check(K && N && K<=10240 && N<=12288 && K%64==0 && [m[@"rows"] intValue]==1 &&
                        [m[@"group"] intValue]==64 && [m[@"bits"] intValue]==8 && ![m[@"fused"] boolValue] &&
                        ![m[@"gathered"] boolValue],"unsupported fixture matrix");
                    auto allocate=[&](uint64_t size) {
                        auto buffer=[device newBufferWithLength:size options:MTLResourceStorageModeShared];
                        check(buffer!=nil,"buffer allocation failed");allocated+=buffer.allocatedSize;
                        check(allocated<=128*1024*1024,"probe exceeds 128MiB allocation bound");return buffer;
                    };
                    auto read=[&](NSString* key,uint64_t expected) {
                        NSDictionary* e=c[@"tensors"][key];NSString* file=e[@"file"];
                        check([file isEqual:file.lastPathComponent] && ![file isEqual:@".."] && file.length &&
                            [e[@"bytes"] unsignedLongLongValue]==expected,"invalid fixture path or size");
                        auto data=[NSData dataWithContentsOfFile:[path stringByAppendingPathComponent:file]];
                        check(data && data.length==expected && [digest(data.bytes,data.length) isEqual:e[@"sha256"]],"fixture hash mismatch");
                        auto buffer=allocate(expected);std::memcpy(buffer.contents,data.bytes,expected);return buffer;
                    };
                    Fixture f{K,N,(K%512==0 && N%8==0)?8u:4u,read(@"w",uint64_t(K)*N),
                        read(@"s",uint64_t(K)*N/64*2),read(@"b",uint64_t(K)*N/64*2),read(@"x",K*4),
                        allocate(N*4),allocate(N*4),@{@"manifest":manifest[@"build_fingerprint"],@"case":c}};
                    fixtures.push_back(std::move(f));
                }
            }
            auto encode=[&](id<MTLComputeCommandEncoder> encoder,const Fixture& f,bool changed,bool floating) {
                [encoder setComputePipelineState:changed?candidate[f.width==8]:reference[f.width==8]];
                const id<MTLBuffer> buffers[]={f.w,f.s,f.b,f.x,changed?f.out:f.control};
                for(uint32_t i=0;i<5;++i)[encoder setBuffer:buffers[i] offset:0 atIndex:i];
                const uint32_t p[]={f.K,f.N,64,uint32_t(floating)};[encoder setBytes:p length:sizeof(p) atIndex:5];
                [encoder dispatchThreads:MTLSizeMake(32*((f.N+1)/2),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            };
            auto execute=[&](size_t index,bool changed,bool floating,uint32_t repeats) {
                auto start=std::chrono::steady_clock::now();auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
                for(uint32_t j=0;j<repeats;++j) {
                    if(index<fixtures.size())encode(encoder,fixtures[index],changed,floating);
                    else for(const auto& f:fixtures)encode(encoder,f,changed,floating);
                }
                [encoder endEncoding];[command commit];[command waitUntilCompleted];
                check(command.status==MTLCommandBufferStatusCompleted,"GPU execution failed");
                const auto wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/repeats;
                double gpu=(command.GPUEndTime-command.GPUStartTime)*1e6/repeats;
                check(gpu>0 && std::isfinite(gpu),"GPU timing missing");
                return @{@"wall_us":@(wall),@"gpu_us":@(gpu)};
            };
            auto before=memory();NSMutableArray* cases=[NSMutableArray new];
            for(size_t i=0;i<fixtures.size();++i) {
                const auto& f=fixtures[i];
                for(bool floating:{false,true}) {
                    execute(i,false,floating,1);execute(i,true,floating,1);
                    check(!std::memcmp(f.control.contents,f.out.contents,f.N*4),"Q8 lookahead changes output bytes");
                    for(uint32_t n=0;n<f.N;++n) check(std::isfinite(((float*)f.out.contents)[n]),"nonfinite fixture output");
                }
                [cases addObject:@{@"origin":f.origin,@"K":@(f.K),@"N":@(f.N),@"width":@(f.width),
                    @"fp32_output_sha256":digest(f.out.contents,f.N*4),@"exact":@YES}];
            }
            NSMutableArray* pairs=[NSMutableArray new];
            for(uint32_t pair=0;pair<(validate?1u:5u);++pair) {
                printf("pair %u\n",pair);fflush(stdout);
                for(size_t i=0;i<=fixtures.size();++i) {
                    NSMutableArray* arms=[NSMutableArray new];
                    for(uint32_t arm=0;arm<2;++arm) {
                        bool changed=(pair+arm)%2;execute(i,changed,false,2);auto a=memory();
                        auto sample=execute(i,changed,false,validate?1:32);
                        [arms addObject:@{@"candidate":@(changed),@"sample":sample,@"memory_before":a,@"memory_after":memory()}];
                    }
                    for(const auto& f:fixtures) check(!std::memcmp(f.control.contents,f.out.contents,f.N*4),"timing changed outputs");
                    [pairs addObject:@{@"pair":@(pair),@"case":@(i),@"arms":arms}];
                }
            }
            auto report=@{@"kind":@"q8_lookahead_probe_v1",@"complete":@YES,@"device":device.name,
                @"shader_sha256":digest(source.data(),source.size()),@"max_shared_buffer_bytes":@(allocated),
                @"memory_before":before,@"memory_after":memory(),@"validation":@(validate),
                @"cases":cases,@"pairs":pairs,@"dispatch_repeats":@(validate?1:32),@"normal_request_latency_qualified":@NO,@"production_promoted":@NO,
                @"limitations":@[@"Old captured activations; current reference shader, original producer identity retained.",
                    @"Six resident matrices; repeated dispatches reuse weight caches. No full-model state or SSD work.",
                    @"One lookahead candidate. Group size and arithmetic fixed. No normal-request speed claim."]};
            auto data=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
            check(data && [data writeToFile:@(argv[4]) options:NSDataWritingAtomic error:&error],"report write failed");
            return 0;
        } catch(const std::exception& error) {fprintf(stderr,"%s\n",error.what());return 2;}
    }
}
