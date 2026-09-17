// Bounded Q4 expert arithmetic experiment; no runtime or artifact changes.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool v,const char* message) {if(!v)throw std::runtime_error(message);}
static NSString* hash(const void* data,size_t size) {
    check(size<=UINT32_MAX,"hash bound exceeded");unsigned char bytes[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data,CC_LONG(size),bytes);std::string out;
    for(auto b:bytes) {out+="0123456789abcdef"[b>>4];out+="0123456789abcdef"[b&15];}return @(out.c_str());
}
static std::string read_source(const char* name) {
    std::ifstream file(name);check(bool(file),"missing shader source");return {(std::istreambuf_iterator<char>(file)),{}};
}
static NSDictionary* memory() {
    task_vm_info_data_t info{};mach_msg_type_number_t count=TASK_VM_INFO_COUNT;
    check(task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&info,&count)==KERN_SUCCESS &&
          count>=TASK_VM_INFO_REV5_COUNT,"memory counters unavailable");
    xsw_usage swap{};size_t bytes=sizeof(swap);
    check(!sysctlbyname("vm.swapusage",&swap,&bytes,nullptr,0),"swap counter unavailable");
    return @{@"physical_footprint_bytes":@(info.phys_footprint),@"compressed_bytes":@(info.compressed),
        @"decompressions":@(uint32_t(info.decompressions)),@"system_swap_used_bytes":@(swap.xsu_used)};
}
struct Fixture {id<MTLBuffer> record,input,hidden[3],output[3];uint32_t layer,expert;};

int main(int argc,const char** argv) {
    @autoreleasepool {
        try {
            check(argc==6,"usage: probe_q4_packed REFERENCE_SHADER CANDIDATE_SHADER FIXTURES REPORT timing|validate");
            const bool validation=std::string(argv[5])=="validate";
            check(validation || std::string(argv[5])=="timing","invalid mode");
            check(validation==(getenv("MTL_DEBUG_LAYER") && getenv("MTL_SHADER_VALIDATION")),"wrong validation mode");
            if(!validation)check(!getenv("MTL_DEBUG_LAYER") && !getenv("MTL_SHADER_VALIDATION"),"timing validation enabled");
            check(![[NSFileManager defaultManager] fileExistsAtPath:@(argv[4])],"report already exists");
            const auto reference=read_source(argv[1]),candidate=read_source(argv[2]),source=reference+'\n'+candidate;
            auto device=MTLCreateSystemDefaultDevice();check(device && device.hasUnifiedMemory,"unified Metal unavailable");
            auto queue=[device newCommandQueue];auto options=[MTLCompileOptions new];
            if(@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
            else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled=NO;
#pragma clang diagnostic pop
            }
            NSError* error=nil;auto library=[device newLibraryWithSource:@(source.c_str()) options:options error:&error];
            if(!library)throw std::runtime_error(error.localizedDescription.UTF8String);
            const char* gates[]={"q4_gate_up","q4_gate_up_pair","probe_q4_gate_r2"};
            const char* downs[]={"q4_mm","q4_mm_r2_t1","probe_q4_down_r2"};
            id<MTLComputePipelineState> gate[3],down[3];
            for(uint32_t i=0;i<3;++i) {
                gate[i]=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@(gates[i])] error:&error];
                down[i]=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@(downs[i])] error:&error];
                check(gate[i] && down[i],"pipeline compilation failed");
            }
            NSString* directory=@(argv[3]);auto data=[NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"manifest.json"]];
            check(data!=nil,"missing fixture manifest");NSDictionary* manifest=[NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            check([manifest[@"kind"] isEqual:@"q3_probe_capture_v2"] && [manifest[@"complete"] boolValue] &&
                  [manifest[@"origin"] isEqual:@"existing-Q4-experts"] &&
                  [manifest[@"artifact_revision"] isEqual:@"b2c422f3c643e36f04227a64d61796b44a4b1029"],"wrong Q4 fixture identity");
            uint64_t allocated=0;
            auto allocate=[&](uint64_t size) {
                auto buffer=[device newBufferWithLength:size options:MTLResourceStorageModeShared];check(buffer!=nil,"allocation failed");
                allocated+=buffer.allocatedSize;check(allocated<=64*1024*1024,"64MiB allocation bound exceeded");return buffer;
            };
            auto load=[&](NSDictionary* entry,uint64_t bytes) {
                NSString* name=entry[@"file"];
                check(name.length && [name isEqual:name.lastPathComponent] && ![name isEqual:@".."] &&
                      [entry[@"bytes"] unsignedLongLongValue]==bytes,"invalid fixture path/length");
                auto raw=[NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:name]];
                check(raw && raw.length==bytes && [hash(raw.bytes,raw.length) isEqual:entry[@"sha256"]],"missing or changed fixture");
                auto buffer=allocate(bytes);std::memcpy(buffer.contents,raw.bytes,bytes);return buffer;
            };
            std::vector<Fixture> fixtures;
            for(uint32_t layer:{0u,16u,32u,47u}) {
                NSDictionary* item=manifest[@"layers"][[NSString stringWithFormat:@"%u",layer]];
                check([item[@"experts"] count]==2 && [item[@"offsets"] count]==8,"missing expert/input coverage");
                auto inputs=load(item[@"inputs"],8*2560*4);
                for(NSDictionary* e in item[@"experts"]) {
                    Fixture f{};f.input=inputs;f.record=load(e[@"record"],2764800);f.layer=layer;f.expert=[e[@"expert"] unsignedIntValue];
                    for(uint32_t v=0;v<3;++v) {f.hidden[v]=allocate(8*640*4);f.output[v]=allocate(8*2560*4);}
                    fixtures.push_back(f);
                }
            }
            auto encode=[&](id<MTLComputeCommandEncoder> encoder,const Fixture& f,uint32_t variant,uint32_t row,bool floating) {
                [encoder setComputePipelineState:gate[variant]];
                const NSUInteger offsets[]={0,819200,870400,921600,1740800,1792000};
                for(uint32_t i=0;i<6;++i)[encoder setBuffer:f.record offset:offsets[i] atIndex:i];
                [encoder setBuffer:f.input offset:row*2560*4 atIndex:6];[encoder setBuffer:f.input offset:0 atIndex:7];
                [encoder setBuffer:f.hidden[variant] offset:row*640*4 atIndex:8];
                const uint32_t gp[]={2560,640,1,64,0};[encoder setBytes:gp length:sizeof(gp) atIndex:9];
                [encoder dispatchThreads:MTLSizeMake(640*32/(variant==2?2:1),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                [encoder setComputePipelineState:down[variant]];
                const NSUInteger dp[]={1843200,2662400,2713600};
                for(uint32_t i=0;i<3;++i)[encoder setBuffer:f.record offset:dp[i] atIndex:i];
                [encoder setBuffer:f.hidden[variant] offset:row*640*4 atIndex:3];[encoder setBuffer:f.output[variant] offset:row*2560*4 atIndex:4];
                const uint32_t p[]={640,2560,1,64,uint32_t(floating)};[encoder setBytes:p length:sizeof(p) atIndex:5];
                [encoder dispatchThreads:MTLSizeMake(2560*32/(variant?2:1),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
            };
            auto execute=[&](uint32_t variant,uint32_t group,bool floating,uint32_t cycles) {
                double gpu_us=0;const auto start=std::chrono::steady_clock::now();
                for(uint32_t c=0;c<cycles;++c) for(uint32_t first=0;first<64;first+=group) {
                    @autoreleasepool {
                        auto command=[queue commandBuffer];auto encoder=[command computeCommandEncoder];
                        for(uint32_t i=first;i<first+group;++i)encode(encoder,fixtures[i%8],variant,i/8,floating);
                        [encoder endEncoding];[command commit];[command waitUntilCompleted];
                        check(command.status==MTLCommandBufferStatusCompleted,"GPU command failed");
                        const double us=(command.GPUEndTime-command.GPUStartTime)*1e6;
                        check(us>0 && std::isfinite(us),"missing GPU time");gpu_us+=us;
                    }
                }
                const auto wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
                return @{@"gpu_us_per_expert":@(gpu_us/(64*cycles)),@"wall_us_per_expert":@(wall/(64*cycles))};
            };
            auto exact=[&] {
                for(const auto& f:fixtures) for(uint32_t v=1;v<3;++v) {
                    check(!std::memcmp(f.hidden[0].contents,f.hidden[v].contents,8*640*4),"gate/up changes output bytes");
                    check(!std::memcmp(f.output[0].contents,f.output[v].contents,8*2560*4),"down chain changes output bytes");
                    for(uint32_t i=0;i<8*2560;++i)check(std::isfinite(((float*)f.output[v].contents)[i]),"nonfinite output");
                }
            };
            for(bool floating:{false,true}) {for(uint32_t v=0;v<3;++v)execute(v,4,floating,1);exact();}
            NSMutableArray* pairs=[NSMutableArray new];
            for(uint32_t pair=0;pair<(validation?1u:5u);++pair) for(uint32_t group:{1u,4u}) {
                printf("pair %u group %u\n",pair,group);fflush(stdout);NSMutableArray* arms=[NSMutableArray new];
                // Alternate primary reference/packed arms. The existing optimization is diagnostic third.
                const uint32_t order[]={pair%2?2u:0u,pair%2?0u:2u,1u};
                for(uint32_t v:order) {
                    execute(v,group,false,1);auto before=memory();auto sample=execute(v,group,false,validation?1:4);
                    [arms addObject:@{@"variant":@(v),@"sample":sample,@"memory_before":before,@"memory_after":memory()}];
                }
                exact();[pairs addObject:@{@"pair":@(pair),@"group":@(group),@"arms":arms}];
            }
            auto report=@{@"kind":@"q4_packed_operator_probe_v1",@"complete":@YES,@"validation":@(validation),
                @"device":device.name,@"reference_shader_sha256":hash(reference.data(),reference.size()),
                @"candidate_shader_sha256":hash(candidate.data(),candidate.size()),@"fixture_manifest":manifest,
                @"max_shared_buffer_bytes":@(allocated),@"cycles":@(validation?1:4),@"pairs":pairs,@"exact":@YES,
                @"expert_input_cases":@64,@"output_modes":@[@"BF16",@"FP32"],
                @"normal_request_latency_qualified":@NO,@"production_promoted":@NO};
            auto bytes=[NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
            check(bytes && [bytes writeToFile:@(argv[4]) options:NSDataWritingAtomic error:&error],"cannot write report");return 0;
        } catch(const std::exception& e) {fprintf(stderr,"%s\n",e.what());return 2;}
    }
}
