// Standalone experiment: no runtime selector or native-build change.
// Compile with clang++ -std=c++23 -O2 -fobjc-arc -framework Foundation -framework Metal.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, const char** argv) {
    @autoreleasepool {
        try {
            require(argc == 3, "usage: probe_plain_projection qwen.metal output.json");
            std::ifstream stream(argv[1]);
            require(bool(stream), "cannot read reference shader");
            std::string source((std::istreambuf_iterator<char>(stream)), {});
            auto start=source.find("kernel void plain_mm(");
            require(start != std::string::npos, "reference kernel missing");
            auto end=source.find("\n}", start);
            require(end != std::string::npos, "reference kernel end missing");
            auto candidate=source.substr(start, end+2-start);
            candidate.replace(candidate.find("plain_mm"), 8, "plain_mm_load4");
            const std::string loop="for(uint k=lane;k<K;k+=32) sum+=x[t*K+k]*scalar(w,row*K+k,p[3]);";
            auto position=candidate.find(loop);
            require(position != std::string::npos, "reference arithmetic changed");
            candidate.replace(position,loop.size(),R"(
    uint k=lane;
    for(;k+96<K;k+=128) {
        float a=x[t*K+k], b=x[t*K+k+32], c=x[t*K+k+64], d=x[t*K+k+96];
        float wa=scalar(w,row*K+k,p[3]), wb=scalar(w,row*K+k+32,p[3]);
        float wc=scalar(w,row*K+k+64,p[3]), wd=scalar(w,row*K+k+96,p[3]);
        sum+=a*wa; sum+=b*wb; sum+=c*wc; sum+=d*wd;
    }
    for(;k<K;k+=32) sum+=x[t*K+k]*scalar(w,row*K+k,p[3]);)");
            source+='\n'+candidate;
            auto device=MTLCreateSystemDefaultDevice();
            require(device && device.hasUnifiedMemory, "Apple Silicon GPU unavailable");
            auto queue=[device newCommandQueue];
            auto options=[MTLCompileOptions new];
            if (@available(macOS 15.0,*)) options.mathMode=MTLMathModeSafe;
            else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled=NO;
#pragma clang diagnostic pop
            }
            NSError* error=nil;
            auto library=[device newLibraryWithSource:@(source.c_str()) options:options error:&error];
            if (!library) throw std::runtime_error(error.localizedDescription.UTF8String);
            auto reference=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"plain_mm"] error:&error];
            auto changed=[device newComputePipelineStateWithFunction:[library newFunctionWithName:@"plain_mm_load4"] error:&error];
            require(reference && changed, "pipeline compilation failed");
            NSMutableArray* cases=[NSMutableArray new];
            uint64_t maxBytes=0;
            // Actual narrow-projection dimensions plus an irregular reduction tail.
            for (auto shape : {std::array<uint32_t,3>{10240,4,1},{10240,4,72},{10240,4,128},{10239,5,3},{31,1,1}}) {
                @autoreleasepool {
                    auto [K,N,T]=shape;
                    auto weights=[device newBufferWithLength:K*N*2 options:MTLResourceStorageModeShared];
                    auto inputs=[device newBufferWithLength:K*T*4 options:MTLResourceStorageModeShared];
                    auto control=[device newBufferWithLength:N*T*4 options:MTLResourceStorageModeShared];
                    auto output=[device newBufferWithLength:N*T*4 options:MTLResourceStorageModeShared];
                    require(weights && inputs && control && output, "buffer allocation failed");
                    maxBytes=std::max(maxBytes,uint64_t(weights.allocatedSize+inputs.allocatedSize+control.allocatedSize+output.allocatedSize));
                    uint32_t p[]={K,N,T,0,0}; // BF16 weights; BF16-rounded FP32 inputs.
                    auto dispatch=[&](id<MTLComputePipelineState> pipeline,id<MTLBuffer> destination,uint32_t repeats) {
                        auto command=[queue commandBuffer];
                        auto encoder=[command computeCommandEncoder];
                        [encoder setComputePipelineState:pipeline];
                        [encoder setBuffer:weights offset:0 atIndex:0];
                        [encoder setBuffer:inputs offset:0 atIndex:1];
                        [encoder setBuffer:destination offset:0 atIndex:2];
                        [encoder setBytes:p length:sizeof(p) atIndex:3];
                        for(uint32_t i=0;i<repeats;++i)
                            [encoder dispatchThreads:MTLSizeMake(N*32,T,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
                        [encoder endEncoding];[command commit];[command waitUntilCompleted];
                        require(command.status==MTLCommandBufferStatusCompleted, "GPU command failed");
                        return (command.GPUEndTime-command.GPUStartTime)*1e6/repeats;
                    };
                    for(uint32_t seed=0;seed<8;++seed) {
                        std::mt19937 random(seed);
                        for(uint32_t i=0;i<K*N;++i) {
                            float value=(int32_t(random()%8193)-4096)/4096.0f;
                            ((uint16_t*)weights.contents)[i]=std::bit_cast<uint32_t>(value)>>16;
                        }
                        for(uint32_t i=0;i<K*T;++i) {
                            float value=(int32_t(random()%8193)-4096)/1024.0f;
                            ((float*)inputs.contents)[i]=std::bit_cast<float>(std::bit_cast<uint32_t>(value)&0xffff0000u);
                        }
                        for(uint32_t floating=0;floating<2;++floating) {
                            p[4]=floating;dispatch(reference,control,1);dispatch(changed,output,1);
                            require(std::memcmp(control.contents,output.contents,N*T*4)==0,"candidate changes output bytes");
                        }
                    }
                    p[4]=0;dispatch(reference,control,32);dispatch(changed,output,32);
                    NSMutableArray* pairs=[NSMutableArray new];
                    for(uint32_t pair=0;pair<10;++pair) {
                        double a,b;
                        if(pair%2) {b=dispatch(changed,output,32);a=dispatch(reference,control,32);}
                        else {a=dispatch(reference,control,32);b=dispatch(changed,output,32);}
                        require(a>0 && b>0,"GPU timestamps unavailable");
                        [pairs addObject:@{@"pair":@(pair),@"reference_gpu_us":@(a),@"candidate_gpu_us":@(b),@"ratio":@(b/a)}];
                    }
                    [cases addObject:@{@"K":@(K),@"N":@(N),@"T":@(T),@"exact_checks":@16,@"pairs":pairs}];
                }
            }
            auto result=@{@"kind":@"plain_projection_load4_probe_v1",@"complete":@YES,@"device":device.name,
                @"threads_per_threadgroup":@32,@"dispatches_per_sample":@32,@"paired_repetitions":@10,
                @"metal_validation":@(getenv("MTL_DEBUG_LAYER")!=nullptr || getenv("MTL_SHADER_VALIDATION")!=nullptr),
                @"max_shared_buffer_bytes":@(maxBytes),@"input_kind":@"synthetic finite BF16, eight fixed seeds",
                @"correctness":@"byte exact against unchanged native shader; BF16 and FP32 outputs",
                @"normal_request_latency_qualified":@NO,@"production_promoted":@NO,@"cases":cases};
            auto data=[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:&error];
            require(data && [data writeToFile:@(argv[2]) options:NSDataWritingAtomic error:&error],"cannot write report");
            return 0;
        } catch (const std::exception& error) {
            fprintf(stderr,"%s\n",error.what());return 1;
        }
    }
}
