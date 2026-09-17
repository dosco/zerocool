#include "qwen/metal.hpp"
#include <fstream>
#include <print>
using namespace freellm::qwen;

// Bounded real PLE isolation: use production kernels without loading the
// embedding table, vocabulary projection, or unrelated decoder weights.
int main(int argc,char** argv) {
    try {
        if(argc!=6) throw std::invalid_argument("usage: qwen_ple_replay MODEL ARTIFACT INPUT EMBEDDING OUTPUT_DIR");
        const std::string artifact=argv[2];
        if(artifact!="q4-control" && artifact!="mixed-4_8bit") throw std::invalid_argument("unknown artifact");
        Checkpoint cp(argv[1],true,artifact=="q4-control"?Artifact::Q4:Artifact::Mixed);Metal gpu;
        if(available_memory()<GiB+GiB/2+128*MiB) throw std::runtime_error("insufficient memory for bounded PLE replay");
        gpu.budget(128*MiB);
        auto file=[&](const std::filesystem::path& path) {File f(path);auto b=gpu.allocate(f.size());f.read(0,{b->data,size_t(b->bytes)});return b;};
        auto input=file(argv[3]),embedding=file(argv[4]);
        if(!input->bytes || input->bytes%(Hyper*4) || input->bytes>32ull*Hyper*4 || embedding->bytes!=input->bytes/4)
            throw std::invalid_argument("PLE fixture geometry");
        const uint32_t T=uint32_t(input->bytes/(Hyper*4));
        const std::string base="model.layers.1.ple";
        auto weight=[&](const std::string& name) {return cp.load(name,[&](auto n){return gpu.allocate(n);});};
        auto linear=[&](const std::string& name) {
            const auto fmt=cp.config["quantization"].value(name,cp.config["quantization"]);
            const uint32_t bits=fmt["bits"],group=fmt["group_size"];
            const auto& ref=cp.at(name+".weight");
            Linear l;l.weight={weight(name+".weight")};l.scales={weight(name+".scales")};l.biases={weight(name+".biases")};
            l.quantized=true;l.bits=bits;l.group=group;l.input=uint32_t(ref.shape[1])*32/bits;l.output=uint32_t(ref.shape[0]);
            return gpu.linear(l,embedding,T);
        };
        auto norm=[&](const Buf& x,const std::string& name) {
            auto out=gpu.allocate(x->bytes);
            gpu.dispatch("norm",{{x},{weight(base+"."+name+".weight")},{out}},{Hidden,Hyper,T,0,1},T*4*32);
            return out;
        };
        auto projected=linear(base+".key_proj"),value=linear(base+".value_proj");
        auto key=norm(projected,"norm_key"),query=norm(input,"norm_query"),gated=gpu.allocate(input->bytes);
        gpu.dispatch("ple_gate",{{key},{query},{value},{gated}},{T},640*4,T,1,640);
        auto normalized=norm(gated,"norm_conv"),convolved=gpu.allocate(input->bytes),out=gpu.allocate(input->bytes);
        gpu.dispatch("conv",{{normalized},{gpu.zeros(9*Hyper)},{weight(base+".conv1d.weight")},{convolved}},
                     {Hyper,T,4,3,0},Hyper,T);
        gpu.dispatch("binary",{{gated},{convolved},{out}},{uint32_t(input->bytes/4),0},uint32_t(input->bytes/4));
        gpu.finish();
        const std::filesystem::path directory=argv[5];std::filesystem::create_directories(directory);
        for(const auto& [name,buffer]:std::vector<std::pair<std::string,Buf>>{{"ple.key_proj",projected},{"ple.value_proj",value},
            {"ple.norm_key",key},{"ple.norm_query",query},{"ple.norm_conv.input",gated},{"ple.norm_conv",normalized},{"ple.conv",convolved},{"ple",out}}) {
            std::ofstream f(directory/(name+".bin"),std::ios::binary);f.write(reinterpret_cast<const char*>(buffer->data),buffer->bytes);
            if(!f) throw std::runtime_error("cannot write PLE fixture");
        }
        Json report={{"artifact_revision",cp.revision()},{"tokens",T},{"machine",gpu.statistics()},{"full_model_verified",false}};
        std::ofstream f(directory/"report.json");f<<report.dump(2)<<'\n';if(!f)throw std::runtime_error("cannot write PLE report");
        std::println("Captured production PLE operators with {} bytes peak Metal allocation",gpu.peak());
    } catch(const std::exception& e) {std::println(stderr,"PLE replay: {}",e.what());return 1;}
}
