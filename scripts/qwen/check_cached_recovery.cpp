#include "engine/model.hpp"
#include "cached_recovery_checks.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
using namespace zerocool::engine;

// Explicit real-model check: missing artifacts fail instead of skipping.
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::invalid_argument("usage: qwen_cached_recovery MODEL PREPARED REPORT");
        const std::filesystem::path report_path=argv[3];
        if(std::filesystem::exists(report_path)) throw std::invalid_argument("report already exists");
        const auto trace=std::filesystem::path(report_path.string()+".trace.jsonl");
        if(std::filesystem::exists(trace)) throw std::invalid_argument("trace already exists");
        Options options;
        options.model=argv[1];options.prepared=argv[2];options.dependency_trace=trace;
        options.artifact=Artifact::Mixed;options.memory=12*GiB;
        options.expert_slots=480;options.cached_token_replay=true;options.cached_compare=true;
        options.sparse_selection="gpu";options.kernels.policy="candidate";
        options.kernels.q8_decode_rows=2;options.kernels.attention_score_tiles="skip-masked";
        Model model(options);model.prepare_pipelines();
        const auto before=model.stats();
        const std::array<int,2> tokens={760,369};Json cases=Json::array();
        bool passed=true;
        // Two setup forwards produce 96 layer records. Record 97 belongs to
        // the first warmup control forward, where packed Q8 is disabled.
        for(size_t target:{1u,97u}) {
            {std::ofstream reset(trace);if(!reset) throw std::runtime_error("cannot create trace");}
            model.phase("recovery_reference_setup");
            const auto case_before=model.stats();
            std::atomic<bool> cancel=false,stop=false,reached=false;
            std::atomic<size_t> observed=0;
            std::thread watcher([&] {
                while(!stop.load()) {
                    std::ifstream input(trace);size_t lines=0;std::string line;
                    while(std::getline(input,line)) if(!line.empty() && line.back()=='}') ++lines;
                    if(lines>=target) {observed=lines;reached=true;cancel=true;return;}
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
            struct Join {std::atomic<bool>& stop;std::thread& worker;
                ~Join(){stop=true;if(worker.joinable())worker.join();}} join{stop,watcher};
            std::string failure;
            try {model.cached_token_replay(tokens,5,&cancel);} catch(const std::exception& error) {failure=error.what();}
            stop=true;watcher.join();
            const auto after=model.stats();
            const bool cancelled=reached && failure=="generation cancelled";
            std::ifstream trace_input(trace);const std::string trace_text((std::istreambuf_iterator<char>(trace_input)),{});
            Json window;std::string trace_error;
            try {window=recovery_trace_evidence(trace_text,target,observed.load(),
                    before.at("metal").at("build_fingerprint"),before.at("artifact_revision"));}
            catch(const std::exception& error) {trace_error=error.what();}
            const auto saved_trace=std::filesystem::path(report_path.string()+"."+std::to_string(target)+".trace.jsonl");
            if(std::filesystem::exists(saved_trace)) throw std::runtime_error("recovery evidence already exists");
            {std::ofstream output(saved_trace);output<<trace_text;if(!output) throw std::runtime_error("cannot save recovery trace");}
            unsigned char digest[32];CC_SHA256(trace_text.data(),CC_LONG(trace_text.size()),digest);
            std::string hash;constexpr char digits[]="0123456789abcdef";for(auto v:digest){hash+=digits[v>>4];hash+=digits[v&15];}
            const bool restored=before.at("execution")==after.at("execution") &&
                before.at("metal").at("kernels")==after.at("metal").at("kernels") &&
                model.options().kernels.json()==after.at("metal").at("kernels");
            const bool drained=after.at("metal").at("live_command_groups")==0;
            bool reused=false;
            if(cancelled && trace_error.empty() && restored && drained) {
                auto state=model.make_state();model.forward(tokens,state,false);
                reused=state.valid && state.tokens==tokens.size();
            }
            cases.push_back({{"cancel_after_layer_records",target},{"cancelled",cancelled},{"failure",failure},
                {"configuration_restored",restored},{"gpu_drained",drained},{"reused_model_succeeded",reused},
                {"trace",saved_trace.filename().string()},{"trace_sha256",hash},{"trace_error",trace_error},
                {"window",window},{"before",case_before},{"after_cancel",after}});
            passed=passed && cancelled && trace_error.empty() && restored && drained && reused;
            if(!passed) break;
        }
        Json report={{"kind","cached_comparison_recovery"},{"passed",passed},{"before",before},{"after",model.stats()},
            {"cases",cases},{"normal_request_latency_qualified",false}};
        std::ofstream output(report_path);output<<report.dump(2)<<'\n';
        if(!output) throw std::runtime_error("cannot write recovery report");
        std::cout<<cases.dump()<<'\n';return passed?0:1;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
