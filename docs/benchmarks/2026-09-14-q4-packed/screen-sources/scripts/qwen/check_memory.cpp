#include "qwen/session.hpp"
#include "qwen/memory_trace.hpp"
#include <csignal>
#include <fstream>
#include <print>
#include <thread>

using namespace freellm::qwen;
namespace {std::atomic<bool> stopped=false;void interrupt(int) {stopped=true;}}
int main(int argc,char** argv) {
    if(argc!=7) {std::println(stderr,"usage: qwen_memory_check MODEL PREPARED WORKLOAD MODE REPORT TRACE");return 2;}
    Json report={{"kind","ingestion_memory_diagnostic_v2"},{"complete",false},{"runs",Json::array()},
        {"normal_request_latency_qualified",false},{"production_promoted",false},{"gpu_reference_mode","off"},
        {"gpu_references",Json::array()}};
    auto save=[&] {std::ofstream out(argv[5]);out<<report.dump(2)<<'\n';if(!out) throw std::runtime_error("cannot write memory diagnostic report");};
    std::unique_ptr<MemoryTrace> trace;
    try {
        Options o;o.model=argv[1];o.prepared=argv[2];o.artifact=Artifact::Mixed;
        o.memory=12*GiB;o.context=8192;o.chunk=128;o.panel=512;o.expert_slots=1848;
        o.kernels.policy="candidate";o.kernels.q8_decode_rows=2;o.kernels.route_selection="simd";
        o.decode_scratch=argv[4];o.decode_diagnostics=true;o.validate_decode_scratch();
        const auto work=read_json(argv[3]);
        if(!work.is_array() || work.size()!=2) throw std::invalid_argument("requires initial and append workloads");
        std::signal(SIGINT,interrupt);std::signal(SIGTERM,interrupt);
        trace=std::make_unique<MemoryTrace>(argv[6]);
        o.memory_observer=[&](const Json& where,const Metal& gpu) {trace->capture(where,[&]{return gpu.memory_counters();});};
        report["workloads"]=work;report["model_revision"]=artifact_revision(o.artifact);
        report["sampling"]={{"temperature",o.temperature},{"top_k",o.top_k},{"top_p",o.top_p},{"seed",o.seed}};
        trace->capture({{"event","before_model_load"}}, {},true);save();
        {
            const auto started=monotonic_ns();Model model(o);Tokenizer tokenizer(o.model);
            model.prepare_pipelines();const auto initialization=monotonic_ns()-started;
            const auto checkpoint=[&](const char* event) {trace->capture({{"event",event}},[&]{return model.memory_counters();},true);};
            checkpoint("model_ready");
            if(model.memory_plan().limit!=12*GiB || model.memory_plan().slots!=1848 || model.memory_plan().panel_tokens!=512)
                throw std::runtime_error("memory admission requires at least 12GiB, 1848 slots and panel 512");
            {
                Session session(model,tokenizer);std::vector<int> history;
                for(const auto& task:work) {
                    const auto name=task.at("name").get<std::string>();std::println(stderr,"{} begin",name);
                    auto ids=task.at("tokens").get<std::vector<int>>();
                    if(task.value("append",false)) {history.insert(history.end(),ids.begin(),ids.end());ids=history;}
                    else {session.clear();history=ids;}
                    Options settings=o;settings.max_tokens=task.at("max_tokens");
                    const auto before=model.stats();auto result=session.generate(ids,settings,{},&stopped);
                    if(stopped) throw std::runtime_error("memory diagnostic cancelled");
                    auto row=result.json();row["name"]=name;row["repetition"]=0;row["before"]=before;row["after"]=model.stats();
                    row["initialization_ns"]=initialization;row["profiling_enabled"]=true;row["gpu_reference_mode"]="off";
                    row["runtime_cache_state"]=report["runs"].empty()?"empty_at_process_start":"retained";
                    report["runs"].push_back(row);history.insert(history.end(),result.tokens.begin(),result.tokens.end());
                    checkpoint("request_end");report["trace_coverage"]=trace->summary();save();
                    std::println(stderr,"{} end",name);
                }
                model.diagnostic_drain();checkpoint("users_drained");
            }
            checkpoint("session_destroyed");model.diagnostic_drain();checkpoint("session_users_drained");
        }
        trace->capture({{"event","model_destroyed"}}, {},true);
        const auto destroyed_at=monotonic_ns();
        for(const auto delay:{250,1000}) {
            const auto target=destroyed_at+uint64_t(delay)*1000000;
            const auto now=monotonic_ns();
            if(now<target) std::this_thread::sleep_for(std::chrono::nanoseconds(target-now));
            trace->capture({{"event","after_destroy_"+std::to_string(delay)+"ms"}}, {},true);
        }
        if(stopped) throw std::runtime_error("memory diagnostic cancelled");
        report["trace_coverage"]=trace->summary();report["complete"]=true;report["status"]="captured";save();return 0;
    } catch(const std::exception& e) {
        report["status"]=stopped?"interrupted":"failed";report["error"]=e.what();
        if(trace) try {trace->capture({{"event","failed_after_unwind"}}, {},true);report["trace_coverage"]=trace->summary();} catch(...) {}
        try {save();} catch(...) {}std::println(stderr,"{}",e.what());return 1;
    }
}
