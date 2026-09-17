#include "qwen/model.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <fstream>
#include <print>
#include <thread>

using namespace freellm::qwen;
namespace {
std::string hash(std::span<const std::byte> bytes) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes.data(),CC_LONG(bytes.size()),digest);
    std::string result;constexpr char digits[]="0123456789abcdef";
    for(auto c:digest) {result+=digits[c>>4];result+=digits[c&15];}return result;
}
Json snapshot(const State& state,std::span<const float> logits,int layers,const Json& routes) {
    Json result={{"tokens",state.tokens},{"history",state.history},{"valid",state.valid},
                 {"artifact_revision",artifact_revision(state.artifact)},
                 {"logits_sha256",hash(std::as_bytes(logits))},{"routes",routes},{"layers",Json::array()}};
    for(int i=0;i<layers;++i) {
        const auto& l=state.layers[i];Json row={{"position",l.position}};
        const std::array<std::pair<const char*,Buf>,6> buffers={{{"conv",l.conv},{"recurrence",l.recurrence},
            {"keys",l.keys},{"values",l.values},{"index",l.index},{"ple_conv",l.ple_conv}}};
        for(const auto& [name,b]:buffers) if(b) row[name]={{"bytes",b->bytes},{"sha256",hash({b->data,size_t(b->bytes)})}};
        result["layers"].push_back(std::move(row));
    }
    return result;
}
std::vector<float> feed(Model& model,State& state,std::span<const int> tokens) {
    std::vector<float> logits;
    model.prepare_ingest(tokens.size());
    try {
    for(size_t at=0;at<tokens.size();) {
        const auto n=std::min<size_t>(model.input_limit(),tokens.size()-at);
        logits=model.forward(tokens.subspan(at,n),state,model.options().probe_layers==Layers && at+n==tokens.size());at+=n;
    }
    model.finish_ingest();
    } catch(...) {const auto error=std::current_exception();try {model.finish_ingest();} catch(...) {}std::rethrow_exception(error);}
    return logits;
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=5) throw std::invalid_argument("usage: qwen_panel_check MODEL PREPARED CASE_JSON REPORT");
        const auto config=read_json(argv[3]);const auto prefix=config.at("prefix").get<std::vector<int>>();
        const auto append=config.at("append").get<std::vector<int>>(),continuation=config.at("continuation").get<std::vector<int>>();
        const auto panels=config.at("panels").get<std::vector<int>>();
        if(prefix.empty() || append.empty() || continuation.empty() || panels.empty())
            throw std::invalid_argument("requires prefix, append, continuation and a reference panel");
        Options options;options.model=argv[1];if(std::string(argv[2])!="-") options.prepared=argv[2];options.context=config.at("context");
        const auto artifact=config.value("artifact",std::string("q4-control"));
        if(artifact=="mixed-4_8bit") options.artifact=Artifact::Mixed;
        else if(artifact!="q4-control") throw std::invalid_argument("unknown panel fixture artifact");
        options.cache_policy=config.value("cache_policy",std::string("clock"));
        options.audit_routes=true;options.chunk=config.at("chunk");options.memory=uint64_t(config.value("memory_gib",4.0)*GiB);
        options.probe_layers=config.value("layers",Layers);options.expert_slots=config.value("expert_slots",size_t(32));
        options.diagnostic_stream_trunk=config.value("diagnostic_stream_trunk",true);
        options.residency=config.value("residency",std::string("off"));
        options.expert_tail=config.value("expert_tail",std::string("wait"));
        options.decode_submission=config.value("decode_submission",std::string("immediate"));
        options.decode_scratch=config.value("decode_scratch",std::string("none"));
        options.decode_path=config.value("decode_path",std::string("reference"));
        options.prefill_pipeline=config.value("prefill_pipeline",std::string("serial"));
        options.sparse_selection=config.value("sparse_selection",std::string("cpu"));
        options.kernels.attention_score_tiles=config.value("attention_score_tiles",std::string("full"));
        options.kernels.affine_rows=config.value("affine_rows",1u);
        options.kernels.q8_decode_rows=config.value("q8_decode_rows",0u);
        options.kernels.route_selection=config.value("route_selection",std::string("serial"));
        options.kernels.gate_pair=config.value("gate_pair",std::string("off"))=="on";
        options.phase_memory=config.value("phase_memory",std::string("fixed"));
        options.ready_group=config.value("ready_group",size_t(4));
        options.kernels.policy=config.value("kernel_policy",std::string("reference"));
        options.kernels.token_tile=config.value("token_tile",1u);
        options.kernels.gdn=config.value("gdn_path",std::string("original"));
        options.kernels.gdn_rows=config.value("gdn_rows",4u);options.kernels.gdn_block=config.value("gdn_block",8u);
        if(config.contains("shape_policy")) options.kernels.shape_table=read_json(config.at("shape_policy").get<std::string>());
        Json reference,checks=Json::array(),runs=Json::array();bool passed=true;
        auto check=[&](const std::string& name,bool success) {checks.push_back({{"name",name},{"passed",success}});passed&=success;};
        std::vector<int> all=prefix;all.insert(all.end(),append.begin(),append.end());all.insert(all.end(),continuation.begin(),continuation.end());
        if(all.size()>size_t(options.context)) throw std::invalid_argument("case exceeds context");
        for(auto panel:panels) {
            options.panel=panel;Model model(options);auto state=model.make_state();Json stages=Json::array();
            Json probes=Json::array();
            if(config.value("gpu_reference",false)) probes.push_back(model.gpu_reference());
            if(config.value("require_exact_panel",false) && model.memory_plan().panel_tokens!=uint32_t(panel))
                throw std::runtime_error("requested panel differs from admitted panel");
            state.artifact=options.artifact==Artifact::Q4?Artifact::Mixed:Artifact::Q4;
            bool wrong_artifact=false;
            try {model.forward(std::span<const int>(prefix).first(1),state,false);}
            catch(const std::invalid_argument& error) {wrong_artifact=std::string(error.what()).find("state artifact")!=std::string::npos;}
            check("panel_"+std::to_string(panel)+"_rejects_other_artifact_state",wrong_artifact && state.valid && state.tokens==0);
            state.artifact=options.artifact;
            auto logits=feed(model,state,prefix);stages.push_back(snapshot(state,logits,options.probe_layers,model.route_identity()));
            logits=feed(model,state,append);stages.push_back(snapshot(state,logits,options.probe_layers,model.route_identity()));
            for(auto id:continuation) logits=feed(model,state,std::span<const int>(&id,1));
            stages.push_back(snapshot(state,logits,options.probe_layers,model.route_identity()));
            if(config.value("gpu_reference",false)) {
                probes.push_back(model.gpu_reference());
                check("probe_preserves_state",snapshot(state,logits,options.probe_layers,model.route_identity())==stages.back());
            }
            if(reference.is_null()) reference=stages;
            check("panel_"+std::to_string(panel)+"_continued_state",stages==reference);
            if(panel) check("panel_"+std::to_string(panel)+"_executed",model.stats()["passes"]["panel"].get<uint64_t>()>0);
            const auto after_continued=model.stats();
            state=State{};model.reset_expert_cache();state=model.make_state();
            logits=feed(model,state,all);
            check("panel_"+std::to_string(panel)+"_fresh_replay",snapshot(state,logits,options.probe_layers,model.route_identity())==stages.back());
            runs.push_back({{"panel",panel},{"stages",stages},{"continued_statistics",after_continued},{"after_fresh",model.stats()},{"gpu_references",probes}});
        }
        // Deliberately fail after layer-zero recurrent work and expert execution.
        // This exercises the production partial-panel failure path without altering weights.
        const auto dir=std::filesystem::temp_directory_path()/("freellm-panel-fault-"+std::to_string(monotonic_ns()));
        std::filesystem::create_directory(dir);
        struct Cleanup {std::filesystem::path path;~Cleanup(){std::filesystem::remove_all(path);}} cleanup{dir};
        options.panel=1024;options.dependency_trace=dir;
        {
            Model model(options);auto state=model.make_state();bool failed=false;
            const auto n=std::min<size_t>(model.input_limit(),prefix.size());
            if(n<=size_t(options.chunk)) throw std::runtime_error("fault case must execute a panel");
            try {model.forward(std::span<const int>(prefix).first(n),state,false);} catch(const std::runtime_error& e) {
                failed=std::string(e.what()).find("dependency trace")!=std::string::npos;
            }
            check("failed_panel_invalidates_state",failed && !state.valid && state.tokens==0 && state.layers[0].position==n);
            check("failed_panel_drains_gpu",model.stats()["metal"]["live_command_groups"]==0);
            bool refused=false;try {model.forward(std::span<const int>(prefix).first(1),state,false);} catch(const std::invalid_argument&) {refused=true;}
            check("failed_panel_refuses_reuse",refused);model.reset_expert_cache();
        }
        options.dependency_trace=dir/"cancel.jsonl";
        {
            Model model(options);auto state=model.make_state();std::atomic<bool> cancel=false;
            std::atomic<bool> stop=false;
            std::thread watcher([&] {
                while(!stop.load()) {
                    std::error_code error;auto size=std::filesystem::file_size(options.dependency_trace,error);
                    if(!error && size>0) {cancel=true;return;}
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
            struct Join {std::atomic<bool>& stop;std::thread& worker;~Join(){stop=true;if(worker.joinable())worker.join();}} join{stop,watcher};
            bool stopped=false;
            try {model.forward(std::span<const int>(prefix).first(std::min<size_t>(model.input_limit(),prefix.size())),state,false,&cancel);}
            catch(const std::runtime_error& e) {stopped=std::string(e.what()).find("cancelled")!=std::string::npos;}
            stop=true;watcher.join();
            check("cancelled_panel_invalidates_partial_state",stopped && !state.valid && state.tokens==0 && state.layers[0].position>0);
            check("cancelled_panel_drains_gpu",model.stats()["metal"]["live_command_groups"]==0);model.reset_expert_cache();
        }
        if(options.expert_tail=="overlap" || options.decode_scratch=="reuse") {
            const bool scratch=options.decode_scratch=="reuse";
            const std::string lifetime=scratch?"scratch":"tail";
            auto exercised=[&](const Json& stats) {return stats[scratch?"decode_scratch_passes":"expert_tail_deferrals"].get<uint64_t>()>0;};
            auto released=[&](const Json& stats) {
                if(!scratch) return true;
                for(const auto& pool:stats["metal"]["scratch_pools"]) if(pool["allocated_bytes"]!=0) return false;
                return stats["metal"]["active_scratch_slot"]==-1;
            };
            // Fault only after single-token expert work has been submitted and
            // its tail moved into model ownership. Multi-token panel failures
            // above do not exercise this lifetime.
            options.dependency_trace=dir;
            {
                Model model(options);auto state=model.make_state();bool failed=false;
                try {model.forward(std::span<const int>(prefix).first(1),state,false);}
                catch(const std::runtime_error& e) {failed=std::string(e.what()).find("dependency trace")!=std::string::npos;}
                const auto stats=model.stats();
                check("failed_decode_"+lifetime+"_invalidates_state",failed && !state.valid && state.tokens==0 && exercised(stats));
                check("failed_decode_"+lifetime+"_drains_gpu",stats["expert_tail_pending"]==false && stats["metal"]["live_command_groups"]==0 && released(stats));
                model.reset_expert_cache();
            }
            options.dependency_trace=dir/"cancel-tail.jsonl";
            {
                Model model(options);auto state=model.make_state();std::atomic<bool> cancel=false,stop=false;
                std::thread watcher([&] {
                    while(!stop.load()) {
                        std::error_code error;auto size=std::filesystem::file_size(options.dependency_trace,error);
                        if(!error && size>0) {cancel=true;return;}
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                });
                struct Join {std::atomic<bool>& stop;std::thread& worker;~Join(){stop=true;if(worker.joinable())worker.join();}} join{stop,watcher};
                bool stopped=false;
                try {model.forward(std::span<const int>(prefix).first(1),state,false,&cancel);}
                catch(const std::runtime_error& e) {stopped=std::string(e.what()).find("cancelled")!=std::string::npos;}
                stop=true;watcher.join();const auto stats=model.stats();
                check("cancelled_decode_"+lifetime+"_invalidates_state",stopped && !state.valid && state.tokens==0 && exercised(stats));
                check("cancelled_decode_"+lifetime+"_drains_gpu",stats["expert_tail_pending"]==false && stats["metal"]["live_command_groups"]==0 && released(stats));
                model.reset_expert_cache();
            }
        }
        Json report={{"kind","real_panel_state_and_failure_check"},{"passed",passed},{"case",config},{"checks",checks},{"runs",runs},
            {"layers",options.probe_layers},{"full_model",options.probe_layers==Layers},{"performance_qualified",false}};
        std::ofstream out(argv[4]);out<<report.dump(2)<<'\n';if(!out) throw std::runtime_error("cannot write panel report");
        std::println("{} panel state/failure checks: {}",checks.size(),passed);return passed?0:1;
    } catch(const std::exception& e) {std::println(stderr,"panel check: {}",e.what());return 1;}
}
