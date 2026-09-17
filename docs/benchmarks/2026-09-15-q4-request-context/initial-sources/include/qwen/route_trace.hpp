#pragma once
#include "qwen/storage.hpp"
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

namespace freellm::qwen {
// Coordinator-only route capture. A route is tentative until forward_commit.
// This diagnostic introduces no GPU synchronization and changes no arithmetic.
class RouteTrace {
public:
    RouteTrace(const std::filesystem::path& path,Json identity) {
        const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if(fd<0) throw std::runtime_error("route trace must be a new file: "+path.string());
        stream_=::fdopen(fd,"w");
        if(!stream_) {::close(fd);throw std::runtime_error("cannot open route trace");}
        try {emit("trace_begin",{{"identity",std::move(identity)}});}
        catch(...) {std::fclose(stream_);stream_=nullptr;throw;}
    }
    RouteTrace(const RouteTrace&)=delete;
    ~RouteTrace() {
        if(stream_) {
            if(!terminal_) try {emit("trace_end",{{"status","incomplete"}});} catch(...) {}
            std::fclose(stream_);
        }
    }
    void request_begin(const std::string& name,std::span<const int> prompt,int max_tokens,bool prime) {
        if(request_ || forward_ || terminal_) throw std::logic_error("route request already active");
        request_=true;++request_id_;
        emit("request_begin",{{"name",name},{"input_token_ids",prompt},{"max_tokens",max_tokens},{"prime",prime}});
    }
    void request_end(std::span<const int> output,const std::string& reason,uint64_t reused) {
        if(!request_ || forward_ || terminal_) throw std::logic_error("route request not complete");
        emit("request_end",{{"output_token_ids",output},{"finish_reason",reason},{"reused_tokens",reused}});
        request_=false;
    }
    void begin(uint64_t session,uint32_t offset,std::span<const int> input,const std::string& phase,size_t slots) {
        if(!request_ || forward_ || terminal_ || input.empty() || offset+input.size()>8192 || !session)
            throw std::logic_error("invalid route forward start");
        forward_=true;++forward_id_;next_layer_=0;offset_=offset;tokens_=input.size();
        emit("forward_begin",{{"session_id",std::to_string(session)},{"offset",offset},{"tokens",tokens_},
            {"input_token_ids",input},{"request_phase",phase},{"expert_slots",slots}});
    }
    void routes(int layer,uint32_t offset,uint32_t tokens,std::span<const int> selected) {
        if(!forward_ || terminal_ || layer!=next_layer_ || layer>=Layers || offset!=offset_ || tokens!=tokens_ || selected.size()!=tokens*TopK)
            throw std::logic_error("route pass does not match active forward");
        emit("routes",{{"layer",layer},{"routes",selected}});++next_layer_;
    }
    void commit(uint32_t position) {
        if(!forward_ || terminal_ || next_layer_!=Layers || position!=offset_+tokens_)
            throw std::logic_error("route forward did not complete all layers");
        emit("forward_commit",{{"position",position}});forward_=false;
    }
    void abort() {
        if(forward_ && !terminal_) {emit("forward_abort",{{"captured_layers",next_layer_}});forward_=false;}
    }
    void cache_reset() {emit("cache_reset",Json::object());}
    void finish() {
        if(request_ || forward_ || terminal_) throw std::logic_error("route trace is incomplete");
        emit("trace_end",{{"status","complete"}});terminal_=true;
    }
private:
    void emit(const char* event,Json detail) {
        if(sequence_>=20000) throw std::runtime_error("route trace event limit exceeded");
        const auto line=Json{{"kind","qwen_route_trace_v1"},{"sequence",sequence_+1},{"monotonic_ns",monotonic_ns()},
            {"event",event},{"request_id",request_id_},{"forward_id",forward_id_},{"details",std::move(detail)}}.dump()+"\n";
        if(line.size()>65536 || bytes_+line.size()>16*MiB) throw std::runtime_error("route trace byte limit exceeded");
        if(std::fwrite(line.data(),1,line.size(),stream_)!=line.size() || std::fflush(stream_))
            throw std::runtime_error("cannot flush route trace");
        ++sequence_;bytes_+=line.size();
    }
    std::FILE* stream_=nullptr;
    uint64_t sequence_=0,bytes_=0,request_id_=0,forward_id_=0;
    uint32_t offset_=0,tokens_=0;
    int next_layer_=0;
    bool request_=false,forward_=false,terminal_=false;
};
}
