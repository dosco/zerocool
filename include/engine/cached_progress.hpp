#pragma once
#include "engine/storage.hpp"
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

namespace zerocool::engine {
// Coordinator-only diagnostic. Call at phase boundaries, never inside the
// measured forward interval. Each flushed line survives ordinary cancellation.
class CachedProgress {
public:
    CachedProgress(const std::filesystem::path& path,Json identity,std::string kind="cached_replay_progress_v1") : identity_(std::move(identity)),kind_(std::move(kind)) {
        const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if(fd<0) throw std::runtime_error("cannot create cached progress file (must be new): "+path.string());
        stream_=::fdopen(fd,"w");
        if(!stream_) {::close(fd);throw std::runtime_error("cannot open cached progress stream");}
        started_=phase_started_=monotonic_ns();
    }
    CachedProgress(const CachedProgress&)=delete;
    CachedProgress& operator=(const CachedProgress&)=delete;
    ~CachedProgress() {if(stream_) std::fclose(stream_);}
    void begin(std::string phase,Json details=Json::object()) {
        if(active_ || terminal_) throw std::logic_error("invalid cached progress phase start");
        phase_=std::move(phase);phase_started_=monotonic_ns();active_=true;
        emit("begin",std::move(details));
    }
    void update(Json details) {
        if(!active_ || terminal_) throw std::logic_error("no active cached progress phase");
        emit("progress",std::move(details));
    }
    void end(Json details=Json::object()) {
        if(!active_ || terminal_) throw std::logic_error("no active cached progress phase");
        emit("end",std::move(details));active_=false;
    }
    void finish(const std::string& status,const std::string& error="") {
        if(terminal_ || (status!="complete" && status!="interrupted" && status!="failed") ||
           (status=="complete" && active_)) throw std::logic_error("invalid cached progress termination");
        emit(status.c_str(),{{"phase_incomplete",active_},{"error",error.substr(0,1024)}});terminal_=true;
    }
private:
    void emit(const char* event,Json details) {
        if(++sequence_>10000) throw std::runtime_error("cached progress event limit exceeded");
        const auto now=monotonic_ns();
        const auto line=Json{{"kind",kind_},{"sequence",sequence_},
            {"monotonic_ns",now},{"elapsed_ns",now-started_},{"phase_elapsed_ns",now-phase_started_},
            {"phase",phase_},{"event",event},{"identity",identity_},{"details",std::move(details)}}.dump()+"\n";
        if(line.size()>4096) throw std::runtime_error("cached progress record limit exceeded");
        if(std::fwrite(line.data(),1,line.size(),stream_)!=line.size() || std::fflush(stream_))
            throw std::runtime_error("cannot flush cached progress");
    }
    std::FILE* stream_=nullptr;
    Json identity_;
    std::string phase_,kind_;
    uint64_t started_=0,phase_started_=0,sequence_=0;
    bool active_=false,terminal_=false;
};
} // namespace zerocool::engine
