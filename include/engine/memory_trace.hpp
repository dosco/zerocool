#pragma once
#include "engine/metal.hpp"
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

namespace zerocool::engine {
// Diagnostic-only, coordinator-owned. Samples do not submit/reap/wait or keep
// resource owners alive. Lifecycle records have separate admission so a full
// detailed trace cannot hide cleanup. Flushed JSONL survives partial execution.
class MemoryTrace {
public:
    explicit MemoryTrace(const std::filesystem::path& path,size_t limit=512):limit_(limit) {
        if(!limit || limit>2048) throw std::invalid_argument("memory trace limit must be 1..2048");
        const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if(fd<0) throw std::runtime_error("memory trace must be a new writable file");
        stream_=::fdopen(fd,"w");
        if(!stream_) {::close(fd);throw std::runtime_error("cannot open memory trace stream");}
    }
    MemoryTrace(const MemoryTrace&)=delete;
    ~MemoryTrace() {if(stream_) std::fclose(stream_);}
    void capture(Json where,const std::function<Json()>& counters={},bool lifecycle=false) {
        if(lifecycle) {if(++lifecycle_>32) throw std::runtime_error("memory lifecycle trace limit exceeded");}
        else if(++observed_>limit_) return;
        const auto start=monotonic_ns();
        auto process=process_memory();auto system=system_memory();auto metal=counters?counters():Json(nullptr);
        const auto sample_ns=monotonic_ns()-start;sample_ns_+=sample_ns;
        const auto line=Json{{"kind","memory_boundary_v1"},{"sequence",++sequence_},{"monotonic_ns",start},
            {"lifecycle",lifecycle},{"where",std::move(where)},{"process",std::move(process)},
            {"system",std::move(system)},{"metal",std::move(metal)},{"sample_ns",sample_ns}}.dump()+"\n";
        if(line.size()>16384 || bytes_+line.size()>64*MiB) throw std::runtime_error("memory trace byte limit exceeded");
        if(std::fwrite(line.data(),1,line.size(),stream_)!=line.size() || std::fflush(stream_))
            throw std::runtime_error("cannot flush memory trace");
        bytes_+=line.size();
    }
    Json summary() const {
        return {{"kind","memory_trace_coverage_v1"},{"limit",limit_},{"observed",observed_},
            {"captured",std::min(observed_,limit_)},{"omitted",observed_>limit_?observed_-limit_:0},
            {"lifecycle_records",lifecycle_},{"records",sequence_},{"bytes",bytes_},{"sample_ns",sample_ns_},
            {"scope","Ingestion encoding boundaries; no added GPU waits. Sampling time excludes JSON encoding/writes. Process peaks cover process lifetime. No per-buffer physical residency attribution."}};
    }
private:
    std::FILE* stream_=nullptr;
    size_t limit_,observed_=0,lifecycle_=0;
    uint64_t sequence_=0,bytes_=0,sample_ns_=0;
};
} // namespace zerocool::engine
