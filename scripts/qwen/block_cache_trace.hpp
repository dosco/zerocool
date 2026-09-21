// Developer-only coordinator trace. Never included by production sources.
#pragma once
#include "engine/storage.hpp"
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

namespace zerocool::engine::block_trace {
inline std::FILE* stream=nullptr;
inline std::array<char,128*1024> buffer{};
inline uint64_t sequence=0,bytes=0;
inline bool failed=false;
inline uint32_t offset=0,tokens=0;
inline constexpr uint64_t workspace_bytes=32*MiB;

inline void emit(const char* event,Json detail) {
    if(!stream || failed) throw std::runtime_error("cache trace is unavailable");
    const auto line=Json{{"sequence",sequence+1},{"event",event},{"detail",std::move(detail)}}.dump()+"\n";
    if(sequence>=100000 || line.size()>2*MiB || bytes+line.size()>32*MiB) {
        failed=true;throw std::runtime_error("cache trace exceeds bounded capture");
    }
    if(std::fwrite(line.data(),1,line.size(),stream)!=line.size()) {
        failed=true;throw std::runtime_error("cache trace write failed");
    }
    ++sequence;bytes+=line.size();
}
inline void open(const std::filesystem::path& path,uint32_t slots) {
    if(stream) throw std::logic_error("cache trace already open");
    const int fd=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
    if(fd<0) throw std::runtime_error("cache trace must be a new file");
    stream=::fdopen(fd,"w");
    if(!stream) {::close(fd);throw std::runtime_error("cannot open cache trace");}
    if(std::setvbuf(stream,buffer.data(),_IOFBF,buffer.size())) throw std::runtime_error("cannot buffer cache trace");
    emit("begin",{{"kind","qwen_block_cache_events_v1"},{"capacity",slots},{"policy","clock"},
        {"build",build_fingerprint()},{"payload_bytes",ExpertBytes},{"slot_bytes",ExpertStride},
        {"workspace_bound_bytes",workspace_bytes}});
}
inline void forward_begin(uint32_t at,std::span<const int> input,const std::string& phase) {
    offset=at;tokens=uint32_t(input.size());
    emit("forward_begin",{{"offset",at},{"tokens",tokens},{"input",input},{"phase",phase}});
}
inline void layer(int index,uint32_t at,uint32_t count,std::span<const int> routes,const std::vector<int>& selected) {
    emit("layer",{{"layer",index},{"offset",at},{"tokens",count},{"routes",routes},{"selected",selected}});
}
inline void acquire(uint64_t key,size_t slot,int64_t victim,int acquisition) {
    emit("acquire",{{"key",key},{"slot",slot},{"victim",victim},{"acquisition",acquisition}});
}
// Lease destructors and move assignment are noexcept. Preserve native unwinding
// if tracing fails, and reject the capture at the next boundary or finish.
inline void lease(uint64_t key,unsigned pins,bool release,bool ready) noexcept {
    try {emit(release?"release":"pin",{{"key",key},{"pins",pins},{"ready",release?Json(ready):Json(nullptr)}});}
    catch(...) {failed=true;}
}
inline void snapshot(const Json& state,const Json& stats) {emit("snapshot",{{"state",state},{"stats",stats}});}
inline void forward_end(uint32_t position) {emit("forward_end",{{"position",position}});}
inline Json finish() {
    emit("end",{{"complete",true}});
    auto* closing=stream;stream=nullptr;
    const int flushed=std::fflush(closing),closed=std::fclose(closing);
    if(failed || flushed || closed) throw std::runtime_error("cache trace did not finish");
    return {{"complete",true},{"events",sequence},{"bytes",bytes},{"workspace_bound_bytes",workspace_bytes},
        {"performance_measurement",false}};
}
}
