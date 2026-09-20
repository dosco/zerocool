// Developer-only target trace. Draft cache traffic is outside the target scope.
#pragma once
#include "block_cache_trace.hpp"

namespace freellm::qwen::horizon_trace {
inline bool enabled = false, active = false;
inline Json forwards = Json::array();
struct ForwardScope {
    ForwardScope() {
        if(active) throw std::logic_error("nested target cache trace");
        active = enabled;
    }
    ~ForwardScope() { active = false; }
    ForwardScope(const ForwardScope&) = delete;
    ForwardScope& operator=(const ForwardScope&) = delete;
};
inline void open(const std::filesystem::path& output, bool capture) {
    enabled = capture;
    if(!enabled) return;
    auto path = output;
    path.replace_extension(".cache.jsonl");
    block_trace::open(path, 1460);
    if(::fcntl(::fileno(block_trace::stream), F_NOCACHE, 1) != 0)
        throw std::runtime_error("cannot disable cache for trace output");
}
inline void forward_begin(uint32_t at, std::span<const int> ids, const std::string& phase) {
    if(active) block_trace::forward_begin(at, ids, phase);
}
inline void layer(int index, uint32_t at, uint32_t count, std::span<const int> routes,
                  const std::vector<int>& selected) {
    if(active) block_trace::layer(index, at, count, routes, selected);
}
inline void acquire(uint64_t key, size_t slot, int64_t victim, int acquisition) {
    if(active) block_trace::acquire(key, slot, victim, acquisition);
}
inline void lease(uint64_t key, unsigned pins, bool release, bool ready) noexcept {
    if(active) block_trace::lease(key, pins, release, ready);
}
inline void snapshot(const Json& state, const Json& stats) {
    if(active) block_trace::snapshot(state, stats);
}
inline void forward_end(uint32_t position, Json routes) {
    forwards.push_back({{"position", position}, {"routes", std::move(routes)}});
    if(active) block_trace::forward_end(position);
}
inline Json finish() { return enabled ? block_trace::finish() : Json(nullptr); }
} // namespace freellm::qwen::horizon_trace
