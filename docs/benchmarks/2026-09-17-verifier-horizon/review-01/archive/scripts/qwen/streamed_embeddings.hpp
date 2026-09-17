// Developer-only exact token-row storage. Target and MTP share one bounded cache.
#pragma once
#include "qwen/metal.hpp"
#include <cstring>
#include <memory>

namespace freellm::qwen::embedding_rows {
constexpr uint64_t Reserve=2*MiB;
constexpr size_t Capacity=256,MaxRowBytes=2720;
inline thread_local bool enabled=false;
inline void need(bool ok,const char* message) {if(!ok) throw std::runtime_error(message);}
struct Store {
    struct Row {int token=-1;std::array<std::byte,MaxRowBytes> bytes{};};
    std::array<TensorRef,3> tensors;
    std::array<uint64_t,3> strides{};
    std::array<Row,Capacity> rows;
    size_t next=0;uint32_t bits=0;
    uint64_t hits=0,misses=0,evictions=0,read_bytes=0;
    explicit Store(const Checkpoint& cp) {
        const auto& q=cp.config.at("quantization");const auto format=q.value("model.embed_tokens",Json::object());
        bits=format.value("bits",q.at("bits").get<uint32_t>());
        need((bits==4 || bits==8) && format.value("group_size",q.at("group_size").get<uint32_t>())==64,"unsupported row quantization");
        const std::array<std::string,3> suffix={"weight","scales","biases"};
        for(size_t i=0;i<3;++i) {
            tensors[i]=cp.at("model.embed_tokens."+suffix[i]);const auto& r=tensors[i];
            const std::vector<uint64_t> shape={Vocab,uint64_t(i==0?Hidden/(32/bits):Hidden/64)};
            need(r.shape==shape && r.dtype==(i==0?"U32":"BF16"),"unsupported row tensor geometry");
            strides[i]=r.row_bytes();
            need(strides[i]==uint64_t(i==0?Hidden/(8/bits):Hidden/64*2) && r.bytes==Vocab*strides[i],"truncated embedding tensor");
        }
        need(row_bytes()<=MaxRowBytes && sizeof(Store)<Reserve,"row cache exceeds fixed allowance");
    }
    uint64_t row_bytes() const {return strides[0]+strides[1]+strides[2];}
    uint64_t removed_allocation() const {
        uint64_t n=0;for(const auto& t:tensors)n+=(t.bytes+16383)/16384*16384;return n;
    }
    Linear descriptor() const {Linear l;l.input=Hidden;l.output=Vocab;l.group=64;l.bits=bits;l.quantized=true;return l;}
    const Row& get(int token) {
        need(token>=0 && token<Vocab,"embedding row outside vocabulary");
        for(const auto& r:rows) if(r.token==token) {++hits;return r;}
        // Complete all reads before replacing a live cached row. A failed read
        // must not publish partial bytes under either the old or new token ID.
        Row loaded;size_t at=0;
        for(size_t i=0;i<3;++i) {
            const auto& r=tensors[i];r.file->read(r.offset+uint64_t(token)*strides[i],{loaded.bytes.data()+at,size_t(strides[i])});
            at+=strides[i];read_bytes+=strides[i];
        }
        loaded.token=token;if(rows[next].token>=0)++evictions;
        rows[next]=loaded;const auto current=next;next=(next+1)%Capacity;++misses;return rows[current];
    }
    Buf gather(Metal& gpu,const Linear& requested,std::span<const int> ids,uint32_t copies) {
        need(!requested.weight.buffer && requested.quantized && requested.bits==bits && requested.group==64 &&
             requested.input==Hidden && requested.output==Vocab && !ids.empty() && ids.size()<=128 && copies>=1 && copies<=4,
             "unsupported streamed embedding request");
        for(auto id:ids)need(id>=0 && id<Vocab,"embedding row outside vocabulary");
        auto compact=descriptor();compact.output=uint32_t(ids.size());
        compact.weight={gpu.allocate(ids.size()*strides[0])};compact.scales={gpu.allocate(ids.size()*strides[1])};
        compact.biases={gpu.allocate(ids.size()*strides[2])};
        std::array<int,128> local{};
        for(size_t t=0;t<ids.size();++t) {
            const auto& row=get(ids[t]);size_t offset=0;
            for(size_t i=0;i<3;++i) {
                const auto& binding=i==0?compact.weight:i==1?compact.scales:compact.biases;
                std::memcpy(binding.buffer->data+t*strides[i],row.bytes.data()+offset,strides[i]);offset+=strides[i];
            }
            local[t]=int(t);
        }
        // Per-call buffers retain their bytes through GPU completion. CPU cache
        // eviction cannot mutate an in-flight gather or its resulting embedding.
        return gpu.embedding(compact,std::span(local).first(ids.size()),copies);
    }
    Json stats() const {return {{"storage","exact-packed-rows"},{"capacity",Capacity},{"host_reserve_bytes",Reserve},
        {"fixed_host_bytes",sizeof(Store)},{"removed_resident_allocation_bytes",removed_allocation()},{"row_bytes",row_bytes()},
        {"hits",hits},{"misses",misses},{"evictions",evictions},{"application_read_bytes",read_bytes}};}
};
inline thread_local std::unique_ptr<Store> store;
struct Scope {
    Scope() {need(!enabled && !store,"nested streamed embedding owner");enabled=true;}
    ~Scope(){store.reset();enabled=false;}
    Scope(const Scope&)=delete;Scope& operator=(const Scope&)=delete;
};
inline uint64_t planned_resident(const Checkpoint& cp,uint64_t original) {
    if(!enabled)return original;
    Store geometry(cp);need(original>=geometry.removed_allocation(),"missing resident embedding allocation");
    return original-geometry.removed_allocation()+Reserve;
}
} // namespace freellm::qwen::embedding_rows
