#pragma once
// Developer-only trained MTP path. Not included by the production build.
#include "engine/model.hpp"
#include "engine/pipeline.hpp"

namespace zerocool::engine {
struct DraftAccess {
    static Metal& gpu(Model& m) { return m.gpu_; }
    static Resident& resident(Model& m) { return *m.resident_; }
    static ReadPool& reads(Model& m) { return m.reads_; }
};
// The source-copy target writes a bounded, detached buffer before its final
// mixer. It never retains scratch-backed views across forward calls.
extern thread_local Buf mtp_target_hidden;
extern thread_local uint32_t mtp_target_rows,mtp_target_offset;
void capture_mtp_hidden(Metal&,const Buf&,uint32_t,uint32_t);

struct DraftState {
    Buf keys,values,index;
    uint32_t position=0;
    bool valid=true;
};
struct DraftOutput { Buf wide,mixed,logits; };
class DraftCheckpoint {
    std::array<Buf,3> tail_;
    std::array<uint64_t,3> geometry_{};
    uint32_t position_=0,width_=0;
    bool saved_=false;
public:
    explicit DraftCheckpoint(Metal&);
    void save(Metal&,const DraftState&,uint32_t width);
    void restore(Metal&,DraftState&) const;
};

class MtpDraft {
public:
    static constexpr uint32_t Panel=128;
    static uint64_t budget_bytes(size_t slots,uint32_t context);
    MtpDraft(Metal&,ReadPool&,const std::filesystem::path&,Linear embedding,Linear head,
             size_t slots=128,uint32_t context=8192);
    ~MtpDraft();
    DraftState make_state();
    // hidden[t] is the preceding target/draft token's wide output; ids[t]
    // is the next token. Positions stay aligned with the preceding hidden
    // (shift token IDs, not positions), as in the reference MTP proposer.
    DraftOutput forward(std::span<const int> ids,const Buf& hidden,DraftState&,
                        bool logits=true,const std::atomic<bool>* cancel=nullptr);
    void catch_up(std::span<const int> ids,const Buf& hidden,DraftState&,
                  const std::atomic<bool>* cancel=nullptr);
    Json stats() const;
    std::function<void(const std::string&,const Buf&)> observer;
private:
    Metal& gpu_; ReadPool& reads_;
    Linear embedding_,head_;
    uint32_t context_;
    Json manifest_;
    Buf dense_;
    std::unordered_map<std::string,Buf> norms_;
    std::unique_ptr<File> experts_;
    std::unique_ptr<ExpertCache> cache_;
    Linear linear(const std::string&) const;
    Buf norm(const Buf&,const std::string&,uint32_t width,uint32_t group,uint32_t tokens);
    Buf unary(const Buf&,uint32_t op);
    std::pair<Buf,Buf> hyper(const Buf&,const std::string&,uint32_t,bool inject=true);
    Buf attention(const Buf&,DraftState&,uint32_t);
    Buf moe(const Buf&,uint32_t,const std::atomic<bool>*);
    void observe(const std::string&,const Buf&);
    DraftOutput execute(std::span<const int>,const Buf&,DraftState&,bool,
                        const std::atomic<bool>*,bool state_only);
};
} // namespace zerocool::engine
