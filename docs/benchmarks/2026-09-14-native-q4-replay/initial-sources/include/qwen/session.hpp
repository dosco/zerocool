#pragma once
#include "qwen/model.hpp"
#include <random>

namespace freellm::qwen {
class Tokenizer {
public:
    explicit Tokenizer(const std::filesystem::path& model);
    ~Tokenizer();
    std::vector<int> encode(const std::string& text) const;
    std::string decode(std::span<const int> ids) const;
    std::string render(Json messages,const Json& tools=Json::array(),bool thinking=false,
                       const std::string& effort="xhigh") const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
int sample(std::span<const float> logits,float temperature,int top_k,float top_p,std::mt19937_64& rng);
Json parse_output(const std::string& generated,const Json& tools,bool thinking);
class StreamParser {
public:
    explicit StreamParser(bool thinking) : reasoning_(thinking) {}
    std::vector<Json> push(const std::string& bytes,bool final=false);
private:
    std::string pending_;
    bool reasoning_,tools_=false;
};

struct Result {
    std::vector<int> tokens;
    std::string text;
    std::string finish_reason;
    size_t prompt_tokens=0,reused_tokens=0;
    double first_token_ms=0,prefill_ms=0,decode_ms=0,request_ms=0,decode_wall_ms=0;
    std::vector<double> token_ms;
    size_t pending_tokens_ingested=0;
    Json phases=Json::object();
    bool diagnose_decode=false;
    Json decode_samples=Json::array();
    Json json() const;
};

// Only a full continuation of the retained live tokens reuses state. A shorter
// or edited prefix cannot roll back GDN recurrence and forces a complete replay.
class Session {
public:
    Session(Model& model,Tokenizer& tokenizer);
    Result generate(const std::vector<int>& prompt,const Options& options,
                    const std::function<void(int)>& on_token={},const std::atomic<bool>* cancel=nullptr);
    void clear();
    // Benchmark priming shares normal ingestion and creates no pending output.
    Result prime(const std::vector<int>& prompt,const std::atomic<bool>* cancel=nullptr);
private:
    void ingest(const std::vector<int>& prompt,Result& result,const std::atomic<bool>* cancel);
    Model& model_;
    Tokenizer& tokenizer_;
    std::optional<State> state_;
    std::vector<int> retained_;
    std::vector<float> last_logits_;
    std::optional<int> pending_token_;
};

void serve(Model& model,Tokenizer& tokenizer,const Options& options,uint16_t port,const std::atomic<bool>* stop=nullptr);
} // namespace freellm::qwen
