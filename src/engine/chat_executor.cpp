#include "engine/server.hpp"
#include <cmath>
#include <limits>

namespace zerocool::engine {
namespace {
// Request parameters are validated by type. nlohmann's value() would coerce a
// float, a null or an out-of-range number into a silently wrong setting.
[[noreturn]] void invalid(const char* key,const char* expected) {
    throw std::invalid_argument(std::string(key)+" must be "+expected);
}
bool present(const Json& j,const char* key) {return j.contains(key) && !j[key].is_null();}
int integer(const Json& j,const char* key,int fallback) {
    if(!present(j,key)) return fallback;
    if(!j[key].is_number_integer()) invalid(key,"an integer");
    const auto value=j[key].get<int64_t>();
    if(value<std::numeric_limits<int>::min() || value>std::numeric_limits<int>::max()) invalid(key,"within integer range");
    return int(value);
}
double number(const Json& j,const char* key,double fallback) {
    if(!present(j,key)) return fallback;
    if(!j[key].is_number()) invalid(key,"a number");
    const auto value=j[key].get<double>();
    if(!std::isfinite(value)) invalid(key,"finite");
    return value;
}
uint64_t seed_of(const Json& j,uint64_t fallback) {
    if(!present(j,"seed")) return fallback;
    if(!j["seed"].is_number_unsigned()) invalid("seed","a nonnegative integer");
    return j["seed"].get<uint64_t>();
}
bool boolean(const Json& j,const char* key,bool fallback) {
    if(!present(j,key)) return fallback;
    if(!j[key].is_boolean()) invalid(key,"a boolean");
    return j[key].get<bool>();
}
std::string string_of(const Json& j,const char* key,const std::string& fallback) {
    if(!present(j,key)) return fallback;
    if(!j[key].is_string()) invalid(key,"a string");
    return j[key].get<std::string>();
}
}
std::string api_model_id(Artifact artifact) {
    return artifact==Artifact::Mixed?"qwen3.8-flash-next:mixed-4_8bit":"qwen3.8-flash-next:4bit";
}
ChatRequest parse_chat_request(const Json& j,const Options& options) {
    if(!j.is_object()) throw std::invalid_argument("request must be an object");
    if(string_of(j,"model",api_model_id(options.artifact))!=api_model_id(options.artifact)) throw std::invalid_argument("unknown model");
    if(integer(j,"n",1)!=1) throw std::invalid_argument("only n=1 is supported");
    for(const auto* key:{"response_format","logit_bias","logprobs","stop"})
        if(present(j,key)) throw std::invalid_argument(std::string(key)+" is not supported");
    if(number(j,"frequency_penalty",0)!=0 || number(j,"presence_penalty",0)!=0)
        throw std::invalid_argument("sampling penalties are unsupported");
    ChatRequest r; r.options=options;auto& o=r.options;
    o.max_tokens=integer(j,"max_completion_tokens",integer(j,"max_tokens",options.max_tokens));
    o.temperature=float(number(j,"temperature",options.temperature));o.top_p=float(number(j,"top_p",options.top_p));
    o.top_k=integer(j,"top_k",options.top_k);o.seed=seed_of(j,options.seed);
    if(o.max_tokens<1 || o.max_tokens>options.context || !std::isfinite(o.temperature) || o.temperature<0 ||
       !std::isfinite(o.top_p) || o.top_p<=0 || o.top_p>1 || o.top_k<0) throw std::invalid_argument("invalid generation parameters");
    r.messages=j.at("messages");r.tools=j.value("tools",Json::array());
    if(!r.messages.is_array() || r.messages.empty() || !r.tools.is_array()) throw std::invalid_argument("messages/tools must be arrays and messages must not be empty");
    for(const auto& m:r.messages) {
        if(!m.is_object() || !m.contains("role") || !m["role"].is_string()) throw std::invalid_argument("invalid message");
        auto role=m["role"].get<std::string>();
        if(role!="system" && role!="user" && role!="assistant" && role!="tool") throw std::invalid_argument("unsupported message role");
        if(present(m,"content") && m["content"].is_array()) {
            for(const auto& part:m["content"])
                if(!part.is_object() || part.value("type","")!="text" || !part.contains("text") || !part["text"].is_string())
                    throw std::invalid_argument("only text message content is supported");
        } else if(present(m,"content") && !m["content"].is_string())
            throw std::invalid_argument("only text message content is supported");
    }
    if(j.contains("tool_choice")) {
        if(j["tool_choice"]=="none") r.tools=Json::array();
        else if(j["tool_choice"]!="auto") throw std::invalid_argument("tool_choice supports auto or none");
    }
    r.thinking=boolean(j,"enable_thinking",false);
    if(present(j,"chat_template_kwargs")) {
        if(!j["chat_template_kwargs"].is_object()) throw std::invalid_argument("chat_template_kwargs must be an object");
        r.thinking=boolean(j["chat_template_kwargs"],"enable_thinking",r.thinking);
    }
    r.reasoning_effort=string_of(j,"reasoning_effort","xhigh");
    return r;
}
namespace {
class NativeChat final:public ChatExecutor {
    Options options_;
    std::unique_ptr<Model> model_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<Session> session_;
public:
    explicit NativeChat(Options o):options_(std::move(o)){}
    void initialize(const ChatObserver& progress) override {
        progress({{"phase","loading_model"}});
        model_=std::make_unique<Model>(options_);
        progress({{"phase","loading_tokenizer"},{"memory_plan",model_->memory_plan().json()},
                  {"artifact_revision",model_->checkpoint().revision()}});
        tokenizer_=std::make_unique<Tokenizer>(options_.model);
        session_=std::make_unique<Session>(*model_,*tokenizer_);
        // Build every compute pipeline before the listener reports ready, so
        // the first request does not also pay kernel compilation.
        progress({{"phase","preparing_kernels"}});
        model_->prepare_pipelines();
    }
    ChatResponse generate(const ChatRequest& request,const ChatObserver& delta,
                          const ChatObserver& progress,const std::atomic<bool>& cancel) override {
        const auto prompt=tokenizer_->encode_chat(request.messages,request.tools,request.thinking,request.reasoning_effort);
        if(prompt.size()+uint64_t(request.options.max_tokens)>uint64_t(options_.context))
            throw std::invalid_argument("prompt plus max_tokens exceeds context; shorten history or reduce max_tokens");
        if(cancel) throw std::runtime_error("generation cancelled");
        session_->progress(progress);
        StreamParser parser(request.thinking);
        if(delta) delta({{"role","assistant"}});
        auto result=session_->generate(prompt,request.options,[&](int token) {
            if(delta) {
                const std::array<int,1> ids={token};
                for(const auto& item:parser.push(tokenizer_->decode(ids))) delta(item);
            }
        },&cancel);
        auto message=parse_output(result.text,request.tools,request.thinking);
        if(delta) for(const auto& item:parser.push("",true)) delta(item);
        auto diagnostics=result.json();diagnostics["after"]=model_->stats();
        return {message,{{"prompt_tokens",result.prompt_tokens},{"completion_tokens",result.tokens.size()},
            {"total_tokens",result.prompt_tokens+result.tokens.size()},
            {"prompt_tokens_details",{{"cached_tokens",result.reused_tokens}}}},std::move(diagnostics),
            message.contains("tool_calls")?"tool_calls":result.finish_reason};
    }
    void clear() override {if(session_)session_->clear();}
    void recover() override {if(session_ && !session_->reusable()) session_->clear();}
    void drain() override {if(model_)model_->diagnostic_drain();}
};
}
std::unique_ptr<ChatExecutor> make_native_chat(const Options& options) {return std::make_unique<NativeChat>(options);}
} // namespace zerocool::engine
