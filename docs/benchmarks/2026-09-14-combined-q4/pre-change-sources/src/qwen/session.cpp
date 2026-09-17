#include "qwen/session.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace freellm::qwen {
namespace {
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point since) {return std::chrono::duration<double,std::milli>(Clock::now()-since).count();}
std::string trim(std::string s) {
    const auto begin=s.find_first_not_of(" \r\n\t"); if(begin==std::string::npos) return {};
    return s.substr(begin,s.find_last_not_of(" \r\n\t")-begin+1);
}
}
int sample(std::span<const float> logits,float temperature,int top_k,float top_p,std::mt19937_64& rng) {
    if(logits.empty() || !std::isfinite(temperature) || temperature<0 || !std::isfinite(top_p) || top_p<=0 || top_p>1 || top_k<0)
        throw std::invalid_argument("invalid sampling parameters");
    for(float v:logits) if(!std::isfinite(v)) throw std::runtime_error("non-finite sampling logits");
    if(temperature==0) return int(std::max_element(logits.begin(),logits.end())-logits.begin());
    std::vector<size_t> order(logits.size()); std::iota(order.begin(),order.end(),0);
    const size_t count=top_k?std::min<size_t>(top_k,logits.size()):logits.size();
    std::partial_sort(order.begin(),order.begin()+count,order.end(),[&](auto a,auto b){
        return logits[a]==logits[b]?a<b:logits[a]>logits[b];
    });
    order.resize(count); std::vector<double> probs(count); double total=0;
    for(size_t i=0;i<count;++i) {probs[i]=std::exp((double(logits[order[i]])-logits[order[0]])/temperature);total+=probs[i];}
    size_t kept=0; double cumulative=0;
    do {cumulative+=probs[kept++];} while(kept<count && cumulative<top_p*total);
    probs.resize(kept);
    return int(order[std::discrete_distribution<size_t>(probs.begin(),probs.end())(rng)]);
}
Json parse_output(const std::string& generated,const Json& tools,bool thinking) {
    std::string text=generated,reasoning;
    if(thinking) {
        if(auto p=text.find("</think>");p!=std::string::npos) {
            reasoning=trim(text.substr(0,p)); text=text.substr(p+8);
        } else { reasoning=trim(text); text.clear(); }
    }
    Json calls=Json::array(); std::string content;
    size_t cursor=0;
    while(cursor<text.size()) {
        auto start=text.find("<tool_call>",cursor);
        if(start==std::string::npos) {content+=text.substr(cursor);break;}
        content+=text.substr(cursor,start-cursor);
        auto end=text.find("</tool_call>",start+11);
        if(end==std::string::npos) throw std::runtime_error("model emitted an incomplete tool call");
        const auto block=trim(text.substr(start+11,end-start-11));
        if(!block.starts_with("<function=")) throw std::runtime_error("invalid tool-call format");
        auto close=block.find('>'); auto f_end=block.rfind("</function>");
        if(close==std::string::npos || f_end==std::string::npos || f_end<close) throw std::runtime_error("invalid function block");
        const auto name=block.substr(10,close-10);
        Json schema; bool found=false;
        for(const auto& tool:tools) {
            const auto& fn=tool.contains("function")?tool["function"]:tool;
            if(fn.value("name","")==name) {schema=fn.value("parameters",Json::object());found=true;break;}
        }
        if(!found) throw std::runtime_error("model called an undeclared tool: "+name);
        Json arguments=Json::object(); size_t at=close+1;
        while(at<f_end) {
            while(at<f_end && std::isspace(static_cast<unsigned char>(block[at]))) ++at;
            if(at==f_end) break;
            if(block.compare(at,11,"<parameter=")!=0) throw std::runtime_error("invalid tool parameter");
            auto key_end=block.find('>',at+11),value_end=block.find("</parameter>",key_end);
            if(key_end==std::string::npos || value_end==std::string::npos || value_end>f_end) throw std::runtime_error("incomplete tool parameter");
            auto key=block.substr(at+11,key_end-at-11),value=block.substr(key_end+1,value_end-key_end-1);
            if(!value.empty() && value.front()=='\n') value.erase(0,1);
            if(!value.empty() && value.back()=='\n') value.pop_back();
            if(arguments.contains(key)) throw std::runtime_error("duplicate tool parameter");
            Json property=schema.value("properties",Json::object()).value(key,Json::object());
            const auto type=property.value("type",std::string("string"));
            if(type=="string") arguments[key]=value;
            else {
                arguments[key]=Json::parse(value);
                const auto& v=arguments[key];
                if((type=="integer" && !v.is_number_integer()) || (type=="number" && !v.is_number()) ||
                   (type=="boolean" && !v.is_boolean()) || (type=="object" && !v.is_object()) ||
                   (type=="array" && !v.is_array())) throw std::runtime_error("tool parameter type mismatch: "+key);
            }
            at=value_end+12;
        }
        for(const auto& required:schema.value("required",Json::array()))
            if(!arguments.contains(required.get<std::string>())) throw std::runtime_error("required tool parameter missing");
        calls.push_back({{"id","call_"+std::to_string(calls.size())},{"type","function"},
            {"function",{{"name",name},{"arguments",arguments.dump()}}}});
        cursor=end+12;
    }
    Json result={{"role","assistant"},{"content",trim(content)}};
    if(thinking) result["reasoning_content"]=reasoning;
    if(!calls.empty()) result["tool_calls"]=calls;
    return result;
}
std::vector<Json> StreamParser::push(const std::string& bytes,bool final) {
    std::vector<Json> deltas;
    if(tools_) return deltas;
    pending_+=bytes;
    for(;;) {
        const std::string marker=reasoning_?"</think>":"<tool_call>";
        const auto at=pending_.find(marker);
        if(at!=std::string::npos) {
            if(at) deltas.push_back({{reasoning_?"reasoning_content":"content",pending_.substr(0,at)}});
            pending_.erase(0,at+marker.size());
            if(reasoning_) {reasoning_=false;continue;}
            tools_=true;pending_.clear();return deltas;
        }
        size_t keep=0;
        if(!final) for(size_t n=1;n<marker.size() && n<=pending_.size();++n)
            if(pending_.compare(pending_.size()-n,n,marker,0,n)==0) keep=n;
        size_t ready=pending_.size()-keep;
        if(!final && ready) {
            size_t last=ready-1;
            while(last && (uint8_t(pending_[last])&0xc0)==0x80) --last;
            auto c=uint8_t(pending_[last]);size_t width=c<128?1:c<224?2:c<240?3:4;
            if(ready-last<width) ready=last;
        }
        if(ready) {deltas.push_back({{reasoning_?"reasoning_content":"content",pending_.substr(0,ready)}});pending_.erase(0,ready);}
        return deltas;
    }
}
Json Result::json() const {
    auto latencies=token_ms; std::sort(latencies.begin(),latencies.end());
    const double p95=latencies.empty()?0:latencies[std::min(latencies.size()-1,size_t(std::ceil(latencies.size()*0.95)-1))];
    auto percentile=[&](double p) {return latencies.empty()?0:latencies[std::min(latencies.size()-1,size_t(std::ceil(latencies.size()*p)-1))];};
    Json result={{"prompt_tokens",prompt_tokens},{"reused_tokens",reused_tokens},{"prefill_tokens",prompt_tokens-reused_tokens},
        {"output_tokens",tokens.size()},{"pending_tokens_ingested",pending_tokens_ingested},{"phases",phases},{"time_to_first_token_ms",first_token_ms},{"prefill_ms",prefill_ms},
        {"decode_ms",decode_ms},{"tokens_per_second",tokens.size()>1 && decode_ms>0?(tokens.size()-1)*1000.0/decode_ms:0.0},
        {"request_ms",request_ms},{"decode_wall_ms",decode_wall_ms},
        {"wall_tokens_per_second",tokens.size()>1 && decode_wall_ms>0?(tokens.size()-1)*1000.0/decode_wall_ms:0.0},
        {"p50_token_ms",percentile(0.5)},{"p95_token_ms",p95},{"p99_token_ms",percentile(0.99)},
        {"token_latency_ms",token_ms},{"output_token_ids",tokens},{"finish_reason",finish_reason},{"text",text}};
    if(diagnose_decode) result["decode_diagnostics"]={{"kind","decode_step_diagnostics_v1"},
        {"max_steps",32},{"total_decode_steps",token_ms.size()},{"captured_steps",decode_samples.size()},
        {"omitted_steps",token_ms.size()-decode_samples.size()},{"samples",decode_samples},
        {"scope","completed decode forwards only; excludes prefill and sampling; request wall time includes observation overhead"}};
    return result;
}
Session::Session(Model& model,Tokenizer& tokenizer) : model_(model),tokenizer_(tokenizer) {}
void Session::clear() {state_.reset();retained_.clear();last_logits_.clear();pending_token_.reset();}
void Session::ingest(const std::vector<int>& prompt,Result& result,const std::atomic<bool>* cancel) {
    bool reuse=state_ && prompt.size()>=retained_.size() && std::equal(retained_.begin(),retained_.end(),prompt.begin());
    if(!reuse) {clear();state_=model_.make_state();}
    result.reused_tokens=retained_.size();
    result.pending_tokens_ingested=reuse && pending_token_ && prompt.size()>retained_.size() && prompt[retained_.size()]==*pending_token_?1:0;
    model_.phase(retained_.empty()?"prefill":"append");
    try {
        model_.prepare_ingest(prompt.size()-retained_.size());
        for(size_t at=retained_.size();at<prompt.size();) {
            if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
            const auto n=std::min<size_t>(model_.input_limit(),prompt.size()-at);
            const auto batch=std::span<const int>(prompt).subspan(at,n);
            const bool last=at+n==prompt.size();
            auto logits=model_.forward(batch,*state_,last,cancel);
            retained_.insert(retained_.end(),batch.begin(),batch.end());
            if(last) last_logits_=std::move(logits);
            at+=n;
        }
    pending_token_.reset();
        model_.finish_ingest();
    } catch(...) {
        const auto error=std::current_exception();
        try {model_.finish_ingest();} catch(...) {}
        std::rethrow_exception(error);
    }
}
Result Session::prime(const std::vector<int>& prompt,const std::atomic<bool>* cancel) {
    if(prompt.empty() || prompt.size()>size_t(model_.options().context)) throw std::invalid_argument("prime exceeds context");
    Result result;result.prompt_tokens=prompt.size();const auto before=model_.stats();const auto start=Clock::now();
    try {
        ingest(prompt,result,cancel);result.prefill_ms=ms(start);result.request_ms=result.prefill_ms;result.finish_reason="primed";
        result.phases["ingest"]={{"before",before},{"after",model_.stats()}};return result;
    } catch(...) {clear();throw;}
}
Result Session::generate(const std::vector<int>& prompt,const Options& o,
                         const std::function<void(int)>& on_token,const std::atomic<bool>* cancel) {
    if(prompt.empty() || o.max_tokens<1 || prompt.size()+uint64_t(o.max_tokens)>uint64_t(model_.options().context))
        throw std::invalid_argument("prompt plus requested output exceeds context; compact history or reduce max_tokens");
    if(!std::isfinite(o.temperature) || o.temperature<0 || !std::isfinite(o.top_p) || o.top_p<=0 || o.top_p>1 || o.top_k<0)
        throw std::invalid_argument("invalid sampling parameters");
    const auto start=Clock::now(); Result result; result.prompt_tokens=prompt.size();
    result.diagnose_decode=model_.options().decode_diagnostics;
    const auto before=model_.stats();
    try {
        ingest(prompt,result,cancel);
        result.prefill_ms=ms(start); std::mt19937_64 rng(o.seed);
        for(int i=0;i<o.max_tokens;++i) {
            if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");
            const int id=sample(last_logits_,o.temperature,o.top_k,o.top_p,rng);
            if(i==0) result.first_token_ms=ms(start);
            result.tokens.push_back(id);
            if(id!=248044 && id!=248046 && on_token) on_token(id);
            if(i==0) {
                result.phases["ingest"]={{"before",before},{"after",model_.stats()}};
                model_.phase("decode");
            }
            if(id==248044 || id==248046) {result.finish_reason="stop";break;}
            if(i+1==o.max_tokens) {result.finish_reason="length";break;}
            const bool observe=result.diagnose_decode && result.decode_samples.size()<32;
            Json counters;if(observe) counters=model_.decode_counters();
            const auto step_ns=observe?monotonic_ns():0;
            const auto step=Clock::now();
            const std::array<int,1> token={id};
            last_logits_=model_.forward(token,*state_,true,cancel);
            retained_.push_back(id);
            const auto elapsed=ms(step); result.token_ms.push_back(elapsed); result.decode_ms+=elapsed;
            if(observe) {
                const auto end=monotonic_ns();
                result.decode_samples.push_back({{"step",i},{"input_token_id",id},{"offset",state_->tokens-1},
                    {"begin_ns",step_ns},{"end_ns",end},{"forward_ms",elapsed},
                    {"before",std::move(counters)},{"after",model_.decode_counters()}});
            }
        }
        result.request_ms=ms(start);result.decode_wall_ms=result.request_ms-result.first_token_ms;
        result.phases["decode"]={{"before",result.phases["ingest"]["after"]},{"after",model_.stats()}};
        if(!result.tokens.empty()) pending_token_=result.tokens.back();
        auto emitted=std::span<const int>(result.tokens);
        if(!emitted.empty() && (emitted.back()==248044 || emitted.back()==248046)) emitted=emitted.first(emitted.size()-1);
        result.text=tokenizer_.decode(emitted);
        return result;
    } catch(...) {clear();throw;}
}
} // namespace freellm::qwen
