#include "engine/session.hpp"
#include <minja/minja.hpp>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace zerocool::engine {
namespace {
std::string utf8(uint32_t cp) {
    std::string s;
    if(cp<128) s+=char(cp);
    else if(cp<2048) {s+=char(0xc0|(cp>>6));s+=char(0x80|(cp&63));}
    else {s+=char(0xe0|(cp>>12));s+=char(0x80|((cp>>6)&63));s+=char(0x80|(cp&63));}
    return s;
}
// Private-use markers stand in for control-token text while the chat template
// renders. They are valid UTF-8 and valid JSON, so tool arguments still parse,
// and no chat template inspects them.
constexpr std::string_view GuardOpen = "\xee\x80\x80", GuardClose = "\xee\x80\x81";
}
struct Tokenizer::Impl {
    std::unordered_map<std::string,int> vocab,merges;
    std::vector<std::string> decoded;
    std::vector<std::pair<std::string,int>> special;
    std::array<std::string,256> byte_unicode;
    NSRegularExpression* regex;
    std::shared_ptr<minja::TemplateNode> tmpl;

    std::vector<int> bpe(const std::string& raw) const {
        struct Node {std::string text; int prev,next,version=0; bool live=true;};
        struct Pair {int rank,left,right,lv,rv; bool operator>(const Pair& p) const {
            return rank==p.rank?left>p.left:rank>p.rank;
        }};
        std::vector<Node> nodes; nodes.reserve(raw.size());
        for(size_t i=0;i<raw.size();++i) nodes.push_back({byte_unicode[uint8_t(raw[i])],int(i)-1,int(i)+1});
        if(nodes.empty()) return {};
        nodes.back().next=-1;
        std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>> queue;
        auto push=[&](int left) {
            if(left<0 || !nodes[left].live || nodes[left].next<0) return;
            int right=nodes[left].next;
            auto it=merges.find(nodes[left].text+" "+nodes[right].text);
            if(it!=merges.end()) queue.push({it->second,left,right,nodes[left].version,nodes[right].version});
        };
        for(size_t i=0;i<nodes.size();++i) push(int(i));
        while(!queue.empty()) {
            const auto p=queue.top(); queue.pop();
            auto& l=nodes[p.left]; auto& r=nodes[p.right];
            if(!l.live || !r.live || l.next!=p.right || l.version!=p.lv || r.version!=p.rv) continue;
            l.text+=r.text; l.next=r.next; ++l.version; r.live=false;
            if(r.next>=0) nodes[r.next].prev=p.left;
            push(l.prev); push(p.left);
        }
        std::vector<int> result;
        for(int i=0;i>=0;i=nodes[i].next) {
            auto v=vocab.find(nodes[i].text);
            if(v==vocab.end()) throw std::runtime_error("BPE merge is absent from vocabulary");
            result.push_back(v->second);
        }
        return result;
    }
    void ordinary(const std::string& text,std::vector<int>& result) const {
        if(text.empty()) return;
        NSString* value=[[NSString alloc] initWithBytes:text.data() length:text.size() encoding:NSUTF8StringEncoding];
        if(!value) throw std::invalid_argument("input is not valid UTF-8");
        value=value.precomposedStringWithCanonicalMapping;
        NSArray<NSTextCheckingResult*>* matches=[regex matchesInString:value options:0 range:NSMakeRange(0,value.length)];
        NSUInteger cursor=0;
        for(NSTextCheckingResult* match in matches) {
            if(match.range.location!=cursor) throw std::runtime_error("pre-tokenizer did not cover input");
            NSData* bytes=[[value substringWithRange:match.range] dataUsingEncoding:NSUTF8StringEncoding];
            const auto piece=std::string(static_cast<const char*>(bytes.bytes),bytes.length);
            auto tokens=bpe(piece); result.insert(result.end(),tokens.begin(),tokens.end());
            cursor=NSMaxRange(match.range);
        }
        if(cursor!=value.length) throw std::runtime_error("pre-tokenizer left unmatched input");
    }
    // Control-token text is recognized anywhere, so this is only ever applied
    // to the rendered template, never to untrusted message text.
    void with_special(const std::string& text,std::vector<int>& tokens) const {
        size_t cursor=0;
        while(cursor<text.size()) {
            size_t first=text.size(); const std::pair<std::string,int>* selected=nullptr;
            for(const auto& s:special) {
                const auto at=text.find(s.first,cursor);
                if(at<first) {first=at;selected=&s;}
            }
            ordinary(text.substr(cursor,first-cursor),tokens);
            if(!selected) break;
            tokens.push_back(selected->second); cursor=first+selected->first.size();
        }
    }
};
Tokenizer::Tokenizer(const std::filesystem::path& model) : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        // ordered_json's vector-backed object insertion is quadratic for this
        // 248K-entry vocabulary. Token IDs provide order; use the tree map.
        auto& p=*impl_; auto j=nlohmann::json::parse(read_text(model/"tokenizer.json"));
        if(j.at("model").at("type")!="BPE" || j.at("normalizer").at("type")!="NFC")
            throw std::runtime_error("unsupported tokenizer; requires pinned Qwen BPE/NFC");
        std::unordered_map<std::string,unsigned char> reverse;
        uint32_t extra=256;
        for(int b=0;b<256;++b) {
            bool direct=(b>=33 && b<=126)||(b>=161 && b<=172)||(b>=174 && b<=255);
            p.byte_unicode[b]=utf8(direct?uint32_t(b):extra++); reverse[p.byte_unicode[b]]=uint8_t(b);
        }
        p.decoded.resize(Vocab);
        auto decode_piece=[&](const std::string& text) {
            std::string out;
            for(size_t i=0;i<text.size();) {
                const auto c=uint8_t(text[i]); size_t n=c<128?1:(c<224?2:3);
                auto it=reverse.find(text.substr(i,n));
                if(it==reverse.end()) throw std::runtime_error("invalid byte vocabulary");
                out+=char(it->second); i+=n;
            }
            return out;
        };
        for(auto it=j["model"]["vocab"].begin();it!=j["model"]["vocab"].end();++it) {
            const int id=it.value().get<int>();
            if(id<0 || id>=Vocab) throw std::runtime_error("vocabulary id outside model");
            p.vocab[it.key()]=id;
            p.decoded[id]=decode_piece(it.key());
        }
        int rank=0;
        for(const auto& merge:j["model"]["merges"]) {
            const auto key=merge.is_array()?merge.at(0).get<std::string>()+" "+merge.at(1).get<std::string>():merge.get<std::string>();
            p.merges[key]=rank++;
        }
        for(const auto& token:j["added_tokens"]) {
            const int id=token.at("id"); const auto text=token.at("content").get<std::string>();
            if(id<0 || id>=Vocab || text.empty()) throw std::runtime_error("invalid added token");
            p.special.emplace_back(text,id); p.decoded[id]=text;
        }
        std::sort(p.special.begin(),p.special.end(),[](const auto& a,const auto& b){return a.first.size()>b.first.size();});
        const auto pattern=j["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"].get<std::string>();
        NSError* error=nil;
        p.regex=[NSRegularExpression regularExpressionWithPattern:[NSString stringWithUTF8String:pattern.c_str()] options:0 error:&error];
        if(!p.regex) throw std::runtime_error("cannot compile Qwen pre-tokenizer: "+std::string(error.localizedDescription.UTF8String));
        auto source=read_text(model/"chat_template.jinja");
        // minja implements `defined` but not its Jinja synonym `undefined`.
        // Normalize the three pinned predicates without changing asset bytes.
        for(size_t at=0;(at=source.find(" is undefined",at))!=std::string::npos;at+=15)
            source.replace(at,13," is not defined");
        p.tmpl=minja::Parser::parse(source,{});
    }
}
Tokenizer::~Tokenizer()=default;
std::vector<int> Tokenizer::encode(const std::string& text) const {
    @autoreleasepool {
        std::vector<int> tokens; impl_->with_special(text,tokens); return tokens;
    }
}
std::string Tokenizer::decode(std::span<const int> ids) const {
    std::string text;
    for(int id:ids) {
        if(id<0 || size_t(id)>=impl_->decoded.size()) throw std::out_of_range("invalid decode token");
        text+=impl_->decoded[id];
    }
    return text;
}
std::string Tokenizer::render(Json messages,const Json& tools,bool thinking,const std::string& effort) const {
    if(!messages.is_array() || messages.empty() || messages.size()>4096 || !tools.is_array())
        throw std::invalid_argument("messages/tools must be arrays and messages must not be empty");
    for(auto& m:messages) {
        if(!m.is_object() || !m.contains("role") || !m["role"].is_string()) throw std::invalid_argument("invalid message role");
        const auto role=m["role"].get<std::string>();
        if(role!="user" && role!="assistant" && role!="system" && role!="tool") throw std::invalid_argument("unsupported message role");
        if(m.contains("content") && m["content"].is_array()) {
            for(const auto& part:m["content"]) if(part.value("type","")!="text") throw std::invalid_argument("only text content is supported");
        } else if(m.contains("content") && !m["content"].is_null() && !m["content"].is_string())
            throw std::invalid_argument("invalid message content");
        if(m.contains("tool_calls")) for(auto& call:m["tool_calls"]) {
            auto& f=call.contains("function")?call["function"]:call;
            if(f.contains("arguments") && f["arguments"].is_string()) f["arguments"]=Json::parse(f["arguments"].get<std::string>());
            if(f.contains("arguments") && !f["arguments"].is_object()) throw std::invalid_argument("tool arguments must be an object");
        }
    }
    Json input={{"messages",messages},{"tools",tools},{"add_generation_prompt",true},
        {"enable_thinking",thinking},{"preserve_thinking",true},{"reasoning_effort",effort},{"add_vision_id",false}};
    return impl_->tmpl->render(minja::Context::make(minja::Value(input)));
}
std::vector<int> Tokenizer::encode_chat(Json messages,const Json& tools,bool thinking,const std::string& effort) const {
    @autoreleasepool {
        if(!messages.is_array() || !tools.is_array()) throw std::invalid_argument("messages/tools must be arrays");
        // Replace control-token text with a marker naming the token. The marker
        // survives rendering, and the split below encodes the original bytes as
        // ordinary text, so message content cannot forge a turn boundary.
        auto guard=[&](const std::string& text,bool keep_tool_response) {
            if(text.find(GuardOpen)!=std::string::npos || text.find(GuardClose)!=std::string::npos)
                throw std::invalid_argument("message text contains a reserved private-use marker");
            std::string out; size_t cursor=0;
            while(cursor<text.size()) {
                size_t first=text.size(),selected=impl_->special.size();
                for(size_t k=0;k<impl_->special.size();++k) {
                    // The template itself tests for a tool-response wrapper, so
                    // that pair keeps its meaning inside tool results.
                    const auto& token=impl_->special[k].first;
                    if(keep_tool_response && (token=="<tool_response>" || token=="</tool_response>")) continue;
                    const auto at=text.find(token,cursor);
                    if(at<first) {first=at;selected=k;}
                }
                out+=text.substr(cursor,first-cursor);
                if(selected==impl_->special.size()) break;
                out+=std::string(GuardOpen)+std::to_string(selected)+std::string(GuardClose);
                cursor=first+impl_->special[selected].first.size();
            }
            return out;
        };
        const std::function<void(Json&,bool)> walk=[&](Json& value,bool keep) {
            if(value.is_string()) value=guard(value.get<std::string>(),keep);
            else if(value.is_array() || value.is_object()) for(auto& item:value) walk(item,keep);
        };
        for(auto& message:messages) {
            if(!message.is_object()) throw std::invalid_argument("invalid message");
            const bool tool=message.contains("role") && message["role"].is_string() && message["role"]=="tool";
            for(const char* key:{"content","reasoning_content","tool_calls"})
                if(message.contains(key)) walk(message[key],tool);
        }
        Json declared=tools; walk(declared,false);
        const auto rendered=render(std::move(messages),declared,thinking,effort);
        std::vector<int> tokens; size_t cursor=0;
        for(;;) {
            const auto at=rendered.find(GuardOpen,cursor);
            if(at==std::string::npos) break;
            const auto end=rendered.find(GuardClose,at+GuardOpen.size());
            if(end==std::string::npos) throw std::runtime_error("chat template damaged a control marker");
            const auto index=std::stoull(rendered.substr(at+GuardOpen.size(),end-at-GuardOpen.size()));
            if(index>=impl_->special.size()) throw std::runtime_error("chat template produced an unknown control marker");
            impl_->with_special(rendered.substr(cursor,at-cursor),tokens);
            impl_->ordinary(impl_->special[index].first,tokens);
            cursor=end+GuardClose.size();
        }
        impl_->with_special(rendered.substr(cursor),tokens);
        return tokens;
    }
}
} // namespace zerocool::engine
