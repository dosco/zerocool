#include "qwen/model.hpp"
#include <algorithm>
#include <stdexcept>

namespace freellm::qwen {
std::vector<float> Model::forward_panel(std::span<const int> ids,State& state,bool logits,
                                      const std::atomic<bool>* cancel) {
    // forward() validated IDs, admission and all layer positions. Recurrent
    // state cannot roll back: only the completed panel commits global history.
    StateUpdate update(state,ids);
    const auto check_cancel=[&] {if(cancel && cancel->load()) throw std::runtime_error("generation cancelled");};
    const uint32_t start=state.tokens,T=uint32_t(ids.size());
    ++panel_passes_;
    Buf h,ng;std::future<void> ng_future;
    try {
        gpu_.label("embedding",-1,T,start);
        h=gpu_.embedding(resident_->linear("model.embed_tokens"),ids,4);
        ng=gpu_.allocate(uint64_t(T)*Hidden*4);
        ng_future=std::async(std::launch::async,[this,ids,history=state.history,ng]{ngrams_->embedding(ids,history,ng->floats());});
        for(int l=0;l<options_.probe_layers;++l) {
            check_cancel();resident_->activate_layer(l);
            auto& st=state.layers[l];const auto b="model.layers."+std::to_string(l);
            trace_offset_=start;
            if(!options_.trace_dir.empty()) {gpu_.finish();trace("input_"+std::to_string(l),h);}
            if(l==1) ng_future.get();
            auto after=gpu_.allocate(uint64_t(T)*Hyper*4);
            auto inputs=gpu_.allocate(uint64_t(T)*Hidden*4),injections=gpu_.allocate(uint64_t(T)*4*4);
            size_t workspace=0;
            for(uint32_t at=0;at<T;) {
                if(options_.prefill_pipeline=="double") gpu_.begin_scratch(workspace++%2,plan_.scratch);
                check_cancel();
                const auto n=std::min<uint32_t>(options_.chunk,T-at);
                if(st.position!=start+at) throw std::runtime_error("panel layer position mismatch");
                trace_offset_=st.position;
                auto part=gpu_.slice(h,uint64_t(at)*Hyper*4,uint64_t(n)*Hyper*4);
                if(l==1) {
                    auto embedding=gpu_.slice(ng,uint64_t(at)*Hidden*4,uint64_t(n)*Hidden*4);
                    part=binary(part,ple(part,embedding,st,n),0);
                }
                gpu_.label("attention_input",l,n,st.position);
                auto [x,inj]=hyper(part,b+".attn_hyper_connection",n);
                auto attention_out=(l+1)%4?gdn(x,st,l,n):attention(x,st,l,n,st.position);
                gpu_.label("attention_residual",l,n,st.position);
                auto residual=gpu_.allocate(part->bytes);
                gpu_.dispatch("hc_add",{{part},{attention_out},{inj},{residual}},{n},Hyper,n);
                gpu_.label("mlp_input",l,n,st.position);
                auto [mx,mi]=hyper(residual,b+".mlp_hyper_connection",n);
                gpu_.copy(residual,0,after,uint64_t(at)*Hyper*4,residual->bytes);
                gpu_.copy(mx,0,inputs,uint64_t(at)*Hidden*4,mx->bytes);
                gpu_.copy(mi,0,injections,uint64_t(at)*4*4,mi->bytes);
                st.position+=n;at+=n;
                // Complete the microchunk as one group so its temporary buffers
                // are released before the next; the expert pipeline still overlaps
                // up to two groups with reads after these inputs are ready.
                if(options_.prefill_pipeline=="double") gpu_.end_scratch();
                else gpu_.finish();
            }
            h.reset();
            trace_offset_=start;
            // Every selected expert is acquired once for the whole panel.
            // Its token rows execute in bounded microbatches inside moe().
            auto contributions=moe(inputs,l,T,cancel);
            h=gpu_.allocate(uint64_t(T)*Hyper*4);
            gpu_.dispatch("hc_add",{{after},{contributions},{injections},{h}},{T},Hyper,T);
            if(!options_.trace_dir.empty()) {gpu_.finish();trace("layer_"+std::to_string(l),h);}
        }
        // A truncated diagnostic may omit the ngram layer; drain its users too.
        if(ng_future.valid()) ng_future.get();
        check_cancel();
        std::vector<float> result;
        if(logits) result=compute_logits(h,T);else gpu_.finish();
        update.commit();return result;
    } catch(...) {
        const auto failure=std::current_exception();
        if(ng_future.valid()) ng_future.wait();
        try {gpu_.end_scratch();} catch(...) {}
        try {gpu_.finish();} catch(...) {}
        reads_.drain();std::rethrow_exception(failure);
    }
}
} // namespace freellm::qwen
