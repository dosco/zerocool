#include "qwen/metal.hpp"
#include <fstream>
#include <print>
using namespace freellm::qwen;
// Isolated production-operator replay; no full-model or performance claim.
int main(int argc,char** argv) {
 try {
 if(argc!=4) throw std::invalid_argument("usage: replay_attention MODEL TRACE OUTPUT");Checkpoint cp(argv[1]);Metal gpu;gpu.budget(64*MiB);
 std::filesystem::path trace(argv[2]),outdir(argv[3]);std::filesystem::create_directories(outdir);
 auto load=[&](const std::string& name){return cp.load(name,[&](uint64_t size){return gpu.allocate(size);});};
 auto linear=[&](const std::string& name,const Buf& x,int T){auto &r=cp.at(name+".weight");Linear l{{load(name+".weight")},{load(name+".scales")},{load(name+".biases")},uint32_t(r.shape[1]*8),uint32_t(r.shape[0]),64,0,true};return gpu.linear(l,x,T);};
 auto save=[&](const std::string& name,const Buf& b){std::ofstream f(outdir/(name+".bin"),std::ios::binary);f.write(reinterpret_cast<const char*>(b->data),std::streamsize(b->bytes));if(!f) throw std::runtime_error("cannot write fixture output");};
 for(int l=3;l<48;l+=4){
  auto b="model.layers."+std::to_string(l)+".self_attn";File input(trace/("x1_"+std::to_string(l)+".bin"));auto x=gpu.allocate(input.size());input.read(0,{x->data,size_t(x->bytes)});uint32_t T=x->bytes/(Hidden*4);
  if(T!=5 || x->bytes!=5*Hidden*4) throw std::invalid_argument("expected a five-token fixture");
  auto qg=linear(b+".q_proj",x,T),rk=linear(b+".k_proj",x,T),v=linear(b+".v_proj",x,T);
  auto q=gpu.zeros(T*6144),k=gpu.zeros(T*512);
  gpu.dispatch("norm_rope",{{qg},{load(b+".q_norm.weight")},{q}},{256,24,12288,512,0,T,0},32*24,T);
  gpu.dispatch("norm_rope",{{rk},{load(b+".k_norm.weight")},{k}},{256,2,512,256,0,T,0},32*2,T);
  auto mask=gpu.zeros(T*T),scores=gpu.zeros(T*24*T),out=gpu.zeros(T*6144);
  gpu.dispatch("attention_scores",{{q},{k},{mask},{scores}},{T,0,T,0},((T+7)/8)*32,(T+7)/8,24);
  gpu.finish();save(b+".scores",scores);
  gpu.dispatch("attention_softmax",{{scores}},{T},T*24*32);
  gpu.dispatch("attention_values",{{scores},{v},{qg},{out}},{T,T},32*32,(T+7)/8,24);
  auto final=linear(b+".o_proj",out,T);gpu.finish();
  save(b+".q",q);save(b+".k",k);save(b+".v",v);save(b+".qg",qg);save(b+".probabilities",scores);save(b+".gated",out);save("attn_"+std::to_string(l),final);
 }
 std::println("{}",gpu.statistics().dump(2));
 } catch(const std::exception& e) {std::println(stderr,"replay_attention: {}",e.what());return 1;}
}
