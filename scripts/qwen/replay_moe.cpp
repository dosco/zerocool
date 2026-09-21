#include "engine/metal.hpp"
#include <fstream>
#include <print>
using namespace zerocool::engine;
// Bounded numerical diagnostic, linked to the production Metal executor.
// Input traces must come from a five-token native forward pass. This replays
// isolated operators on recorded inputs; it does not verify the full model.
int main(int argc,char** argv) {
 try {
 if(argc!=4) throw std::invalid_argument("usage: replay_moe MODEL TRACE OUTPUT");
 Checkpoint cp(argv[1]);Metal gpu;gpu.budget(64*MiB);
 std::filesystem::path trace(argv[2]),out(argv[3]);std::filesystem::create_directories(out);
 auto load=[&](const std::string& name) {File f(trace/(name+".bin"));auto b=gpu.allocate(f.size());f.read(0,{b->data,size_t(b->bytes)});return b;};
 auto save=[&](const std::string& name,const Buf& b) {std::ofstream f(out/(name+".bin"),std::ios::binary);f.write(reinterpret_cast<const char*>(b->data),std::streamsize(b->bytes));if(!f) throw std::runtime_error("cannot write fixture output");};
 for(int l=0;l<48;++l) {
  const auto base="model.layers."+std::to_string(l)+".mlp";
  auto w=cp.load(base+".gate.weight",[&](uint64_t size){return gpu.allocate(size);});
  auto x=load("x2_"+std::to_string(l));auto T=uint32_t(x->bytes/(Hidden*4));
  if(T!=5 || x->bytes!=5*Hidden*4) throw std::invalid_argument("expected a five-token fixture");
  auto router=gpu.linear({{w},{},{},Hidden,Experts,64,0,false},x,T,true);
  auto ids=gpu.allocate(T*10*4),weights=gpu.allocate(T*10*4);
  gpu.dispatch("route",{{router},{ids},{weights}},{T},T*32);
  auto expert=load(base+".expert_out"),shared=load(base+".shared"),gate=load(base+".gate"),moe=gpu.zeros(T*Hidden);
  if(expert->bytes!=T*TopK*Hidden*4 || shared->bytes!=T*Hidden*4 || gate->bytes!=T*4)
   throw std::invalid_argument("invalid expert reduction fixture shape");
  gpu.dispatch("moe_sum",{{expert},{weights},{shared},{gate},{moe}},{T},Hidden,T);
  gpu.finish();
  save("router_"+std::to_string(l),router);save("route_"+std::to_string(l),ids);save(base+".weights",weights);save("moe_"+std::to_string(l),moe);
 }
 std::println("{}",gpu.statistics().dump(2));
 } catch(const std::exception& e) {std::println(stderr,"replay_moe: {}",e.what());return 1;}
}
