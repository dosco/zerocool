#include "engine/session.hpp"
#include "engine/q3_probe.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <csignal>
#include <fstream>
#include <print>

using namespace zerocool::engine;
namespace {
std::atomic<bool> stopped=false;
void interrupt(int) {stopped=true;}
void check(bool v,const char* message) {if(!v) throw std::runtime_error(message);}
std::string hash(std::span<const std::byte> bytes) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes.data(),CC_LONG(bytes.size()),digest);
    std::string out;for(auto c:digest) {out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;
}
Json write(const std::filesystem::path& root,const std::string& name,std::span<const std::byte> bytes) {
    std::ofstream f(root/name,std::ios::binary);f.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
    check(bool(f),"cannot write fixture");return {{"file",name},{"bytes",bytes.size()},{"sha256",hash(bytes)}};
}
void save(const std::filesystem::path& path,const Json& value) {
    std::ofstream out(path);out<<value.dump(2)<<'\n';check(bool(out),"cannot save Q3 report");
}
std::vector<std::byte> read(const std::filesystem::path& root,const Json& file) {
    const std::filesystem::path name=file.at("file").get<std::string>();
    check(name==name.filename() && !name.empty(),"invalid fixture path");
    File f(root/name);const uint64_t size=file.at("bytes");check(size && size<=32*MiB && f.size()==size,"fixture size mismatch");
    std::vector<std::byte> bytes(size);f.read(0,bytes);check(hash(bytes)==file.at("sha256").get<std::string>(),"fixture hash mismatch");return bytes;
}
Buf upload(Metal& gpu,std::span<const std::byte> bytes) {auto b=gpu.allocate(bytes.size());std::memcpy(b->data,bytes.data(),bytes.size());return b;}
std::vector<float> dequantize(const Linear& l) {
    const auto* w=reinterpret_cast<const uint32_t*>(l.weight.buffer->data+l.weight.offset);
    const auto* s=reinterpret_cast<const uint16_t*>(l.scales.buffer->data+l.scales.offset);
    const auto* b=reinterpret_cast<const uint16_t*>(l.biases.buffer->data+l.biases.offset);
    std::vector<float> out(uint64_t(l.input)*l.output);
    for(size_t i=0;i<out.size();++i)out[i]=float((w[i/8]>>(4*(i%8)))&15)*q3_probe::value(s[i/64])+q3_probe::value(b[i/64]);
    return out;
}
// Independent representation check: expand codes into the already validated
// Q4 arithmetic, retaining the Q3 metadata. This is not the original Q4 model.
Linear expand(Metal& gpu,std::span<const std::byte> bytes) {
    constexpr size_t count=Hidden*Intermediate,groups=count/64;
    auto w=gpu.zeros(count/8),s=gpu.allocate(groups*2),b=gpu.allocate(groups*2);
    auto* words=reinterpret_cast<uint32_t*>(w->data);
    for(size_t i=0;i<count;++i) words[i/8]|=q3_probe::code(bytes,i)<<(4*(i%8));
    for(size_t i=0;i<groups;++i) {std::memcpy(s->data+i*2,bytes.data()+i*28+24,2);std::memcpy(b->data+i*2,bytes.data()+i*28+26,2);}
    return {{w},{s},{b},Hidden,Intermediate,64,0,true,4};
}
void capture(int argc,char** argv) {
    const std::filesystem::path out=argv[5];check(!std::filesystem::exists(out),"capture directory already exists");
    std::filesystem::create_directories(out);
    auto work=read_json(argv[4]);auto tokens=work.at("tokens").get<std::vector<int>>();
    const auto continuation=work.at("continuation").get<std::vector<int>>();
    check(tokens.size()==72 && continuation.size()==8,"capture requires 72 prompt and eight continuation tokens");
    const size_t slots=argc==7?std::stoul(argv[6]):1460;
    check(slots==1460 || slots==1072,"capture slots must be 1460 or 1072");
    Options o;o.model=argv[2];o.prepared=argv[3];o.artifact=Artifact::Mixed;o.memory=12*GiB;o.expert_slots=slots;o.panel=512;
    o.kernels.policy="candidate";o.kernels.q8_decode_rows=2;o.kernels.route_selection="simd";o.decode_scratch="reuse";
    Json manifest={{"kind","q3_probe_capture_v2"},{"complete",false},{"origin","existing-Q4-experts"},
        {"capture_mode","teacher-forced recorded continuation; no sampling or EOS override"},
        {"build",build_fingerprint()},{"artifact_revision",artifact_revision(o.artifact)},
        {"workload",work},{"quality_qualified",false},{"layers",Json::object()}};
    std::array<std::vector<uint32_t>,Layers> offsets,experts;
    std::array<std::vector<std::byte>,Layers> inputs;
    o.expert_observer=[&](ExpertKey key,const Buf& record,const Buf& x,uint32_t rows,const std::string& phase,uint32_t offset) {
        if(stopped) throw std::runtime_error("cancelled");
        if(phase!="decode" || rows!=1 || (key.layer!=0 && key.layer!=16 && key.layer!=32 && key.layer!=47))return;
        auto& item=manifest["layers"][std::to_string(key.layer)];
        if(item.is_null())item={{"experts",Json::array()}};
        if(offsets[key.layer].size()<8 && (offsets[key.layer].empty() || offsets[key.layer].back()!=offset)) {
            offsets[key.layer].push_back(offset);inputs[key.layer].insert(inputs[key.layer].end(),x->data,x->data+Hidden*4);
        }
        if(experts[key.layer].size()<2 && std::find(experts[key.layer].begin(),experts[key.layer].end(),key.expert)==experts[key.layer].end()) {
            experts[key.layer].push_back(key.expert);
            item["experts"].push_back({{"expert",key.expert},{"selected_at",offset},
                {"record",write(out,"expert-"+std::to_string(key.layer)+"-"+std::to_string(key.expert)+".bin",{record->data,ExpertBytes})}});
        }
    };
    save(out/"manifest.json",manifest);
    try {
        Model model(o);model.prepare_pipelines();auto state=model.make_state();
        check(model.memory_plan().limit==12*GiB && model.memory_plan().slots==slots,"Q3 capture memory admission differs");
        model.phase("prefill");model.prepare_ingest(tokens.size());model.forward(tokens,state,true,&stopped);model.finish_ingest();
        model.phase("decode");
        for(const auto id:continuation)model.forward(std::span(&id,1),state,true,&stopped);
        check(!stopped && state.valid && state.tokens==tokens.size()+continuation.size(),"incomplete Q3 capture request");
        model.diagnostic_drain();manifest["after"]=model.stats();
        for(auto layer:{0,16,32,47}) {
            check(offsets[layer].size()==8 && experts[layer].size()==2,"missing Q3 capture coverage");
            auto& item=manifest["layers"][std::to_string(layer)];item["offsets"]=offsets[layer];
            item["inputs"]=write(out,"inputs-"+std::to_string(layer)+".bin",inputs[layer]);
        }
        manifest["complete"]=true;
    } catch(const std::exception& e) {manifest["error"]=e.what();save(out/"manifest.json",manifest);throw;}
    save(out/"manifest.json",manifest);
}
void probe(char** argv) {
    const std::filesystem::path root=argv[2],output=argv[3];const bool validation=std::string(argv[4])=="validate";
    check(validation || std::string(argv[4])=="timing","probe mode must be validate or timing");
    if(!validation) check(!std::getenv("MTL_DEBUG_LAYER") && !std::getenv("MTL_SHADER_VALIDATION"),"timing cannot enable Metal validation");
    const auto manifest=read_json(root/"manifest.json");
    check(manifest.at("kind")=="q3_probe_capture_v2" && manifest.at("complete")==true && manifest.at("layers").size()==4,"incomplete capture");
    Metal gpu;gpu.budget(512*MiB);gpu.prepare_pipelines();
    Json result={{"kind","q3_probe_result_v1"},{"complete",false},{"build",build_fingerprint()},
        {"origin","Q4-to-Q3 exploratory; not original-weight quantization"},{"quality_qualified",false},
        {"capture_mode",manifest.at("capture_mode")},
        {"normal_request_latency_qualified",false},{"validation",validation},
        {"input_scope","real layer inputs; each row need not have selected both fixture experts"},{"cases",Json::array()}};
    save(output,result);
    for(auto layer:{0,16,32,47}) {
        const auto& item=manifest.at("layers").at(std::to_string(layer));
        check(item.at("experts").size()==2 && item.at("offsets").size()==8,"wrong fixture coverage");
        const auto raw_input=read(root,item.at("inputs"));check(raw_input.size()==8*Hidden*4,"wrong input size");
        auto x=upload(gpu,raw_input);
        for(const auto& e:item.at("experts")) {
            check(!stopped,"cancelled");auto raw=read(root,e.at("record"));check(raw.size()==ExpertBytes,"wrong expert size");
            auto record=upload(gpu,raw);auto gate=expert_linear(record,0),up=expert_linear(record,1),down=expert_linear(record,2);
            auto g=q3_probe::quantize(dequantize(gate)),u=q3_probe::quantize(dequantize(up));
            auto gb=upload(gpu,g),ub=upload(gpu,u);auto eg=expand(gpu,g),eu=expand(gpu,u);
            for(uint32_t rows:{1,2,4,8}) {
                auto hidden=gpu.allocate(rows*Intermediate*4),output_buffer=gpu.allocate(rows*Hidden*4);
                auto execute=[&](bool q3) {
                    if(q3) gpu.dispatch("q3_probe_gate_up",{{gb},{ub},{x},{hidden}},{Hidden,Intermediate,rows},Intermediate*32,rows);
                    else gpu.dispatch("q4_gate_up",{gate.weight,gate.scales,gate.biases,up.weight,up.scales,up.biases,{x},{x},{hidden}},
                        {Hidden,Intermediate,rows,64,0},Intermediate*32,rows);
                    gpu.linear_into(down,hidden,rows,{output_buffer});gpu.finish();
                };
                Json row={{"layer",layer},{"expert",e.at("expert")},{"rows",rows},
                    {"q4_record_bytes",ExpertBytes},{"q3_record_bytes",g.size()+u.size()+ExpertBytes/3},
                    {"q4_aligned_bytes",ExpertStride},{"q3_aligned_bytes",((g.size()+u.size()+ExpertBytes/3+16383)/16384)*16384}};
                execute(true);auto expected=gpu.gated_linear(eg,eu,x,rows);gpu.finish();
                check(std::memcmp(hidden->data,expected->data,hidden->bytes)==0,"Q3 differs from independent code expansion and reference arithmetic");
                auto final_expected=gpu.linear(down,expected,rows);gpu.finish();
                check(std::memcmp(output_buffer->data,final_expected->data,output_buffer->bytes)==0,"Q3 down-chain differs");
                row["exact_expanded_reference"]=true;
                if(!validation) {
                    row["pairs"]=Json::array();
                    for(int pair=0;pair<5;++pair) {
                        Json sample={{"pair",pair}};
                        for(int arm=0;arm<2;++arm) {
                            const auto before=process_memory();
                            bool q3=((pair+arm)%2)!=0;const auto start=monotonic_ns();execute(q3);const auto cold=monotonic_ns()-start;
                            execute(q3);const auto begin=monotonic_ns();for(int repeat=0;repeat<20;++repeat)execute(q3);
                            const auto duration=monotonic_ns()-begin;
                            sample[q3?"q3":"q4"]={{"setup_ns",cold},{"ns_per_chain",duration/20.0},
                                {"memory_before",before},{"memory_after",process_memory()}};
                        }
                        row["pairs"].push_back(sample);
                    }
                }
                result["cases"].push_back(row);save(output,result);
            }
        }
    }
    result["memory"]=process_memory();result["peak_gpu_bytes"]=gpu.peak();
    check(result["memory"].at("physical_footprint_peak_bytes").get<uint64_t>()<=2*GiB,"Q3 process budget exceeded");
    result["complete"]=true;save(output,result);
}
}
int main(int argc,char** argv) {
    std::signal(SIGINT,interrupt);std::signal(SIGTERM,interrupt);
    try {
        if((argc==6 || argc==7) && std::string(argv[1])=="capture")capture(argc,argv);
        else if(argc==5 && std::string(argv[1])=="probe")probe(argv);
        else throw std::invalid_argument("qwen_q3_probe capture MODEL PREPARED WORKLOAD OUTPUT [SLOTS] | probe FIXTURES REPORT validate|timing");
        return 0;
    }catch(const std::exception& e){std::println(stderr,"{}",e.what());return 1;}
}
