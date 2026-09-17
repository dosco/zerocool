// Developer prepared-MTP format check, using the existing native Q4/Q8 kernels.
// One-hot inputs expose decoded matrix columns without sharing GPU dot code.
#include "qwen/metal.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
using namespace freellm::qwen;
namespace {
void require(bool v,const char* message) {if(!v) throw std::runtime_error(message);}
std::string hash(std::span<const std::byte> bytes) {
    unsigned char d[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes.data(),CC_LONG(bytes.size()),d);
    std::string out;for(auto c:d){out+="0123456789abcdef"[c>>4];out+="0123456789abcdef"[c&15];}return out;
}
float expected(const Linear& l,uint32_t row,uint32_t col) {
    // Byte addressing is independent of Metal's packed word/lane indexing.
    const auto* codes=l.weight.buffer->data+l.weight.offset;
    const uint64_t element=uint64_t(row)*l.input+col;
    const uint8_t byte=std::to_integer<uint8_t>(codes[element*l.bits/8]);
    const uint32_t code=l.bits==8?byte:(byte>>(4*(element%2)))&15;
    const uint64_t group=element/l.group;uint16_t s,b;
    std::memcpy(&s,l.scales.buffer->data+l.scales.offset+group*2,2);
    std::memcpy(&b,l.biases.buffer->data+l.biases.offset+group*2,2);
    return round_bf16(float(code)*bf16(s)+bf16(b));
}
Json check_matrix(Metal& gpu,const Linear& l,const std::string& name) {
    const std::array<uint32_t,4> columns={0,31,63,l.input-1};Json cases=Json::array();
    for(uint32_t tokens:{1u,4u}) {
        KernelConfig config;config.policy="candidate";config.q8_decode_rows=2;config.token_tile=tokens;gpu.configure(config);
        const uint32_t calls=tokens==1?4:1;
        for(uint32_t call=0;call<calls;++call) {
            std::vector<float> values(uint64_t(tokens)*l.input,0);
            for(uint32_t row=0;row<tokens;++row) values[uint64_t(row)*l.input+columns[call+row]]=1;
            auto input=gpu.upload(values),output=gpu.linear(l,input,tokens);gpu.finish();
            require(output->bytes==uint64_t(tokens)*l.output*4,"MTP kernel returned wrong shape");
            for(uint32_t row=0;row<tokens;++row) for(uint32_t n=0;n<l.output;++n)
                require(output->floats()[uint64_t(row)*l.output+n]==expected(l,n,columns[call+row]),"MTP packed affine column differs from CPU decode");
            cases.push_back({{"tokens",tokens},{"column_start",call},{"output_values",uint64_t(tokens)*l.output},{"exact_decoded_columns",true}});
        }
    }
    return {{"name",name},{"input",l.input},{"output",l.output},{"bits",l.bits},{"cases",cases}};
}
}
int main(int argc,char** argv) {
    Json report={{"kind","native_mtp_prepared_check_v1"},{"complete",false},{"production_promoted",false},{"full_draft_verified",false},{"acceptance_qualified",false}};
    bool may_write=false;
    try {
        require(argc==3,"usage: check_mtp PREPARED_DIR REPORT");require(!std::filesystem::exists(argv[2]),"report exists");may_write=true;
        require(std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"),"MTP format check requires Metal validation");
        const std::filesystem::path root=argv[1];auto manifest=read_json(root/"manifest.json");const auto raw=read_text(root/"manifest.json");
        require(manifest.at("kind")=="qwen_mtp_prepared_v1" && manifest.at("complete")==true &&
            manifest.at("recipe")=="mtp-affine-q4-experts-q8-dense-64-v1" && manifest.at("source_revision")=="de4b8e4d43b917e7706784d8bb445c9af86a3540","wrong MTP sidecar");
        report.update(Json{{"manifest_sha256",hash(std::as_bytes(std::span(raw)))},{"before",process_memory()},
            {"host_before",host_conditions()},{"cases",Json::array()},{"validation",true},{"byte_budget",192*MiB}});
        {
            Metal gpu;gpu.budget(192*MiB);gpu.prepare_pipelines();gpu.request_phase("decode");
            File dense_file(root/"dense.bin");require(dense_file.size()==manifest["files"]["dense.bin"]["bytes"] && dense_file.size()<=128*MiB,"wrong dense MTP size");
            auto dense=gpu.allocate(dense_file.size());dense_file.read(0,{dense->data,size_t(dense->bytes)});
            require(hash({dense->data,size_t(dense->bytes)})==manifest["files"]["dense.bin"]["sha256"].get<std::string>(),"changed dense MTP payload");
            for(auto& [name,tensor]:manifest.at("tensors").items()) {
                if(tensor.at("format")!="affine-Q8") continue;
                require(tensor.at("bits")==8 && tensor.at("group_size")==64 && tensor.at("shape").size()==2,"unsupported MTP affine format");
                Linear l;l.quantized=true;l.bits=8;l.group=64;l.input=tensor["shape"][1];l.output=tensor["shape"][0];
                auto part=[&](const char* name,uint64_t bytes) {auto p=tensor["parts"].at(name);const uint64_t offset=p.at("offset");
                    require(p.at("bytes")==bytes && offset%16384==0 && offset+bytes<=dense->bytes,"MTP tensor outside prepared allocation");return Binding{dense,offset};};
                l.weight=part("weight",uint64_t(l.input)*l.output);l.scales=part("scales",uint64_t(l.input)*l.output/64*2);l.biases=part("biases",uint64_t(l.input)*l.output/64*2);
                report["cases"].push_back(check_matrix(gpu,l,name));
            }
            require(report["cases"].size()==16,"missing MTP dense matrix coverage");
            File experts(root/"experts.bin");require(experts.size()==512*ExpertStride,"wrong MTP expert file size");
            for(int expert:{0,256,511}) {
                auto record=gpu.allocate(ExpertStride);experts.read(uint64_t(expert)*ExpertStride,{record->data,size_t(record->bytes)});
                require(hash({record->data,size_t(record->bytes)})==manifest["experts"]["record_sha256"][expert].get<std::string>(),"changed MTP expert record");
                for(int projection=0;projection<3;++projection)report["cases"].push_back(check_matrix(gpu,expert_linear(record,projection),"expert"+std::to_string(expert)+"/projection"+std::to_string(projection)));
            }
            gpu.finish();report["metal"]=gpu.statistics();report["after"]=process_memory();
            require(report["metal"]["live_command_groups"]==0 && gpu.peak()<=192*MiB,"MTP validation retained excessive GPU users");
        }
        report["after_destroy"]=process_memory();report["host_after"]=host_conditions();report["complete"]=true;
        std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';require(bool(out),"cannot write MTP check");
        std::cout<<"25 actual MTP matrices passed native Q4/Q8 column checks at one and four tokens\n";return 0;
    } catch(const std::exception& e) {report["error"]=e.what();if(may_write){std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';}std::cerr<<e.what()<<'\n';return 2;}
}
