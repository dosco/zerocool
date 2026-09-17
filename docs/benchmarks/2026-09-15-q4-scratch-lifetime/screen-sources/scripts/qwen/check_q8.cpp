#include "qwen/metal.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <print>

using namespace freellm::qwen;
namespace {
std::string sha(std::span<const std::byte> bytes) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(bytes.data(),CC_LONG(bytes.size()),digest);
    std::string hex;constexpr char digits[]="0123456789abcdef";
    for(auto c:digest) {hex+=digits[c>>4];hex+=digits[c&15];}return hex;
}
Json compare(const Buf& actual,const Buf& expected) {
    if(actual->bytes!=expected->bytes || actual->bytes%4) throw std::runtime_error("Q8 output shape mismatch");
    double error=0,norm=0,max_error=0;uint64_t differing=0;
    for(size_t i=0;i<actual->bytes/4;++i) {
        const auto a=actual->floats()[i],b=expected->floats()[i];
        if(!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("non-finite Q8 output");
        double delta=double(a)-b;error+=delta*delta;norm+=double(b)*b;
        max_error=std::max(max_error,std::abs(delta));differing+=a!=b;
    }
    return {{"bit_identical",std::memcmp(actual->data,expected->data,actual->bytes)==0},
            {"differing_values",differing},{"values",actual->bytes/4},
            {"relative_l2",std::sqrt(error/std::max(norm,1e-30))},{"max_abs",max_error}};
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::invalid_argument("usage: qwen_q8_check FIXTURE_DIR SOURCE_LOCK REPORT");
        const std::filesystem::path root(argv[1]);const auto text=read_text(root/"manifest.json");
        const auto manifest=Json::parse(text),lock=read_json(argv[2]);
        if(manifest.at("schema")!=1 || manifest.at("kind")!="mixed_q8_operator_reference" ||
           manifest.at("source")!=lock || manifest.at("mlx")!="0.31.1" || manifest.at("cases").empty())
            throw std::runtime_error("incompatible or unpinned Q8 fixture");
        Metal gpu;gpu.budget(128*MiB);Json cases=Json::array();bool passed=true;
        auto load=[&](const std::string& name) {
            if(std::filesystem::path(name).filename()!=name) throw std::runtime_error("fixture path must be a filename");
            File f(root/name);const auto& entry=manifest.at("files").at(name);
            if(!f.size() || f.size()>16*MiB || f.size()!=entry.at("bytes")) throw std::runtime_error("fixture size mismatch");
            auto b=gpu.allocate(f.size());f.read(0,{b->data,size_t(b->bytes)});
            if(sha({b->data,size_t(b->bytes)})!=entry.at("sha256").get<std::string>()) throw std::runtime_error("fixture hash mismatch");
            return b;
        };
        auto matrix=[&](const std::string& name) {
            const auto& m=manifest.at("matrices").at(name);const auto& t=m.at("tensors");
            if(m.at("bits")!=8 || m.at("group")!=64) throw std::runtime_error("expected Q8/64 fixture");
            return Linear{{load(t.at("weight").at("file"))},{load(t.at("scales").at("file"))},
                {load(t.at("biases").at("file"))},m.at("input").get<uint32_t>(),m.at("rows").get<uint32_t>(),64,0,true,8};
        };
        for(const auto& c:manifest.at("cases")) {
            auto l=matrix(c.at("matrix"));Buf result;
            const auto op=c.at("op").get<std::string>();
            if(op=="embedding") result=gpu.embedding(l,c.at("ids").get<std::vector<int>>(),c.at("copies"));
            else {
                auto x=load(c.at("input"));const auto tokens=c.at("tokens").get<uint32_t>();
                if(op=="linear") result=gpu.linear(l,x,tokens);
                else if(op=="gated") {
                    auto indices=c.at("rows").get<std::vector<int>>();
                    if(indices.size()!=tokens) throw std::runtime_error("fixture row count");
                    auto rows=gpu.allocate(indices.size()*4);std::memcpy(rows->data,indices.data(),rows->bytes);
                    result=gpu.gated_linear(l,matrix(c.at("up")),x,tokens,rows);
                } else throw std::runtime_error("unknown Q8 fixture operator");
            }
            gpu.finish();auto comparison=compare(result,load(c.at("expected")));
            passed=passed && comparison.at("bit_identical").get<bool>();
            Json row={{"name",c.at("name")},{"fixed_qmv_reference",comparison}};
            if(c.contains("batch_expected")) row["mlx_batch_comparison"]=compare(result,load(c.at("batch_expected")));
            cases.push_back(std::move(row));
        }
        Json report={{"kind","real_mixed_q8_operator_check"},{"passed",passed},{"source",lock},
            {"fixture_manifest_sha256",sha({reinterpret_cast<const std::byte*>(text.data()),text.size()})},
            {"machine",gpu.statistics()},{"cases",cases},{"full_model_verified",false},{"quality_qualified",false}};
        std::ofstream out(argv[3]);out<<report.dump(2)<<'\n';if(!out) throw std::runtime_error("cannot write Q8 report");
        std::println("{} real Q8 cases: {}",cases.size(),passed?"bit-identical to fixed MLX QMV":"numerical mismatch");
        return passed?0:1;
    } catch(const std::exception& e) {std::println(stderr,"Q8 check: {}",e.what());return 1;}
}
