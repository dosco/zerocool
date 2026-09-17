// Developer-only mixed-Q8 shared-expert reference and queued prelude.
#pragma once
namespace {
class SharedQ4Prelude {
    static constexpr std::array<int,4> layers={0,16,32,47};
    static constexpr std::array<uint32_t,3> widths={640,2560,1};
    Metal& gpu_;Resident& resident_;
    std::array<std::array<std::vector<float>,3>,4> cpu_;
    std::array<std::array<std::array<std::vector<std::byte>,3>,8>,4> native_;
    std::array<Buf,3> last_;
    Json manifest_,checks_=Json::array();std::string manifest_sha_,native_sha_;
    static std::vector<float> payload(const std::filesystem::path& root,const Json& entry,uint32_t width) {
        const auto name=entry.at("file").get<std::string>();
        require(!name.empty() && name!="." && name!=".." && std::filesystem::path(name).filename()==name,"invalid shared fixture path");
        File file(root/name);const uint64_t bytes=8ull*width*4;
        require(file.size()==bytes && entry.at("bytes")==bytes && entry.at("shape")==Json::array({8,width}),"shared fixture shape differs");
        std::vector<float> out(8*width);file.read(0,{reinterpret_cast<std::byte*>(out.data()),size_t(bytes)});
        require(hash({reinterpret_cast<const std::byte*>(out.data()),size_t(bytes)})==entry.at("sha256").get<std::string>(),"shared fixture hash differs");
        for(auto v:out) require(std::isfinite(v),"nonfinite CPU shared reference");return out;
    }
    static Json vector_error(std::span<const float> actual,std::span<const float> expected) {
        double error=0,aa=0,bb=0,dot=0;
        for(size_t i=0;i<actual.size();++i) {
            require(std::isfinite(actual[i]),"nonfinite native shared output");
            const double a=actual[i],b=expected[i],d=a-b;error+=d*d;aa+=a*a;bb+=b*b;dot+=a*b;
        }
        const double relative=bb?std::sqrt(error/bb):(error?std::numeric_limits<double>::infinity():0);
        const double cosine=aa && bb?dot/std::sqrt(aa*bb):(aa==bb?1:0);
        require(relative<=0.01 && cosine>=0.99995,"native shared output differs from independent CPU reference");
        return {{"relative_l2",relative},{"cosine",cosine}};
    }
public:
    SharedQ4Prelude(Metal& gpu,Resident& resident,const Checkpoint& checkpoint,
                    const std::filesystem::path& directory,const Json& expert_manifest,
                    const std::filesystem::path& expert_root):gpu_(gpu),resident_(resident) {
        const auto text=read_text(directory/"manifest.json");manifest_=Json::parse(text);
        manifest_sha_=hash({reinterpret_cast<const std::byte*>(text.data()),text.size()});
        const auto expert_text=read_text(expert_root/"manifest.json");
        const Json tolerance={{"vector_relative_l2_max",0.01},{"vector_cosine_min",0.99995},
            {"scalar_absolute_max",1e-6},{"scalar_relative_max",1.0/128}};
        require(manifest_.at("kind")=="independent_shared_expert_reference_v1" && manifest_.at("tolerance")==tolerance &&
            manifest_.at("complete")==true && manifest_.at("artifact_revision")==checkpoint.revision() &&
            manifest_.at("fixture_manifest_sha256")==hash({reinterpret_cast<const std::byte*>(expert_text.data()),expert_text.size()}) &&
            manifest_.at("layers").size()==4,"shared reference identity differs");
        for(size_t li=0;li<layers.size();++li) {
            const auto& item=manifest_.at("layers").at(std::to_string(layers[li]));
            require(item.at("inputs").at("sha256")==expert_manifest.at("layers").at(std::to_string(layers[li])).at("inputs").at("sha256"),"shared input identity differs");
            const auto prefix="model.layers."+std::to_string(layers[li])+".mlp.";
            std::vector<std::string> names;
            for(const auto part:{"gate_proj","up_proj","down_proj"}) for(const auto suffix:{".weight",".scales",".biases"})
                names.push_back(prefix+"shared_expert."+part+suffix);
            names.push_back(prefix+"shared_expert_gate.weight");
            require(item.at("tensors").size()==names.size(),"shared tensor coverage differs");
            for(const auto& name:names) {
                const auto& expected=item.at("tensors").at(name);const auto& tensor=checkpoint.at(name);const auto& value=resident_.at(name);
                require(expected.at("bytes")==tensor.bytes && expected.at("shape")==tensor.shape && expected.at("dtype")==tensor.dtype &&
                    value->bytes>=tensor.bytes && hash({value->data,size_t(tensor.bytes)})==expected.at("sha256").get<std::string>(),"CPU reference resident weight identity differs");
            }
            const std::array<const char*,3> fields={"activation","shared","gate"};
            for(size_t i=0;i<3;++i) cpu_[li][i]=payload(directory,item.at(fields[i]),widths[i]);
        }
    }
    void queue(int layer,uint32_t row,const Buf& input) {
        require(row<8,"shared row out of range");
        const auto prefix="model.layers."+std::to_string(layer)+".mlp.";
        gpu_.label("shared_expert",layer,1,72+row);
        last_[0]=gpu_.gated_linear(resident_.linear(prefix+"shared_expert.gate_proj"),resident_.linear(prefix+"shared_expert.up_proj"),input,1);
        last_[1]=gpu_.linear(resident_.linear(prefix+"shared_expert.down_proj"),last_[0],1);
        last_[2]=gpu_.linear(resident_.linear(prefix+"shared_expert_gate"),input,1);
    }
    void clear() {last_={};}
    template<class Input> void prepare(Input input) {
        std::vector<std::byte> bytes;
        for(size_t li=0;li<layers.size();++li) for(uint32_t row=0;row<8;++row) {
            gpu_.begin_scratch(0,128*MiB);queue(layers[li],row,input(layers[li],row));gpu_.end_scratch();gpu_.finish();
            Json check={{"layer",layers[li]},{"row",row}};
            const std::array<const char*,3> fields={"activation","shared","gate"};
            for(size_t i=0;i<3;++i) {
                require(last_[i]->bytes==widths[i]*4ull,"shared native output shape differs");
                auto actual=last_[i]->floats();const auto expected=std::span(cpu_[li][i]).subspan(row*widths[i],widths[i]);
                if(i<2) check[fields[i]]=vector_error(actual,expected);
                else {
                    require(std::isfinite(actual[0]),"nonfinite native shared gate");
                    const double error=std::abs(double(actual[0])-expected[0]);
                    const double limit=1e-6+std::abs(double(expected[0]))/128;
                    require(error<=limit,"native scalar shared gate differs from independent CPU reference");
                    check["gate"]={{"expected",expected[0]},{"absolute_error",error},{"limit",limit}};
                }
                native_[li][row][i].assign(last_[i]->data,last_[i]->data+last_[i]->bytes);
                bytes.insert(bytes.end(),native_[li][row][i].begin(),native_[li][row][i].end());
            }
            checks_.push_back(std::move(check));clear();
        }
        native_sha_=hash(bytes);
    }
    void check(uint32_t batch) const {
        const auto li=batch%4;
        for(size_t i=0;i<3;++i) require(last_[i] && last_[i]->bytes==native_[li][batch][i].size() &&
            std::memcmp(last_[i]->data,native_[li][batch][i].data(),last_[i]->bytes)==0,"queued shared outputs changed bytes");
    }
    static int layer(uint32_t batch) {return layers[batch%4];}
    Json report() const {
        return {{"enabled",true},{"reference_manifest",manifest_},{"reference_manifest_sha256",manifest_sha_},
            {"expected_native_sha256",native_sha_},{"cpu_reference_passed",checks_.size()==32},{"shared_outputs_exact",true},
            {"reference_checks",checks_},{"layer_order",{0,16,32,47,0,16,32,47}},
            {"reference_host_payload_bytes",2ull*4*8*(640+2560+1)*4},
            {"gpu_scope","shared-and-routed-command-intervals"}};
    }
};
}
