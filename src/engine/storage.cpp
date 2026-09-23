#include "engine/storage.hpp"
#include "qwen_embedded.hpp"

#include <algorithm>
#include <CommonCrypto/CommonDigest.h>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>
#include <mach/mach_time.h>

namespace zerocool::engine {
const char* build_fingerprint() {return BuildFingerprint;}
uint64_t checked_add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("byte offset overflow");
    return a + b;
}
uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("tensor size overflow");
    return a * b;
}
float bf16(uint16_t v) { return std::bit_cast<float>(uint32_t(v) << 16); }
float round_bf16(float v) {
    auto bits = std::bit_cast<uint32_t>(v);
    if ((bits & 0x7fffffff) > 0x7f800000) return std::numeric_limits<float>::quiet_NaN();
    bits += 0x7fff + ((bits >> 16) & 1);
    return std::bit_cast<float>(bits & 0xffff0000);
}
float fp16(uint16_t v) {
    const int sign = (v & 0x8000) ? -1 : 1;
    const int exp = (v >> 10) & 31, frac = v & 1023;
    if (exp == 31) return frac ? NAN : sign * INFINITY;
    return sign * (exp ? std::ldexp(float(1024 + frac), exp - 25) : std::ldexp(float(frac), -24));
}
std::string read_text(const std::filesystem::path& p, uint64_t limit) {
    File f(p, false);
    if (f.size() > limit) throw std::runtime_error("metadata exceeds limit: " + p.string());
    std::string text(f.size(), '\0');
    f.read(0, {reinterpret_cast<std::byte*>(text.data()), text.size()});
    return text;
}
Json read_json(const std::filesystem::path& p, uint64_t limit) { return Json::parse(read_text(p, limit)); }
Buf Buffer::host(uint64_t bytes) {
    if (!bytes || bytes > SIZE_MAX) throw std::invalid_argument("invalid buffer size");
    void* p = nullptr;
    if (posix_memalign(&p, 16384, bytes)) throw std::bad_alloc();
    return std::make_shared<Buffer>(Buffer{bytes, static_cast<std::byte*>(p),
        std::shared_ptr<void>(p, [](void* v) { free(v); }), nullptr});
}
std::span<float> Buffer::floats() const {
    if (bytes % sizeof(float)) throw std::logic_error("buffer is not float-sized");
    return {reinterpret_cast<float*>(data), size_t(bytes / sizeof(float))};
}
File::File(const std::filesystem::path& path, bool uncached) : name_(path.string()) {
    fd_ = open(name_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "open " + name_);
    struct stat st {};
    if (fstat(fd_, &st) || !S_ISREG(st.st_mode) || st.st_size < 0) {
        close(fd_); fd_ = -1;
        throw std::runtime_error("not a readable regular file: " + name_);
    }
    size_ = uint64_t(st.st_size);
#ifdef F_NOCACHE
    if (uncached && fcntl(fd_, F_NOCACHE, 1) != 0) {
        const int err = errno; close(fd_); fd_ = -1;
        throw std::system_error(err, std::generic_category(), "F_NOCACHE " + name_);
    }
#else
    (void)uncached;
#endif
}
File::~File() { if (fd_ >= 0) close(fd_); }
void File::read(uint64_t offset, std::span<std::byte> into) const {
    if (offset > size_ || into.size() > size_ - offset || offset > uint64_t(INT64_MAX))
        throw std::out_of_range("read outside file: " + name_);
    size_t done = 0;
    while (done < into.size()) {
        const size_t count = std::min<size_t>(into.size() - done, 32 * MiB);
        const ssize_t n = pread(fd_, into.data() + done, count, off_t(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) throw std::system_error(errno, std::generic_category(), "pread " + name_);
        if (!n) throw std::runtime_error("truncated file: " + name_);
        done += size_t(n); read_bytes += uint64_t(n); ++read_calls;
    }
}
uint64_t TensorRef::elements() const {
    uint64_t n = 1;
    for (auto d : shape) n = checked_mul(n, d);
    return n;
}
uint64_t TensorRef::row_bytes() const {
    if (shape.empty() || !shape[0] || bytes % shape[0]) throw std::runtime_error("invalid row geometry");
    return bytes / shape[0];
}
bool is_resident(const std::string& key,int layers) {
    if(layers<Layers) {
        if(key.starts_with("lm_head.") || key.starts_with("model.hyper_connection_mixer.")) return false;
        if(key.starts_with("model.layers.") && std::stoi(key.substr(13))>=layers) return false;
    }
    return !key.starts_with("mtp.") && !key.starts_with("vision_tower.") &&
        !key.starts_with("model.visual.") && !key.contains("ngram_embedding.shard_") &&
        !key.contains(".switch_mlp.");
}
Json artifact_lock(Artifact artifact) {
    switch(artifact) {
        case Artifact::Q4: return Json::parse(ModelLock);
        case Artifact::Mixed: return Json::parse(MixedModelLock);
    }
    throw std::invalid_argument("unknown artifact");
}
const char* artifact_revision(Artifact artifact) {
    switch(artifact) {
        case Artifact::Q4: return ModelRevision;
        case Artifact::Mixed: return "b2c422f3c643e36f04227a64d61796b44a4b1029";
    }
    throw std::invalid_argument("unknown artifact");
}
const char* artifact_model_id(Artifact artifact) {
    switch(artifact) {
        case Artifact::Q4: return "qwen3.8-flash-next:4bit";
        case Artifact::Mixed: return "qwen3.8-flash-next:mixed-4_8bit";
    }
    throw std::invalid_argument("unknown artifact");
}
void verify_identity(const std::filesystem::path& directory,Artifact artifact) {
    const auto lock=artifact_lock(artifact);
    const auto receipt=read_json(directory/"zerocool-verification.json");
    const std::string remedy="; run scripts/qwen/verify_checkpoint.py --model "+directory.string()+
        (artifact==Artifact::Mixed?" --lock mixed-models.lock.json":"");
    if(receipt.contains("revision") && receipt.at("revision")!=artifact_revision(artifact))
        throw std::runtime_error("selected artifact differs from the verified model directory; choose matching --artifact and --model values");
    if(receipt.value("schema",0)!=1 || receipt.value("revision","")!=artifact_revision(artifact) || !receipt.contains("files"))
        throw std::runtime_error("missing or stale verification receipt"+remedy);
    for(const auto& entry:lock.at("files")) {
        if(entry.value("optional",false)) continue;
        const auto name=entry.at("path").get<std::string>();
        if(!receipt.at("files").contains(name)) throw std::runtime_error("unverified model file: "+name+remedy);
        const auto& saved=receipt.at("files").at(name);
        struct stat st{};
        if(::stat((directory/name).c_str(),&st)!=0) throw std::runtime_error("missing model file: "+name+remedy);
        const auto mtime=uint64_t(st.st_mtimespec.tv_sec)*1000000000+st.st_mtimespec.tv_nsec;
        const auto ctime=uint64_t(st.st_ctimespec.tv_sec)*1000000000+st.st_ctimespec.tv_nsec;
        if(saved.value("sha256","")!=entry.at("sha256").get<std::string>() || uint64_t(st.st_size)!=entry.at("size").get<uint64_t>() ||
           saved.value("size",uint64_t(0))!=uint64_t(st.st_size) || saved.value("device",uint64_t(0))!=uint64_t(st.st_dev) ||
           saved.value("inode",uint64_t(0))!=uint64_t(st.st_ino) || saved.value("mtime_ns",uint64_t(0))!=mtime ||
           saved.value("ctime_ns",uint64_t(0))!=ctime)
            throw std::runtime_error("model file changed since verification: "+name+remedy);
    }
}
Checkpoint::Checkpoint(const std::filesystem::path& dir, bool validate_model,Artifact artifact) : directory(dir),artifact_(artifact) {
    if(validate_model) verify_identity(dir,artifact_);
    config = read_json(dir / "config.json");
    auto index = read_json(dir / "model.safetensors.index.json");
    std::vector<std::string> names;
    for (const auto& name : index.at("weight_map")) {
        const auto file = name.get<std::string>();
        if (std::filesystem::path(file).filename() != file || file == "." || file == "..")
            throw std::runtime_error("unsafe shard name in checkpoint");
        names.push_back(file);
    }
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    if (names.empty() || names.size() > 128) throw std::runtime_error("invalid shard count");
    for (const auto& name : names) {
        auto f = std::make_shared<File>(dir / name);
        uint64_t header = 0;
        f->read(0, {reinterpret_cast<std::byte*>(&header), 8});
        if (header > 16 * MiB || checked_add(header, 8) > f->size())
            throw std::runtime_error("invalid safetensors header: " + name);
        std::string text(header, '\0');
        f->read(8, {reinterpret_cast<std::byte*>(text.data()), text.size()});
        auto meta = Json::parse(text);
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (auto it = meta.begin(); it != meta.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            const auto& t = it.value();
            auto key = it.key();
            if (key.starts_with("language_model.")) key.erase(0, 15);
            const auto start = t.at("data_offsets").at(0).get<uint64_t>();
            const auto end = t.at("data_offsets").at(1).get<uint64_t>();
            if (end < start || checked_add(header + 8, end) > f->size())
                throw std::runtime_error("invalid tensor range: " + key);
            TensorRef ref{f, header + 8 + start, end - start,
                          t.at("shape").get<std::vector<uint64_t>>(), t.at("dtype").get<std::string>()};
            const uint64_t unit = ref.dtype == "BF16" || ref.dtype == "F16" ? 2 :
                ref.dtype == "U32" || ref.dtype == "F32" || ref.dtype == "I32" ? 4 :
                ref.dtype == "I64" || ref.dtype == "U64" ? 8 : 0;
            if (!unit || checked_mul(ref.elements(), unit) != ref.bytes)
                throw std::runtime_error("unsupported dtype or size: " + key);
            if (!tensors.emplace(key, ref).second) throw std::runtime_error("duplicate tensor: " + key);
            if (end > start) ranges.emplace_back(start, end);
            if (!index.at("weight_map").contains(it.key()) || index.at("weight_map").at(it.key()) != name)
                throw std::runtime_error("index/header disagreement: " + key);
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i)
            if (ranges[i].first < ranges[i - 1].second) throw std::runtime_error("overlapping tensor ranges");
        files_.push_back(std::move(f));
    }
    if (tensors.size() != index.at("weight_map").size()) throw std::runtime_error("index tensors missing from shards");
    if (validate_model) validate();
}
void Checkpoint::validate() const {
    if (config.at("model_type") != "qwen4_exp") throw std::runtime_error("requires Qwen3.8-Flash-Next");
    const auto& t = config.at("text_config");
    const Json required = {{"hidden_size", Hidden}, {"num_hidden_layers", Layers}, {"num_experts", Experts},
        {"num_experts_per_tok", TopK}, {"moe_intermediate_size", Intermediate}, {"vocab_size", Vocab},
        {"num_attention_heads",24}, {"num_key_value_heads",2}, {"head_dim",256}, {"hc_count",4},
        {"hc_lowrank",320}, {"linear_num_key_heads",16}, {"linear_num_value_heads",48},
        {"linear_key_head_dim",128}, {"linear_value_head_dim",128}, {"linear_conv_kernel_dim",4},
        {"indexer_n_heads",4}, {"indexer_head_dim",128}, {"indexer_budget",2048},
        {"indexer_compress_ratio",4}, {"ngram_size",3}, {"heads_per_ngram",8},
        {"split_ngram_parts",128}, {"ple_embed_dim",2560}, {"ple_conv_kernel_size",4},
        {"shared_expert_intermediate_size",640}, {"output_gate_type","sigmoid"},
        {"ple_layer_ids",Json::array({2})}, {"eos_token_id",EndOfText}, {"rms_norm_eps",1e-6}};
    for (auto it = required.begin(); it != required.end(); ++it)
        if (!t.contains(it.key()) || t.at(it.key()) != it.value())
            throw std::runtime_error("unsupported Qwen configuration: " + it.key());
    for (int l = 0; l < Layers; ++l)
        if (t.at("layer_types").at(l) != ((l+1)%4 ? "linear_attention" : "full_attention"))
            throw std::runtime_error("unsupported layer order");
    if (t.at("rope_parameters").at("rope_theta") != 10000000 ||
        t.at("rope_parameters").at("partial_rotary_factor") != 0.25)
        throw std::runtime_error("unsupported rotary configuration");
    const auto& q = config.at("quantization");
    if (q.at("bits") != 4 || q.at("group_size") != 64) throw std::runtime_error("requires affine Q4 group 64");
    for(const auto& [name,ref]:tensors) if(is_resident(name) && name.ends_with(".scales")) {
        (void)ref;
        const auto base=name.substr(0,name.size()-7);
        const auto fmt=q.value(base,Json::object());
        const auto bits=fmt.value("bits",q.at("bits").get<int>());
        const auto group=fmt.value("group_size",q.at("group_size").get<int>());
        if(bits!=(artifact_==Artifact::Mixed?8:4) || group!=64)
            throw std::runtime_error("resident precision differs from selected artifact: "+base);
    }
    for (int s = 0; s < 128; ++s) {
        const auto b = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_" + std::to_string(s);
        if (q.at(b).at("bits") != 4 || q.at(b).at("group_size") != 32)
            throw std::runtime_error("requires affine Q4 group 32 ngrams");
    }
    (void)ExpertStore(*this);
}
const TensorRef& Checkpoint::at(const std::string& key) const {
    auto it = tensors.find(key);
    if (it == tensors.end()) throw std::runtime_error("missing tensor: " + key);
    return it->second;
}
bool Checkpoint::contains(const std::string& key) const { return tensors.contains(key); }
Buf Checkpoint::load(const std::string& key, const Allocator& alloc) const {
    const auto& r = at(key);
    auto out = alloc(r.bytes);
    r.file->read(r.offset, {out->data, size_t(r.bytes)});
    return out;
}
uint64_t Checkpoint::resident_bytes(int layers) const {
    uint64_t bytes = 0;
    for (const auto& [key,r] : tensors) if (is_resident(key,layers)) bytes = checked_add(bytes, (r.bytes + 16383) / 16384 * 16384);
    return bytes;
}
uint64_t Checkpoint::bytes_read() const {
    uint64_t n = 0; for (const auto& f : files_) n += f->read_bytes.load(); return n;
}
uint64_t Checkpoint::diagnostic_resident_bytes(int layers) const {
    uint64_t common=0;std::array<uint64_t,Layers> layer{};
    for(const auto& [key,r]:tensors) if(is_resident(key,layers)) {
        const auto charge=(r.bytes+16383)/16384*16384;
        if(key.starts_with("model.layers.")) layer.at(std::stoi(key.substr(13)))+=charge;
        else common+=charge;
    }
    return common+*std::max_element(layer.begin(),layer.end());
}
Json Checkpoint::inspect() const {
    uint64_t all = 0, ngram = 0, expert = 0;
    for (const auto& [key,r] : tensors) {
        all += r.bytes;
        if (key.contains("ngram_embedding.shard_")) ngram += r.bytes;
        if (key.contains(".switch_mlp.")) expert += r.bytes;
    }
    return {{"model","Qwen3.8-Flash-Next"}, {"revision",revision()}, {"model_id",model_id()}, {"tensors",tensors.size()},
        {"tensor_bytes",all}, {"resident_allocation_bytes",resident_bytes()}, {"expert_bytes",expert},
        {"ngram_bytes",ngram}, {"expert_record_bytes",ExpertBytes},
        {"source_format",artifact_==Artifact::Mixed?"MLX affine mixed Q4/Q8":"MLX affine Q4"},
        {"resident_affine_bits",artifact_==Artifact::Mixed?8:4},{"expert_affine_bits",4},{"ngram_affine_bits",4},
        {"identity_status","pinned hash receipt and current file fingerprints validated"}};
}

void verify_prepared_compatibility(Artifact consumer,const std::string& manifest_sha256) {
    if(consumer==Artifact::Mixed) {
        // All 816 expert/ngram tensors were compared in full before this
        // compatibility receipt was pinned. Resident weights still come from
        // the independently verified mixed checkpoint. Any source-lock change
        // requires renewed evidence; a matching layout alone cannot admit reuse.
        const auto reuse=Json::parse(MixedReuseLock);
        const auto sha=[](std::string_view value) {
            unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(value.data(),CC_LONG(value.size()),digest);
            std::string result;constexpr char digits[]="0123456789abcdef";
            for(auto c:digest) {result+=digits[c>>4];result+=digits[c&15];}return result;
        };
        if(reuse.at("schema")!=1 || reuse.at("source_revision")!=ModelRevision ||
           reuse.at("consumer_revision")!=artifact_revision(consumer) || reuse.at("prepared_manifest_sha256")!=manifest_sha256 ||
           reuse.at("file_locks_sha256").at("models.lock.json")!=sha(ModelLock) ||
           reuse.at("file_locks_sha256").at("mixed-models.lock.json")!=sha(MixedModelLock) ||
           reuse.at("tensor_count")!=816 || reuse.at("expert_payload_bytes")!=ExpertBytes*Layers*Experts ||
           reuse.at("ngram_payload_bytes")!=32000153600ull)
            throw std::runtime_error("prepared payload compatibility needs renewed mixed/Q4 verification");
    }
}

PreparedArtifact::PreparedArtifact(const std::filesystem::path& dir,const Checkpoint& source) : consumer_(source.artifact()) {
    const auto text=read_text(dir/"manifest.json");manifest_=Json::parse(text);
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];CC_SHA256(text.data(),CC_LONG(text.size()),digest);
    constexpr char hex[]="0123456789abcdef";
    for(auto c:digest) {identity_+=hex[c>>4];identity_+=hex[c&15];}
    const auto receipt=read_json(dir/"verification.json");
    if(manifest_.value("schema",0)!=1 || manifest_.value("format","")!="zc-affine-records-v1" ||
       manifest_.value("source_revision","")!=ModelRevision || receipt.value("manifest_sha256","")!=identity_ ||
       receipt.value("source_revision","")!=ModelRevision || manifest_.value("recipe","")!="q4-control-lossless")
        throw std::runtime_error("unverified or incompatible prepared artifact; run prepare_storage.py --verify");
    const auto lock=Json::parse(ModelLock);
    if(lock.at("prepared_control").at("manifest_sha256")!=identity_)
        throw std::runtime_error("prepared manifest differs from pinned lossless Q4 control");
    verify_prepared_compatibility(consumer_,identity_);
    for(const auto& entry:lock.at("files")) if(!entry.value("optional",false)) {
        const auto name=entry.at("path").get<std::string>();
        if(manifest_.at("source_files").at(name)!=entry.at("sha256"))
            throw std::runtime_error("prepared source identity differs from checkpoint");
    }
    std::unordered_map<std::string,std::shared_ptr<File>> files;
    uint64_t bytes=0;
    for(const auto& entry:manifest_.at("files")) {
        const auto name=entry.at("path").get<std::string>();
        if(name.empty() || std::filesystem::path(name).filename()!=name || name=="." || name==".." || files.contains(name))
            throw std::runtime_error("invalid prepared filename");
        const auto& saved=receipt.at("files").at(name);struct stat st{};
        if(::stat((dir/name).c_str(),&st)!=0) throw std::runtime_error("missing prepared file");
        const auto mtime=uint64_t(st.st_mtimespec.tv_sec)*1000000000+st.st_mtimespec.tv_nsec;
        const auto ctime=uint64_t(st.st_ctimespec.tv_sec)*1000000000+st.st_ctimespec.tv_nsec;
        if(st.st_size<0 || entry.at("size").get<uint64_t>()!=uint64_t(st.st_size) ||
           saved.at("sha256")!=entry.at("sha256") || saved.at("size").get<uint64_t>()!=uint64_t(st.st_size) ||
           saved.at("device").get<uint64_t>()!=uint64_t(st.st_dev) || saved.at("inode").get<uint64_t>()!=uint64_t(st.st_ino) ||
           saved.at("mtime_ns").get<uint64_t>()!=mtime || saved.at("ctime_ns").get<uint64_t>()!=ctime)
            throw std::runtime_error("prepared file changed since verification: "+name);
        auto file=std::make_shared<File>(dir/name);bytes=checked_add(bytes,file->size());files.emplace(name,file);
    }
    if(files.size()!=Layers+128 || manifest_.at("prepared_bytes").get<uint64_t>()!=bytes ||
       manifest_.at("experts").size()!=Layers || manifest_.at("ngrams").size()!=128 ||
       manifest_.at("expert_layout").size()!=9) throw std::runtime_error("incomplete prepared layout");
    const std::array<std::string,9> names={"gate_proj.weight","gate_proj.scales","gate_proj.biases",
        "up_proj.weight","up_proj.scales","up_proj.biases","down_proj.weight","down_proj.scales","down_proj.biases"};
    uint64_t offset=0;
    for(size_t p=0;p<names.size();++p) {
        const auto& layout=manifest_.at("expert_layout").at(p);
        const auto& ref=source.at("model.layers.0.mlp.switch_mlp."+names[p]);
        const std::vector<uint64_t> shape(ref.shape.begin()+1,ref.shape.end());
        if(layout.at("name")!=names[p] || layout.at("offset").get<uint64_t>()!=offset ||
           layout.at("length").get<uint64_t>()!=ref.row_bytes() || layout.at("shape").get<std::vector<uint64_t>>()!=shape ||
           layout.at("dtype")!=ref.dtype || layout.at("bits")!=4 || layout.at("group_size")!=64)
            throw std::runtime_error("unsupported prepared expert projection");
        offset+=ref.row_bytes();
    }
    std::unordered_map<std::string,bool> used;
    for(size_t l=0;l<Layers;++l) {
        const auto& e=manifest_.at("experts").at(l);
        if(e.at("layer")!=l || e.at("count")!=Experts || e.at("offset")!=0 || e.at("length")!=ExpertBytes ||
           e.at("stride")!=ExpertStride || e.at("alignment")!=16384)
            throw std::runtime_error("unsupported prepared expert range");
        const auto name=e.at("file").get<std::string>();
        if(!used.emplace(name,true).second) throw std::runtime_error("duplicate prepared file range");
        experts_[l]=files.at(name);
        if(experts_[l]->size()!=ExpertStride*Experts) throw std::runtime_error("invalid prepared expert length");
    }
    const auto fields=Json::parse(R"([{"offset":0,"length":80,"dtype":"U32","shape":[20]},{"offset":80,"length":10,"dtype":"BF16","shape":[5]},{"offset":90,"length":10,"dtype":"BF16","shape":[5]}])");
    for(size_t n=0;n<128;++n) {
        const auto& e=manifest_.at("ngrams").at(n);
        const auto rows=source.at("model.layers.1.ple.ple_embedding.ngram_embedding.shard_"+std::to_string(n)+".weight").shape.at(0);
        if(e.at("shard")!=n || e.at("count")!=rows || e.at("offset")!=0 || e.at("stride")!=100 || e.at("fields")!=fields)
            throw std::runtime_error("unsupported prepared ngram range");
        const auto name=e.at("file").get<std::string>();
        if(!used.emplace(name,true).second) throw std::runtime_error("duplicate prepared file range");
        ngrams_[n]=files.at(name);
        if(ngrams_[n]->size()!=checked_mul(rows,100)) throw std::runtime_error("invalid prepared ngram length");
    }
}
void PreparedArtifact::expert(ExpertKey key,const Buf& into) const {
    if(key.layer>=Layers || key.expert>=Experts || into->bytes<ExpertBytes) throw std::out_of_range("prepared expert range");
    experts_[key.layer]->read(key.expert*ExpertStride,{into->data,ExpertBytes});
}
void PreparedArtifact::ngram(uint64_t shard,uint64_t row,std::span<std::byte> into) const {
    if(shard>=128 || into.size()!=100) throw std::out_of_range("prepared ngram range");
    ngrams_[shard]->read(checked_mul(row,100),into);
}
void PreparedArtifact::ngram_range(uint64_t shard,uint64_t offset,std::span<std::byte> into) const {
    if(shard>=128 || into.empty() || into.size()>4196) throw std::out_of_range("prepared ngram page range");
    ngrams_[shard]->read(offset,into);
}
uint64_t PreparedArtifact::bytes_read() const {
    uint64_t n=0;for(const auto& f:experts_) n+=f->read_bytes.load();
    for(const auto& f:ngrams_) n+=f->read_bytes.load();return n;
}
Json PreparedArtifact::inspect() const {
    return {{"format",manifest_["format"]},{"recipe",manifest_["recipe"]},{"manifest_sha256",identity_},
        {"source_revision",ModelRevision},{"consumer_revision",artifact_revision(consumer_)},
        {"payload_equivalence_sha256",consumer_==Artifact::Mixed?Json::parse(MixedReuseLock).at("evidence_sha256"):Json(nullptr)},
        {"prepared_bytes",manifest_["prepared_bytes"]},
        {"expert_record_bytes",ExpertBytes},{"expert_stride",ExpertStride},{"ngram_record_bytes",100},
        {"ngram_cache_dtype","BF16"},{"application_read_bytes",bytes_read()}};
}

uint64_t monotonic_ns() {
    // Metal GPUStartTime/GPUEndTime and this host timer use Mach uptime.
    static const auto scale=[] {mach_timebase_info_data_t t{};mach_timebase_info(&t);return double(t.numer)/t.denom;}();
    return uint64_t(double(mach_absolute_time())*scale);
}
uint64_t CompletionEvents::ticket() const { std::lock_guard lock(mutex_); return sequence_; }
void CompletionEvents::publish() {
    { std::lock_guard lock(mutex_); ++sequence_; } changed_.notify_all();
}
void CompletionEvents::wait(uint64_t ticket, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_); changed_.wait_for(lock,timeout,[&]{return sequence_!=ticket;});
}
ReadPool::ReadPool(size_t workers, size_t queue_limit) : limit_(queue_limit) {
    if (!workers || workers > 64 || limit_<2) throw std::invalid_argument("invalid read-pool limits");
    try { for (size_t i=0;i<workers;++i) workers_.emplace_back([this]{worker();}); }
    catch (...) {
        { std::lock_guard lock(mutex_); stopping_ = true; }
        ready_.notify_all();
        for (auto& t:workers_) t.join();
        throw;
    }
}
ReadPool::~ReadPool() {
    { std::lock_guard lock(mutex_); stopping_ = true; }
    ready_.notify_all(); space_.notify_all();
    for (auto& t:workers_) t.join();
}
std::shared_future<void> ReadPool::submit(std::function<void()> work, ReadPriority priority,
                                       std::shared_ptr<ReadTiming> timing) {
    if(timing) timing->queued_ns=monotonic_ns();
    // Complete timing before making the future ready, including failed reads.
    std::packaged_task<void()> task([work=std::move(work),timing]{
        try { work(); } catch(...) { if(timing) timing->completed_ns=monotonic_ns(); throw; }
        if(timing) timing->completed_ns=monotonic_ns();
    });
    auto result = task.get_future().share();
    std::unique_lock lock(mutex_);
    space_.wait(lock,[&]{return stopping_ || (demand_.size()+future_.size()<limit_ &&
        (priority==ReadPriority::Demand || future_.size()<limit_/2));});
    if (stopping_) throw std::runtime_error("read pool stopped");
    (priority==ReadPriority::Demand?demand_:future_).push_back({std::move(task),std::move(timing)});
    ready_.notify_one();
    return result;
}
void ReadPool::worker() {
    for (;;) {
        Task task;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock,[this]{return stopping_ || !demand_.empty() || !future_.empty();});
            if (demand_.empty() && future_.empty()) return;
            auto& queue=demand_.empty()?future_:demand_;
            task=std::move(queue.front());queue.pop_front();++active_;space_.notify_all();
        }
        if(task.timing) task.timing->started_ns=monotonic_ns();
        task.work(); // packaged_task stores exceptions and then publishes readiness.
        events_->publish();
        { std::lock_guard lock(mutex_); --active_;
          if (!active_ && demand_.empty() && future_.empty()) drained_.notify_all(); }
    }
}
void ReadPool::drain() {
    std::unique_lock lock(mutex_);
    drained_.wait(lock,[this]{return demand_.empty() && future_.empty() && !active_;});
}
ExpertStore::ExpertStore(const Checkpoint& cp,std::shared_ptr<PreparedArtifact> prepared) : prepared_(std::move(prepared)) {
    const std::array<std::string,9> names = {"gate_proj.weight","gate_proj.scales","gate_proj.biases",
        "up_proj.weight","up_proj.scales","up_proj.biases","down_proj.weight","down_proj.scales","down_proj.biases"};
    for (int l=0;l<Layers;++l) for (int p=0;p<9;++p) {
        auto r = cp.at("model.layers." + std::to_string(l) + ".mlp.switch_mlp." + names[p]);
        const uint64_t rows = p<6 ? Intermediate : Hidden, cols = p<6 ? Hidden : Intermediate;
        const std::vector<uint64_t> shape = {Experts, rows, cols / (p%3 == 0 ? 8 : 64)};
        if (r.shape != shape || r.dtype != (p%3 == 0 ? "U32" : "BF16"))
            throw std::runtime_error("incompatible expert layout");
        refs_[l][p] = r;
        if (!l) offsets_[p+1] = offsets_[p] + r.row_bytes();
    }
    if (offsets_[9] != ExpertBytes) throw std::runtime_error("expert record size differs from pin");
}
void ExpertStore::read(ExpertKey key, const Buf& into) const {
    if (key.layer >= Layers || key.expert >= Experts || into->bytes < ExpertBytes)
        throw std::out_of_range("invalid expert load");
    if(prepared_) {prepared_->expert(key,into);return;}
    for (int p=0;p<9;++p) {
        const auto& r = refs_[key.layer][p];
        const auto bytes = r.row_bytes();
        r.file->read(r.offset + key.expert * bytes, {into->data + offsets_[p], size_t(bytes)});
    }
}
Json CacheStats::json() const {
    return {{"hits",hits},{"misses",misses},{"evictions",evictions},{"application_read_bytes",bytes},
        {"ready_hits",ready_hits},{"loading_joins",loading_joins},
        {"hit_rate",hits+misses ? double(hits)/double(hits+misses) : 0.0},
        {"layer_hits",layer_hits},{"layer_misses",layer_misses}};
}
struct ExpertCache::Entry {
    ExpertKey key{};
    Buf buffer;
    std::shared_future<void> future;
    std::shared_ptr<ReadTiming> timing=std::make_shared<ReadTiming>();
    unsigned pins = 0;
    bool referenced = true;
    unsigned queue = 0; // 0 unlisted, 1 probation, 2 protected.
    Entry *previous = nullptr, *next = nullptr;
};
ExpertCachePolicy parse_cache_policy(std::string_view name) {
    if(name=="clock") return ExpertCachePolicy::Clock;
    if(name=="slru") return ExpertCachePolicy::SegmentedLRU;
    throw std::invalid_argument("cache policy must be clock or slru");
}
std::string_view cache_policy_name(ExpertCachePolicy policy) {
    if(policy==ExpertCachePolicy::Clock) return "clock";
    if(policy==ExpertCachePolicy::SegmentedLRU) return "slru";
    throw std::invalid_argument("unknown expert cache policy");
}
void ExpertCache::unlink(Entry* e) {
    if(!e->queue) return;
    const auto q=e->queue-1;
    if(e->previous) e->previous->next=e->next; else oldest_[q]=e->next;
    if(e->next) e->next->previous=e->previous; else newest_[q]=e->previous;
    if(e->queue==2) --protected_;
    e->queue=0;e->previous=e->next=nullptr;
}
void ExpertCache::link(Entry* e,unsigned queue) {
    const auto q=queue-1;
    e->queue=queue;e->previous=newest_[q];e->next=nullptr;
    if(newest_[q]) newest_[q]->next=e; else oldest_[q]=e;
    newest_[q]=e;if(queue==2) ++protected_;
}
void ExpertCache::demote() {
    while(protected_>3*capacity()/4) {auto* e=oldest_[1];unlink(e);link(e,1);}
}
void ExpertCache::touch(Entry* e) {
    if(policy_==ExpertCachePolicy::Clock) {e->referenced=true;return;}
    unlink(e);link(e,2);demote();
}
ExpertCache::Entry* ExpertCache::slru_victim() const {
    // Protected is a preference, never a pin: avoid deadlock if probation is busy.
    for(auto* first:oldest_) for(auto* e=first;e;e=e->next)
        if(!e->pins && e->future.wait_for(std::chrono::seconds(0))==std::future_status::ready) return e;
    return nullptr;
}
Json ExpertCache::json() const {
    auto result=stats_.json();result["policy"]=cache_policy_name(policy_);
    result["protected_entries"]=protected_;
    result["probation_entries"]=policy_==ExpertCachePolicy::SegmentedLRU?occupancy()-protected_:0;
    result["entry_metadata_bytes"]=occupancy()*sizeof(Entry);
    return result;
}
ExpertCache::Lease::Lease(std::shared_ptr<Entry> e,int acquisition) : entry_(std::move(e)), acquisition_(acquisition) { ++entry_->pins; }
ExpertCache::Lease::~Lease() { if (entry_) --entry_->pins; }
ExpertCache::Lease::Lease(Lease&& other) noexcept : entry_(std::move(other.entry_)), acquisition_(other.acquisition_) {}
ExpertCache::Lease& ExpertCache::Lease::operator=(Lease&& other) noexcept {
    if (this != &other) { if (entry_) --entry_->pins; entry_=std::move(other.entry_); acquisition_=other.acquisition_; } return *this;
}
bool ExpertCache::Lease::ready() const {
    return entry_ && entry_->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}
const Buf& ExpertCache::Lease::wait() const {
    if (!entry_) throw std::logic_error("empty expert lease");
    entry_->future.get(); return entry_->buffer;
}
ExpertKey ExpertCache::Lease::key() const {
    if (!entry_) throw std::logic_error("empty expert lease"); return entry_->key;
}
const ReadTiming& ExpertCache::Lease::timing() const {
    if(!ready()) throw std::logic_error("read timing requires completion");
    entry_->future.get();return *entry_->timing;
}
const char* ExpertCache::Lease::acquisition() const {
    return acquisition_==0?"new_miss":acquisition_==1?"ready_hit":"loading_join";
}
ExpertCache::ExpertCache(size_t slots, Allocator allocator, ReadPool& reads,
                        std::function<void(ExpertKey,const Buf&)> loader, uint64_t stride, ExpertCachePolicy policy)
    : slots_(slots), allocator_(std::move(allocator)), reads_(reads), loader_(std::move(loader)), stride_(stride), policy_(policy) {
    if (!slots || !stride) throw std::invalid_argument("empty expert cache");
    (void)cache_policy_name(policy_);
}
ExpertCache::~ExpertCache() { reads_.drain(); }
ExpertCache::Lease ExpertCache::acquire(ExpertKey key) {
    if (key.layer >= Layers || key.expert >= Experts) throw std::out_of_range("invalid expert id");
    if (auto it=lookup_.find(key.value()); it!=lookup_.end()) {
        auto e=slots_[it->second]; touch(e.get()); ++stats_.hits; ++stats_.layer_hits[key.layer];
        const bool ready=e->future.wait_for(std::chrono::seconds(0))==std::future_status::ready;
        if(ready) ++stats_.ready_hits; else ++stats_.loading_joins;
        return Lease(e,ready?1:2);
    }
    size_t selected=slots_.size();
    if(policy_==ExpertCachePolicy::SegmentedLRU && occupancy()==capacity()) {
        if(auto* victim=slru_victim()) selected=lookup_.at(victim->key.value());
    } else for (size_t trial=0;trial<slots_.size()*2+1;++trial) {
        const size_t slot=hand_;hand_=(hand_+1)%slots_.size();
        auto& old=slots_[slot];
        if (old) {
            if(policy_==ExpertCachePolicy::SegmentedLRU) continue; // Fill free capacity first.
            if (old->pins) continue;
            if (old->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) continue;
            if (old->referenced) { old->referenced=false; continue; }
        }
        selected=slot;break;
    }
    if(selected<slots_.size()) {
        auto& old=slots_[selected];
        auto e=std::make_shared<Entry>(); e->key=key;
        e->buffer=old ? old->buffer : allocator_(stride_);
        const auto loader=loader_; const auto buffer=e->buffer;
        e->future=reads_.submit([loader,key,buffer]{loader(key,buffer);},ReadPriority::Demand,e->timing);
        if (old) {unlink(old.get());lookup_.erase(old->key.value());++stats_.evictions;}
        old=e;lookup_[key.value()]=selected;
        if(policy_==ExpertCachePolicy::SegmentedLRU) link(e.get(),1);
        ++stats_.misses; ++stats_.layer_misses[key.layer]; stats_.bytes+=ExpertBytes;
        return Lease(e,0);
    }
    throw std::runtime_error("expert cache exhausted by outstanding leases; finish a batch before acquiring more");
}
void ExpertCache::clear() {
    for (const auto& e:slots_) if (e && e->pins) throw std::logic_error("cannot clear leased expert slots");
    reads_.drain(); lookup_.clear(); for (auto& e:slots_) e.reset(); hand_=0;
    oldest_={};newest_={};protected_=0;
}
bool ExpertCache::ready(ExpertKey key) const {
    const auto it=lookup_.find(key.value());
    if(it==lookup_.end()) return false;
    const auto& e=slots_[it->second];
    return e->future.valid() && e->future.wait_for(std::chrono::seconds(0))==std::future_status::ready;
}
void ExpertCache::resize(size_t slots) {
    if (!slots) throw std::invalid_argument("empty expert cache");
    // Resizing is a coordinator boundary, never invalidate a lease or read.
    for(const auto& e:slots_) if(e && e->pins) throw std::logic_error("cannot resize leased expert slots");
    reads_.drain();
    if(slots>=slots_.size()) {slots_.resize(slots);return;}
    size_t occupied=lookup_.size();
    while(occupied>slots) {
        size_t victim=hand_;hand_=(hand_+1)%slots_.size();
        if(policy_==ExpertCachePolicy::SegmentedLRU) victim=lookup_.at(slru_victim()->key.value());
        auto& e=slots_[victim];
        if(!e) continue;
        if(policy_==ExpertCachePolicy::Clock && e->referenced) {e->referenced=false;continue;}
        unlink(e.get());lookup_.erase(e->key.value());e.reset();--occupied;++stats_.evictions;
    }
    std::vector<std::shared_ptr<Entry>> survivors;survivors.reserve(slots);
    lookup_.clear();
    for(auto& e:slots_) if(e) {lookup_[e->key.value()]=survivors.size();survivors.push_back(std::move(e));}
    survivors.resize(slots);slots_=std::move(survivors);hand_=0;
    if(policy_==ExpertCachePolicy::SegmentedLRU) demote();
}

MemoryPlan MemoryPlan::make(uint64_t requested,uint64_t physical,uint64_t metal_limit,
                            uint64_t resident,int context,int chunk,int panel,int layers,uint64_t kernel_scratch,bool double_pipeline,uint64_t decode_scratch,bool snapshot) {
    if (!requested || requested>22*GiB || context<1 || context>8192 || chunk<1 || chunk>256)
        throw std::invalid_argument("limits: memory <=22GiB, context 1..8192, chunk 1..256");
    if(panel!=0 && panel!=256 && panel!=512 && panel!=1024)
        throw std::invalid_argument("panel must be 0, 256, 512, or 1024");
    if(layers<1 || layers>Layers) throw std::invalid_argument("layers must be 1..48");
    if (physical<=8*GiB || !metal_limit) throw std::runtime_error("insufficient usable unified memory");
    MemoryPlan p;
    p.limit=std::min({requested,physical-8*GiB,metal_limit}); p.resident=resident;
    // FP32 recurrent state and activations; KV is FP32 in the correctness path.
    const auto aligned=[](uint64_t n){return (n+16383)/16384*16384;};
    const auto attention_layers=uint64_t(layers/4),recurrent_layers=uint64_t(layers)-attention_layers;
    p.state=recurrent_layers*(aligned(48ull*128*128*4)+aligned(3ull*10240*4))+
        attention_layers*(2*aligned(uint64_t(context)*512*4)+aligned(uint64_t(context)*128*4))+
        (layers>1?aligned(9ull*Hyper*4):0);
    p.scratch=chunk<=32 ? 128*MiB :
        std::max<uint64_t>(512*MiB, uint64_t(chunk)*Hidden*TopK*4 + uint64_t(chunk)*Hyper*4*32 + 256*MiB);
    p.kernel_scratch=kernel_scratch;p.scratch+=kernel_scratch;
    p.pipeline_scratch=(double_pipeline?2*p.scratch:0)+decode_scratch;
    p.snapshot=snapshot?p.state:0;
    const auto base=p.resident+p.state+p.scratch+p.ngram+p.reserve+p.pipeline_scratch+p.snapshot+p.runtime_control;
    // Bound complete-panel activations, expert contributions, router partials,
    // and shared work in addition to two bounded attention/recurrent groups.
    // Each successful smaller admission keeps the same precision and arithmetic.
    for(int candidate=panel;candidate>=256;candidate/=2) {
        const auto tokens=std::min(candidate,context);
        if(tokens<=chunk) break;
        const auto bytes=(uint64_t(tokens)*(3*Hyper+Hidden*(TopK+4)+Experts*17+16)*4+MiB-1)/MiB*MiB;
        if(base+bytes+32*ExpertStride<=p.limit) {p.panel_scratch=bytes;p.panel_tokens=uint32_t(tokens);break;}
    }
    const auto fixed=base+p.panel_scratch;
    if (fixed>=p.limit || p.limit-fixed<32*ExpertStride)
        throw std::runtime_error("memory admission requires at least "+std::to_string(fixed+32*ExpertStride)+
            " bytes including 32 expert slots; available engine budget is "+std::to_string(p.limit)+" bytes");
    p.slots=std::min<uint64_t>((p.limit-fixed)/ExpertStride,Layers*Experts);
    p.experts=p.slots*ExpertStride;
    return p;
}
void MemoryPlan::cap_experts(size_t count) {
    if(!count) return;
    if(count<32 || count>slots) throw std::invalid_argument("expert slots exceed admitted capacity or are below 32");
    slots=count;experts=slots*ExpertStride;
}
MemoryPlan MemoryPlan::without_prompt_workspaces(size_t expert_cap) const {
    if(pipeline_scratch<2*scratch) throw std::logic_error("memory plan has no double workspace reservation");
    auto p=*this;p.pipeline_scratch-=2*scratch;
    const auto fixed=p.resident+p.state+p.scratch+p.panel_scratch+p.ngram+p.reserve+p.pipeline_scratch+p.snapshot+p.runtime_control;
    p.slots=std::min<uint64_t>((p.limit-fixed)/ExpertStride,Layers*Experts);
    p.experts=p.slots*ExpertStride;p.cap_experts(expert_cap);return p;
}
Json MemoryPlan::json() const {
    return {{"limit_bytes",limit},{"resident_bytes",resident},{"session_bytes",state},{"scratch_bytes",scratch},{"kernel_scratch_bytes",kernel_scratch},
        {"panel_tokens",panel_tokens},{"panel_scratch_bytes",panel_scratch},
        {"runtime_control_bytes",runtime_control},{"pipeline_scratch_bytes",pipeline_scratch},{"snapshot_bytes",snapshot},{"ngram_bytes",ngram},{"reserve_bytes",reserve},{"expert_bytes",experts},{"expert_slots",slots},
        {"planned_bytes",resident+state+scratch+panel_scratch+ngram+reserve+experts+pipeline_scratch+snapshot+runtime_control}};
}

NgramStore::NgramStore(const Checkpoint& cp,ReadPool& reads,uint64_t cache_bytes,std::shared_ptr<PreparedArtifact> prepared)
    : reads_(reads),prepared_(std::move(prepared)) {
    const std::string base="model.layers.1.ple.ple_embedding.";
    auto loadints=[&](const std::string& name,auto& out) {
        const auto& r=cp.at(base+name);
        if (r.dtype!="I64" || r.bytes!=sizeof(out)) throw std::runtime_error("invalid ngram hash buffers");
        r.file->read(r.offset,{reinterpret_cast<std::byte*>(out.data()),sizeof(out)});
    };
    loadints("layer_multipliers",multipliers_); loadints("ngram_heads_vocab_sizes",sizes_); loadints("ngram_heads_offsets",offsets_);
    for (int i=0;i<16;++i) if (sizes_[i]<=0 || offsets_[i]<0) throw std::runtime_error("invalid ngram hash table");
    for (int s=0;s<128;++s) for(int p=0;p<3;++p) {
        refs_[s][p]=cp.at(base+"ngram_embedding.shard_"+std::to_string(s)+
                         std::array<std::string,3>{".weight",".scales",".biases"}[p]);
        auto& r=refs_[s][p];
        if (r.shape.size()!=2 || r.shape[1]!=(p?5:20) || r.dtype!=(p?"BF16":"U32"))
            throw std::runtime_error("unsupported ngram row shape");
    }
    // Reserve map overhead as well as row payload inside the advertised cap.
    const auto count=cache_bytes/(sizeof(Row)+96);
    if (!count) throw std::invalid_argument("ngram cache too small");
    rows_.resize(count); lookup_.reserve(count);
}
std::vector<std::array<int64_t,16>> NgramStore::row_ids(std::span<const int> tokens,std::array<int,2> history) const {
    std::vector<std::array<int64_t,16>> result(tokens.size());
    for(size_t t=0;t<tokens.size();++t) {
        if(tokens[t]<0 || tokens[t]>=Vocab) throw std::out_of_range("token outside vocabulary");
        const int prev=history[1], prev2=prev==EndOfText ? EndOfText : history[0];
        const std::array<int,3> window={tokens[t],prev,prev2};
        uint64_t mix=uint64_t(window[0])*uint64_t(multipliers_[0]);
        for(int n=1;n<3;++n) {
            mix^=uint64_t(window[n])*uint64_t(multipliers_[n]);
            const auto signed_mix=std::bit_cast<int64_t>(mix);
            for(int h=(n-1)*8;h<n*8;++h) {
                int64_t row=signed_mix%sizes_[h]; if(row<0) row+=sizes_[h];
                result[t][h]=row+offsets_[h];
            }
        }
        history={history[1],tokens[t]};
    }
    return result;
}
void NgramStore::decode_row(std::span<const std::byte> packed,std::span<float> into) {
    if(packed.size()!=100 || into.size()!=160) throw std::invalid_argument("ngram decode geometry");
    std::array<uint32_t,20> w{};std::array<uint16_t,5> s{},b{};
    std::memcpy(w.data(),packed.data(),80);std::memcpy(s.data(),packed.data()+80,10);std::memcpy(b.data(),packed.data()+90,10);
    for(int d=0;d<160;++d) into[d]=round_bf16(bf16(s[d/32])*float((w[d/8]>>(4*(d%8)))&15)+bf16(b[d/32]));
}
void NgramStore::read_row(int64_t id,std::span<float> into) const {
    const uint64_t rows=refs_[0][0].shape[0];
    if(id<0 || uint64_t(id)>=rows*128) throw std::out_of_range("ngram row outside table");
    const auto shard=uint64_t(id)/rows,row=uint64_t(id)%rows;
    std::array<std::byte,100> packed{};
    if(prepared_) prepared_->ngram(shard,row,packed);
    else {
        refs_[shard][0].file->read(refs_[shard][0].offset+row*80,{packed.data(),80});
        refs_[shard][1].file->read(refs_[shard][1].offset+row*10,{packed.data()+80,10});
        refs_[shard][2].file->read(refs_[shard][2].offset+row*10,{packed.data()+90,10});
    }
    decode_row(packed,into);
}
void NgramStore::embedding(std::span<const int> tokens,std::array<int,2> history,std::span<float> out) {
    if(out.size()!=tokens.size()*Hidden) throw std::invalid_argument("ngram output shape");
    const auto ids=row_ids(tokens,history);
    // Coalesce repeated rows inside this chunk before submitting any reads.
    // Jobs write distinct first occurrences; fan-out follows completion.
    std::unordered_map<int64_t,size_t> first;
    std::vector<std::pair<size_t,size_t>> copies;
    std::vector<std::shared_future<void>> pending;
    struct Request { uint64_t offset;size_t destination; };
    std::unordered_map<uint64_t,std::vector<Request>> pages;
    std::exception_ptr failure;
    try {
        for(size_t t=0;t<ids.size();++t) for(int h=0;h<16;++h) {
            const auto id=ids[t][h];const size_t pos=t*Hidden+h*160;
            const auto dest=out.subspan(pos,160);
            if(auto it=lookup_.find(id);it!=lookup_.end()) {
                ++hits;std::transform(rows_[it->second].values.begin(),rows_[it->second].values.end(),dest.begin(),bf16);
            } else if(auto existing=first.find(id);existing!=first.end()) {
                ++hits;copies.emplace_back(existing->second,pos);
            } else {
                ++misses;first.emplace(id,pos);
                if(prepared_) {
                    const auto count=refs_[0][0].shape[0];
                    if(id<0 || uint64_t(id)>=count*128) throw std::out_of_range("ngram row outside table");
                    const auto shard=uint64_t(id)/count,offset=(uint64_t(id)%count)*100;
                    pages[(shard<<48)|(offset/4096)].push_back({offset,pos});
                } else pending.push_back(reads_.submit([this,id,dest]{read_row(id,dest);},ReadPriority::Future));
            }
        }
        // Multiple requested rows whose starts share a 4KiB page issue one
        // bounded range read. Singleton pages still read exactly 100 bytes.
        // A row crossing the page end is included in that same range.
        for(auto& [page,requests]:pages) {
            std::sort(requests.begin(),requests.end(),[](const auto& a,const auto& b){return a.offset<b.offset;});
            pending.push_back(reads_.submit([this,shard=page>>48,requests=std::move(requests),out]{
                const auto begin=requests.front().offset,length=requests.back().offset+100-begin;
                std::array<std::byte,4196> packed{};
                prepared_->ngram_range(shard,begin,{packed.data(),size_t(length)});
                for(const auto& r:requests) decode_row({packed.data()+r.offset-begin,100},out.subspan(r.destination,160));
            },ReadPriority::Future));
        }
    } catch(...) {failure=std::current_exception();}
    for(auto& f:pending) { try { f.get(); } catch(...) { if(!failure) failure=std::current_exception(); } }
    if(failure) std::rethrow_exception(failure);
    for(auto [from,to]:copies) std::copy_n(out.begin()+from,160,out.begin()+to);
    for(size_t t=0;t<ids.size();++t) for(int h=0;h<16;++h) {
        const auto id=ids[t][h]; if(lookup_.contains(id)) continue;
        auto& row=rows_[next_]; if(row.key>=0) lookup_.erase(row.key);
        row.key=id;
        for(size_t d=0;d<160;++d) row.values[d]=uint16_t(std::bit_cast<uint32_t>(out[t*Hidden+h*160+d])>>16);
        lookup_[id]=next_; next_=(next_+1)%rows_.size();
    }
}
} // namespace zerocool::engine
