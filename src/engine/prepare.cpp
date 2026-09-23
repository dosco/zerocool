#include "checkpoint_io.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <map>
#include <print>
#include <sys/stat.h>
#include <unistd.h>

namespace zerocool::engine {
namespace {

// The record geometry the engine reads. ExpertBytes is the nine projection
// pieces back to back; Stride pads that to an alignment boundary so one expert
// is one aligned read.
constexpr uint64_t Alignment = 16384;
constexpr uint64_t NgramRow = 100, NgramChunk = 32768;
constexpr uint64_t WriteHeadroom = 2 * GiB;
constexpr const char* NgramBase = "model.layers.1.ple.ple_embedding.ngram_embedding.shard_";

void check(bool ok, const std::string& message) {
    if(!ok) throw std::runtime_error(message);
}

struct Fd {
    int value = -1;
    Fd() = default;
    explicit Fd(int v) : value(v) {}
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    Fd(Fd&& other) noexcept : value(other.value) { other.value = -1; }
    Fd& operator=(Fd&& other) noexcept {
        if(this != &other) {
            if(value >= 0) ::close(value);
            value = other.value;
            other.value = -1;
        }
        return *this;
    }
    ~Fd() {
        if(value >= 0) ::close(value);
    }
};

struct Ref {
    std::string file, dtype;
    uint64_t base = 0, begin = 0, end = 0;
    std::vector<uint64_t> shape;
};

// A file being published: written to a .partial, hashed as it goes, and renamed
// only once its whole expected length is on disk.
class Publisher {
public:
    Publisher(const std::filesystem::path& path, uint64_t expected)
        : path_(path), temporary_(std::filesystem::path(path).concat(".partial")), expected_(expected) {
        fd_ = Fd(::open(temporary_.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644));
        check(fd_.value >= 0, "cannot open: " + temporary_.string());
        ::fcntl(fd_.value, F_NOCACHE, 1);
        CC_SHA256_Init(&context_);
    }
    void write(const void* data, uint64_t bytes) {
        CC_SHA256_Update(&context_, data, CC_LONG(bytes));
        const auto* at = static_cast<const char*>(data);
        for(uint64_t left = bytes; left;) {
            const auto n = ::write(fd_.value, at, size_t(left));
            check(n > 0, "short prepared write: " + temporary_.string());
            at += n;
            left -= uint64_t(n);
            written_ += uint64_t(n);
        }
    }
    std::string finish() {
        check(written_ == expected_, "unexpected prepared size: " + path_.filename().string());
        check(::fsync(fd_.value) == 0, "cannot flush: " + temporary_.string());
        std::filesystem::rename(temporary_, path_);
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        CC_SHA256_Final(digest, &context_);
        static constexpr char hex[] = "0123456789abcdef";
        std::string out;
        for(auto c : digest) {
            out += hex[c >> 4];
            out += hex[c & 15];
        }
        return out;
    }

private:
    std::filesystem::path path_, temporary_;
    uint64_t expected_ = 0, written_ = 0;
    Fd fd_;
    CC_SHA256_CTX context_{};
};

} // namespace

Json checkpoint_io::prepare(const std::filesystem::path& model, const std::filesystem::path& output,
                            const Json& source_lock, const Json& canonical_lock, const Geometry& geometry,
                            bool verify_only, const std::atomic<bool>& cancel, const ProgressFn& progress) {
    const auto pinned_manifest = canonical_lock.at("prepared_control").at("manifest_sha256").get<std::string>();
    check(geometry.layers && geometry.experts && geometry.ngram_shards && geometry.record_bytes &&
              geometry.record_bytes <= geometry.record_stride && geometry.record_stride % Alignment == 0,
          "invalid prepared geometry");
    const auto source = checkpoint_io::verify(model, source_lock, true, cancel);
    // The output identity is canonical Q4, including when its byte-identical
    // payload is read from the verified mixed checkpoint. Input provenance is
    // recorded separately; never write a Q4 verification receipt for mixed input.
    Json input_files = Json::object();
    for(auto it = source.at("files").begin(); it != source.at("files").end(); ++it)
        input_files[it.key()] = it.value().at("sha256");
    const Json provenance{{"input_revision", source.at("revision")}, {"input_files", input_files}};
    std::filesystem::create_directories(output);

    const Json identity{
        {"schema", 1}, {"source_revision", canonical_lock.at("revision")}, {"format", "zc-affine-records-v1"}};
    const auto identity_path = output / "preparation.json";
    if(std::filesystem::exists(identity_path))
        check(read_json(identity_path) == identity, "output belongs to a different preparation");
    write_json_atomic(identity_path, identity);

    const auto receipt_path = output / "verification.json";
    Json old = std::filesystem::exists(receipt_path) ? read_json(receipt_path) : Json::object();
    if(!old.contains("files")) old["files"] = Json::object();

    // Map every tensor to the shard and byte range holding it.
    std::map<std::string, Ref> refs;
    std::map<std::string, Fd> shards;
    const auto weight_map = read_json(model / "model.safetensors.index.json").at("weight_map");
    std::vector<std::string> names;
    for(const auto& value : weight_map) names.push_back(value.get<std::string>());
    std::sort(names.begin(), names.end());
    names.erase(std::unique(names.begin(), names.end()), names.end());
    for(const auto& name : names) {
        check(std::filesystem::path(name).filename() == name && name != "." && name != "..", "unsafe source filename");
        Fd fd(::open((model / name).c_str(), O_RDONLY));
        check(fd.value >= 0, "cannot open shard: " + name);
        ::fcntl(fd.value, F_NOCACHE, 1);
        uint64_t header = 0;
        check(::pread(fd.value, &header, 8, 0) == 8, "cannot read shard header: " + name);
        check(header > 0 && header <= 16 * MiB, "invalid safetensors header: " + name);
        std::string text(header, '\0');
        check(uint64_t(::pread(fd.value, text.data(), size_t(header), 8)) == header, "short shard header: " + name);
        const auto meta = Json::parse(text);
        for(auto it = meta.begin(); it != meta.end(); ++it) {
            if(it.key() == "__metadata__") continue;
            auto key = it.key();
            if(key.starts_with("language_model.")) key.erase(0, 15);
            refs[key] = Ref{name,
                            it.value().at("dtype").get<std::string>(),
                            8 + header,
                            it.value().at("data_offsets").at(0).get<uint64_t>(),
                            it.value().at("data_offsets").at(1).get<uint64_t>(),
                            it.value().at("shape").get<std::vector<uint64_t>>()};
        }
        shards.emplace(name, std::move(fd));
    }
    const auto find = [&](const std::string& key) -> const Ref& {
        const auto it = refs.find(key);
        check(it != refs.end(), "missing source tensor: " + key);
        return it->second;
    };
    // Read `count` rows of a tensor. Row width comes from the declared shape,
    // so a checkpoint whose geometry moved is rejected rather than misread.
    const auto part = [&](const std::string& key, uint64_t row, uint64_t count, uint64_t expected_width,
                          std::byte* into) -> uint64_t {
        const auto& ref = find(key);
        check(!ref.shape.empty() && ref.shape[0] > 0 && ref.end >= ref.begin &&
                  (ref.end - ref.begin) % ref.shape[0] == 0,
              "unsupported source shape: " + key);
        const auto width = (ref.end - ref.begin) / ref.shape[0];
        check(width == expected_width, "unexpected source row width: " + key);
        check(row <= ref.shape[0] && count <= ref.shape[0] - row, "source row out of bounds: " + key);
        const auto bytes = count * width;
        check(uint64_t(::pread(shards.at(ref.file).value, into, size_t(bytes),
                               off_t(ref.base + ref.begin + row * width))) == bytes,
              "short source read: " + key);
        return width;
    };

    // Expert record layout, taken from layer zero and required to be exact.
    static constexpr std::array<const char*, 3> Projections{"gate_proj", "up_proj", "down_proj"};
    static constexpr std::array<const char*, 3> Pieces{"weight", "scales", "biases"};
    std::vector<std::string> pieces;
    Json layout = Json::array();
    uint64_t cursor = 0;
    for(const auto* projection : Projections)
        for(const auto* piece : Pieces) {
            const auto name = std::string(projection) + "." + piece;
            const auto& ref = find("model.layers.0.mlp.switch_mlp." + name);
            check(!ref.shape.empty() && ref.shape[0] == geometry.experts && ref.end >= ref.begin &&
                      (ref.end - ref.begin) % geometry.experts == 0,
                  "unsupported expert shape");
            const auto size = (ref.end - ref.begin) / geometry.experts;
            layout.push_back(Json{{"name", name},
                                  {"offset", cursor},
                                  {"length", size},
                                  {"shape", std::vector<uint64_t>(ref.shape.begin() + 1, ref.shape.end())},
                                  {"dtype", ref.dtype},
                                  {"bits", 4},
                                  {"group_size", 64}});
            pieces.push_back(name);
            cursor += size;
        }
    check(cursor == geometry.record_bytes, "unsupported expert format");

    uint64_t expected_bytes = geometry.layers * geometry.experts * geometry.record_stride;
    std::vector<uint64_t> shard_rows(geometry.ngram_shards);
    for(uint64_t shard = 0; shard < geometry.ngram_shards; ++shard) {
        shard_rows[shard] = find(NgramBase + std::to_string(shard) + ".weight").shape.at(0);
        expected_bytes += shard_rows[shard] * NgramRow;
    }
    if(!verify_only) {
        uint64_t existing = 0;
        for(const auto& item : std::filesystem::directory_iterator(output))
            if(item.path().extension() == ".bin") existing += uint64_t(item.file_size());
        const auto space = std::filesystem::space(output);
        check(space.available >= (expected_bytes > existing ? expected_bytes - existing : 0) + WriteHeadroom,
              "insufficient space for prepared records plus 2GiB headroom");
    }

    Json files = Json::array();
    // Interim receipts carry every entry known so far so an interrupted run
    // resumes; the final receipt carries only what this run published, which is
    // how a stale entry from another recipe stops being vouched for.
    Json produced = Json::object();
    uint64_t published = 0;
    // Publish one record file, reusing an unchanged completed one.
    const auto publish = [&](const std::string& name, uint64_t expected,
                             const std::function<void(Publisher&)>& produce) {
        check(!cancel.load(), "cancelled");
        const auto path = output / name;
        const Json saved = old.at("files").contains(name) ? old.at("files").at(name) : Json::object();
        Json entry;
        bool reused = false;
        if(std::filesystem::exists(path)) {
            const auto now = file_fingerprint(path);
            reused = true;
            for(auto it = now.begin(); it != now.end(); ++it)
                if(!saved.contains(it.key()) || saved.at(it.key()) != it.value()) reused = false;
            if(reused) {
                check(uint64_t(std::filesystem::file_size(path)) == expected, "completed file size mismatch: " + name);
                if(verify_only)
                    check(file_digest(path, cancel) == saved.value("sha256", std::string()),
                          "corrupt prepared file: " + name);
                entry = saved;
            }
        }
        if(!reused) {
            check(!verify_only, "missing or changed prepared file: " + name);
            Publisher publisher(path, expected);
            produce(publisher);
            const auto digest = publisher.finish();
            entry = file_fingerprint(path);
            entry["sha256"] = digest;
        }
        produced[name] = entry;
        old["files"][name] = entry;
        Json interim = identity;
        interim.update(provenance);
        interim["files"] = old.at("files");
        write_json_atomic(receipt_path, interim); // an interrupted run keeps what finished
        files.push_back(Json{{"path", name}, {"size", expected}, {"sha256", entry.at("sha256")}});
        published += expected;
        if(progress)
            progress(FetchProgress{name, published, expected_bytes, 0, reused, reused ? "reused" : "prepared"});
    };

    Json experts = Json::array();
    std::vector<std::byte> record(geometry.record_stride);
    for(uint64_t layer = 0; layer < geometry.layers; ++layer) {
        char name[32];
        std::snprintf(name, sizeof name, "experts-%02llu.bin", static_cast<unsigned long long>(layer));
        publish(name, geometry.record_stride * geometry.experts, [&](Publisher& out) {
            const auto prefix = "model.layers." + std::to_string(layer) + ".mlp.switch_mlp.";
            for(uint64_t expert = 0; expert < geometry.experts; ++expert) {
                check(!cancel.load(), "cancelled");
                std::memset(record.data(), 0, record.size()); // the pad after the nine pieces is zero
                uint64_t at = 0;
                for(size_t p = 0; p < pieces.size(); ++p)
                    at += part(prefix + pieces[p], expert, 1, layout.at(p).at("length").get<uint64_t>(),
                               record.data() + at);
                check(at == geometry.record_bytes, "unexpected expert record size");
                out.write(record.data(), geometry.record_stride);
            }
        });
        experts.push_back(Json{{"layer", layer},
                               {"file", name},
                               {"count", geometry.experts},
                               {"offset", 0},
                               {"length", geometry.record_bytes},
                               {"stride", geometry.record_stride},
                               {"alignment", Alignment}});
    }

    // Each ngram row interleaves three sources into one hundred contiguous
    // bytes, so a lookup is a single row-sized read.
    static constexpr std::array<std::pair<uint64_t, uint64_t>, 3> Fields{{{0, 80}, {80, 10}, {90, 10}}};
    Json ngrams = Json::array();
    std::vector<std::byte> rows(NgramChunk * NgramRow), field(NgramChunk * 80);
    for(uint64_t shard = 0; shard < geometry.ngram_shards; ++shard) {
        const auto count = shard_rows[shard];
        char name[32];
        std::snprintf(name, sizeof name, "ngram-%03llu.bin", static_cast<unsigned long long>(shard));
        publish(name, count * NgramRow, [&](Publisher& out) {
            const auto prefix = NgramBase + std::to_string(shard) + ".";
            for(uint64_t at = 0; at < count; at += NgramChunk) {
                check(!cancel.load(), "cancelled");
                const auto n = std::min(NgramChunk, count - at);
                std::memset(rows.data(), 0, n * NgramRow);
                for(size_t f = 0; f < Fields.size(); ++f) {
                    const auto [begin, width] = Fields[f];
                    check(part(prefix + Pieces[f], at, n, width, field.data()) == width,
                          "unexpected ngram field width");
                    for(uint64_t row = 0; row < n; ++row)
                        std::memcpy(rows.data() + row * NgramRow + begin, field.data() + row * width, size_t(width));
                }
                out.write(rows.data(), n * NgramRow);
            }
        });
        ngrams.push_back(Json{
            {"shard", shard},
            {"file", name},
            {"count", count},
            {"offset", 0},
            {"stride", NgramRow},
            {"fields",
             Json::array({Json{{"offset", 0}, {"length", 80}, {"dtype", "U32"}, {"shape", Json::array({20})}},
                          Json{{"offset", 80}, {"length", 10}, {"dtype", "BF16"}, {"shape", Json::array({5})}},
                          Json{{"offset", 90}, {"length", 10}, {"dtype", "BF16"}, {"shape", Json::array({5})}}})}});
    }

    for(auto it = source.at("files").begin(); it != source.at("files").end(); ++it) {
        const auto now = file_fingerprint(model / it.key());
        for(auto field_it = now.begin(); field_it != now.end(); ++field_it)
            check(it.value().at(field_it.key()) == field_it.value(), "source changed during preparation: " + it.key());
    }

    Json source_files = Json::object();
    for(const auto& entry : canonical_lock.at("files"))
        if(!entry.value("optional", false)) source_files[entry.at("path").get<std::string>()] = entry.at("sha256");
    Json manifest = identity;
    manifest["recipe"] = "q4-control-lossless";
    manifest["expert_layout"] = layout;
    manifest["experts"] = experts;
    manifest["ngrams"] = ngrams;
    manifest["files"] = files;
    manifest["prepared_bytes"] = expected_bytes;
    manifest["ngram_cache_dtype"] = "BF16";
    manifest["source_files"] = source_files;

    const auto serialized = manifest.dump(2) + "\n";
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(serialized.data(), CC_LONG(serialized.size()), digest);
    static constexpr char hex[] = "0123456789abcdef";
    std::string manifest_sha;
    for(auto c : digest) {
        manifest_sha += hex[c >> 4];
        manifest_sha += hex[c & 15];
    }
    check(manifest_sha == pinned_manifest, "prepared manifest differs from the pinned lossless control");
    write_json_atomic(output / "manifest.json", manifest);

    Json receipt = identity;
    receipt.update(provenance);
    receipt["files"] = produced;
    receipt["manifest_sha256"] = file_digest(output / "manifest.json", cancel);
    receipt["verified_at"] = uint64_t(std::time(nullptr));
    write_json_atomic(receipt_path, receipt);
    return manifest;
}

Json prepare_storage(const std::filesystem::path& model, const std::filesystem::path& output, bool verify_only,
                     const std::atomic<bool>& cancel, const ProgressFn& progress, Artifact artifact) {
    const auto canonical = artifact_lock(Artifact::Q4);
    verify_prepared_compatibility(artifact, canonical.at("prepared_control").at("manifest_sha256").get<std::string>());
    return checkpoint_io::prepare(model, output, artifact_lock(artifact), canonical, {}, verify_only, cancel, progress);
}

} // namespace zerocool::engine
