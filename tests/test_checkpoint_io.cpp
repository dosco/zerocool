#include "doctest.h"
#include "../src/engine/checkpoint_io.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <arpa/inet.h>
#include <fstream>
#include <map>
#include <mutex>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

using namespace zerocool::engine;
namespace {
struct Directory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() / ("zerocool-checkpoint-" + std::to_string(monotonic_ns()));
    Directory() { std::filesystem::create_directories(path); }
    ~Directory() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};
void put(const std::filesystem::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), std::streamsize(bytes.size()));
    if(!out) throw std::runtime_error("fixture write failed");
}
std::string digest(const std::string& bytes) {
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes.data(), CC_LONG(bytes.size()), hash);
    std::string out;
    for(auto c : hash) {
        out += "0123456789abcdef"[c >> 4];
        out += "0123456789abcdef"[c & 15];
    }
    return out;
}
Json file_entry(const std::string& name, const std::string& bytes) {
    return {{"path", name}, {"size", bytes.size()}, {"sha256", digest(bytes)}};
}
Json file_lock(const std::map<std::string, std::string>& files) {
    Json entries = Json::array();
    for(const auto& [name, bytes] : files) entries.push_back(file_entry(name, bytes));
    return {{"revision", "fixture-q4"}, {"files", entries}};
}

// This loopback server deliberately supports short responses, bad bytes and
// ignored ranges. The native downloader, not a Python imitation, consumes it.
class HttpFixture {
public:
    struct Response {
        std::string bytes;
        bool ignore_range = false, truncate = false;
    };
    std::string url;
    std::function<Response(const std::string&)> response;
    explicit HttpFixture(std::function<Response(const std::string&)> handler) : response(std::move(handler)) {
        listener_ = socket(AF_INET, SOCK_STREAM, 0);
        if(listener_ < 0) throw std::runtime_error("fixture socket failed");
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if(bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) || listen(listener_, 16)) {
            close(listener_);
            throw std::runtime_error("fixture listener unavailable");
        }
        socklen_t size = sizeof(address);
        getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &size);
        url = "http://127.0.0.1:" + std::to_string(ntohs(address.sin_port)) + "/";
        thread_ = std::thread([this] { run(); });
    }
    ~HttpFixture() {
        stop_ = true;
        thread_.join();
        close(listener_);
    }
    std::vector<std::string> requests() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

private:
    int listener_ = -1;
    std::atomic<bool> stop_ = false;
    std::thread thread_;
    mutable std::mutex mutex_;
    std::vector<std::string> requests_;
    void run() {
        while(!stop_) {
            pollfd ready{listener_, POLLIN, 0};
            if(poll(&ready, 1, 20) <= 0) continue;
            const int fd = accept(listener_, nullptr, nullptr);
            if(fd < 0) continue;
            int one = 1;
            setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
            timeval timeout{2, 0};
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
            std::string request;
            char block[4096];
            while(request.size() < 16384 && request.find("\r\n\r\n") == std::string::npos) {
                const auto n = recv(fd, block, sizeof(block), 0);
                if(n <= 0) break;
                request.append(block, size_t(n));
            }
            {
                std::lock_guard lock(mutex_);
                requests_.push_back(request);
            }
            try {
                const auto reply = response(request);
                const auto range = request.find("Range: bytes=");
                const bool partial = range != std::string::npos && !reply.ignore_range;
                const auto offset = partial ? std::stoull(request.substr(range + 13)) : 0;
                auto body = reply.bytes.substr(offset);
                std::string header = partial ? "HTTP/1.1 206 Partial Content\r\n" : "HTTP/1.1 200 OK\r\n";
                header += "Connection: close\r\nContent-Length: " + std::to_string(body.size()) + "\r\n";
                if(partial)
                    header += "Content-Range: bytes " + std::to_string(offset) + "-" +
                              std::to_string(reply.bytes.size() - 1) + "/" + std::to_string(reply.bytes.size()) +
                              "\r\n";
                if(reply.truncate) body.resize(body.size() / 2);
                const auto wire = header + "\r\n" + body;
                for(size_t sent = 0; sent < wire.size();) {
                    const auto n = send(fd, wire.data() + sent, wire.size() - sent, 0);
                    if(n <= 0) break;
                    sent += size_t(n);
                }
            } catch(...) { /* A malformed request is a closed connection. */
            }
            close(fd);
        }
    }
};

struct PreparedFixture {
    Directory root;
    const checkpoint_io::Geometry geometry{2, 3, 2, 48, 16384};
    std::filesystem::path q4 = root.path / "q4", mixed = root.path / "mixed";
    Json control, consumer, manifest;
    std::map<std::string, std::string> records;
    PreparedFixture() {
        Json tensors = Json::object(), weight_map = Json::object(), layout = Json::array();
        std::string data;
        const std::array<std::string, 3> projections{"gate_proj", "up_proj", "down_proj"};
        const std::array<std::string, 3> pieces{"weight", "scales", "biases"};
        const auto tensor = [&](const std::string& name, const std::string& dtype, const Json& shape,
                                const std::string& bytes) {
            const auto key = "language_model." + name;
            tensors[key] = {{"dtype", dtype},
                            {"shape", shape},
                            {"data_offsets", Json::array({data.size(), data.size() + bytes.size()})}};
            weight_map[key] = "model.safetensors";
            data += bytes;
        };
        Json experts = Json::array(), ngrams = Json::array(), files = Json::array();
        for(uint64_t layer = 0; layer < geometry.layers; ++layer) {
            const std::string name = "experts-0" + std::to_string(layer) + ".bin";
            auto& record = records[name];
            record.resize(geometry.experts * geometry.record_stride, '\0');
            uint64_t offset = 0;
            for(size_t p = 0; p < projections.size(); ++p)
                for(size_t f = 0; f < pieces.size(); ++f) {
                    const auto field = projections[p] + "." + pieces[f];
                    const uint64_t width = f == 0 ? 8 : 4;
                    const std::string dtype = f == 0 ? "U32" : "BF16";
                    std::string bytes;
                    for(uint64_t e = 0; e < geometry.experts; ++e) {
                        const std::string row(width, char(1 + layer * 32 + e * 9 + p * 3 + f));
                        bytes += row;
                        record.replace(e * geometry.record_stride + offset, width, row);
                    }
                    tensor("model.layers." + std::to_string(layer) + ".mlp.switch_mlp." + field, dtype,
                           Json::array({geometry.experts, 2, 1}), bytes);
                    if(layer == 0)
                        layout.push_back(Json{{"name", field},
                                              {"offset", offset},
                                              {"length", width},
                                              {"shape", Json::array({2, 1})},
                                              {"dtype", dtype},
                                              {"bits", 4},
                                              {"group_size", 64}});
                    offset += width;
                }
            experts.push_back(Json{{"layer", layer},
                                   {"file", name},
                                   {"count", geometry.experts},
                                   {"offset", 0},
                                   {"length", 48},
                                   {"stride", 16384},
                                   {"alignment", 16384}});
        }
        const auto fields = Json::parse(
            R"([{"offset":0,"length":80,"dtype":"U32","shape":[20]},{"offset":80,"length":10,"dtype":"BF16","shape":[5]},{"offset":90,"length":10,"dtype":"BF16","shape":[5]}])");
        for(uint64_t shard = 0; shard < geometry.ngram_shards; ++shard) {
            const std::string name = "ngram-00" + std::to_string(shard) + ".bin";
            const auto rows = 3 + 2 * shard;
            auto& record = records[name];
            record.resize(rows * 100, '\0');
            for(size_t f = 0; f < pieces.size(); ++f) {
                const auto width = fields[f]["length"].get<uint64_t>();
                std::string bytes;
                for(uint64_t r = 0; r < rows; ++r) {
                    const std::string row(width, char(1 + shard * 32 + r * 3 + f));
                    bytes += row;
                    record.replace(r * 100 + fields[f]["offset"].get<uint64_t>(), width, row);
                }
                tensor("model.layers.1.ple.ple_embedding.ngram_embedding.shard_" + std::to_string(shard) + "." +
                           pieces[f],
                       fields[f]["dtype"], Json::array({rows, f == 0 ? 20 : 5}), bytes);
            }
            ngrams.push_back(Json{
                {"shard", shard}, {"file", name}, {"count", rows}, {"offset", 0}, {"stride", 100}, {"fields", fields}});
        }
        // Distinct resident bytes make the two source hashes different while
        // every routed-expert and ngram byte stays identical.
        tensor("model.resident", "U8", Json::array({4}), "Q4!!");
        const auto text = tensors.dump();
        const uint64_t length = text.size();
        const std::string header(reinterpret_cast<const char*>(&length), sizeof(length));
        std::map<std::string, std::string> source{
            {"model.safetensors.index.json", Json{{"weight_map", weight_map}}.dump()},
            {"model.safetensors", header + text + data}};
        std::filesystem::create_directories(q4);
        for(const auto& [name, bytes] : source) put(q4 / name, bytes);
        control = file_lock(source);
        source["model.safetensors"].replace(source["model.safetensors"].size() - 4, 4, "Q8!!");
        std::filesystem::create_directories(mixed);
        for(const auto& [name, bytes] : source) put(mixed / name, bytes);
        consumer = file_lock(source);
        consumer["revision"] = "fixture-mixed";
        uint64_t prepared_bytes = 0;
        for(const auto& [name, bytes] : records) {
            files.push_back(file_entry(name, bytes));
            prepared_bytes += bytes.size();
        }
        Json source_files = Json::object();
        for(const auto& entry : control["files"]) source_files[entry["path"].get<std::string>()] = entry["sha256"];
        manifest = {{"schema", 1},
                    {"source_revision", control["revision"]},
                    {"format", "zc-affine-records-v1"},
                    {"recipe", "q4-control-lossless"},
                    {"expert_layout", layout},
                    {"experts", experts},
                    {"ngrams", ngrams},
                    {"files", files},
                    {"prepared_bytes", prepared_bytes},
                    {"ngram_cache_dtype", "BF16"},
                    {"source_files", source_files}};
        control["prepared_control"] = {{"manifest_sha256", digest(manifest.dump(2) + "\n")}};
    }
};
} // namespace

TEST_CASE("checkpoint IO verifies complete files and repairs corruption without clobbering") {
    Directory dir;
    const std::string good = "pinned checkpoint bytes", corrupt(good.size(), '!');
    const auto lock = file_lock({{"weights", good}});
    std::atomic<bool> cancel = false;
    put(dir.path / "weights", good);
    checkpoint_io::verify(dir.path, lock, false, cancel);
    put(dir.path / "weights", corrupt); // same size, stale valid receipt
    std::atomic<bool> saw_corrupt = false;
    HttpFixture server([&](const std::string&) {
        saw_corrupt = read_text(dir.path / "weights") == corrupt;
        return HttpFixture::Response{good};
    });
    const auto result = checkpoint_io::download(dir.path, lock, server.url, cancel, {}, 2);
    CHECK(saw_corrupt.load());
    CHECK(read_text(dir.path / "weights") == good);
    CHECK(result["files"]["weights"]["sha256"] == digest(good));
    CHECK_FALSE(std::filesystem::exists(dir.path / "weights.part"));
    const auto before = file_fingerprint(dir.path / "weights");
    checkpoint_io::download(dir.path, lock, server.url, cancel, {}, 2);
    CHECK(server.requests().size() == 1);
    CHECK(file_fingerprint(dir.path / "weights") == before);
}

TEST_CASE("checkpoint IO bad downloads never publish and retries discard poisoned parts") {
    Directory dir;
    const std::string good = "exact required bytes", corrupt(good.size(), '!');
    const auto lock = file_lock({{"weights", good}});
    std::atomic<bool> cancel = false, bad = true;
    HttpFixture server([&](const std::string&) { return HttpFixture::Response{bad ? corrupt : good}; });
    bool existing = false;
    SUBCASE("missing target") {}
    SUBCASE("existing corrupt target") {
        existing = true;
        put(dir.path / "weights", corrupt);
    }
    CHECK_THROWS_WITH_AS(checkpoint_io::download(dir.path, lock, server.url, cancel, {}, 1),
                         doctest::Contains("hash mismatch"), std::runtime_error);
    CHECK_FALSE(std::filesystem::exists(dir.path / "weights.part"));
    CHECK(std::filesystem::exists(dir.path / "weights") == existing);
    if(existing) CHECK(read_text(dir.path / "weights") == corrupt);
    CHECK(read_json(dir.path / "zerocool-verification.json")["files"].empty());
    // A complete but corrupt part left by an older binary must also recover.
    put(dir.path / "weights.part", corrupt);
    bad = false;
    checkpoint_io::download(dir.path, lock, server.url, cancel, {}, 1);
    CHECK(read_text(dir.path / "weights") == good);
}

TEST_CASE("checkpoint IO resumes partial transfers and safely handles ignored ranges") {
    Directory dir;
    const std::string good = "a small interrupted checkpoint";
    const auto lock = file_lock({{"weights", good}});
    std::atomic<bool> cancel = false, first = true;
    bool ignore_range = false;
    SUBCASE("server honors Range") {}
    SUBCASE("server ignores Range") { ignore_range = true; }
    HttpFixture server(
        [&](const std::string&) { return HttpFixture::Response{good, ignore_range, first.exchange(false)}; });
    CHECK_THROWS(checkpoint_io::download(dir.path, lock, server.url, cancel, {}, 1));
    CHECK_FALSE(std::filesystem::exists(dir.path / "weights"));
    CHECK(std::filesystem::file_size(dir.path / "weights.part") == good.size() / 2);
    const auto progress = [](const FetchProgress& p) { CHECK(p.done <= p.total); };
    checkpoint_io::download(dir.path, lock, server.url, cancel, progress, 1);
    CHECK(read_text(dir.path / "weights") == good);
    const auto requests = server.requests();
    REQUIRE(requests.size() >= 2);
    CHECK(requests[1].find("Range: bytes=" + std::to_string(good.size() / 2) + "-") != std::string::npos);
}

TEST_CASE("checkpoint IO reuses complete parts and preserves revision and cancellation guards") {
    Directory dir;
    const std::string good = "pinned bytes";
    auto lock = file_lock({{"first", good}, {"second", good}});
    std::atomic<bool> cancel = false;
    put(dir.path / "first.part", good);
    HttpFixture server([&](const std::string&) { return HttpFixture::Response{good}; });
    checkpoint_io::download(dir.path, lock, server.url, cancel, {}, 2);
    REQUIRE(server.requests().size() == 1);
    CHECK(server.requests()[0].starts_with("GET /second "));
    const auto receipt = read_text(dir.path / "zerocool-verification.json");
    lock["revision"] = "a different artifact";
    CHECK_THROWS(checkpoint_io::download(dir.path, lock, server.url, cancel, {}, 2));
    CHECK(read_text(dir.path / "zerocool-verification.json") == receipt);
    cancel = true;
    CHECK_THROWS(checkpoint_io::download(dir.path, lock, server.url, cancel, {}, 2));
    CHECK(server.requests().size() == 1);
}

TEST_CASE("checkpoint IO retains verified completions when a concurrent transfer fails") {
    Directory dir;
    const std::string good = "verified download", corrupt(good.size(), '!');
    const auto lock = file_lock({{"first", good}, {"second", good}});
    std::atomic<bool> cancel = false, bad = true;
    HttpFixture server([&](const std::string& request) {
        return HttpFixture::Response{bad && request.starts_with("GET /second ") ? corrupt : good};
    });
    CHECK_THROWS(checkpoint_io::download(dir.path, lock, server.url, cancel, {}, 2));
    // Request completion order is deliberately not assumed. Anything published
    // is verified, and retry must neither refetch nor rewrite those files.
    const auto receipt = read_json(dir.path / "zerocool-verification.json");
    const bool completed = receipt["files"].contains("first");
    Json before;
    if(completed) before = file_fingerprint(dir.path / "first");
    const auto prior_requests = server.requests().size();
    bad = false;
    checkpoint_io::download(dir.path, lock, server.url, cancel, {}, 2);
    CHECK(read_text(dir.path / "first") == good);
    CHECK(read_text(dir.path / "second") == good);
    if(completed) {
        CHECK(file_fingerprint(dir.path / "first") == before);
        const auto requests = server.requests();
        for(size_t i = prior_requests; i < requests.size(); ++i) CHECK_FALSE(requests[i].starts_with("GET /first "));
    }
}

TEST_CASE("checkpoint IO prepares identical canonical records from independently verified Q4 and mixed sources") {
    PreparedFixture f;
    std::atomic<bool> cancel = false;
    for(const bool mixed : {false, true}) {
        const auto& input = mixed ? f.mixed : f.q4;
        const auto& lock = mixed ? f.consumer : f.control;
        const auto output = f.root.path / (mixed ? "mixed-prepared" : "q4-prepared");
        const auto manifest = checkpoint_io::prepare(input, output, lock, f.control, f.geometry, false, cancel);
        CHECK(manifest == f.manifest);
        CHECK(file_digest(output / "manifest.json", cancel) ==
              f.control["prepared_control"]["manifest_sha256"].get<std::string>());
        for(const auto& [name, bytes] : f.records) CHECK(read_text(output / name) == bytes);
        const auto receipt = read_json(output / "verification.json");
        CHECK(receipt["source_revision"] == f.control["revision"]);
        CHECK(receipt["input_revision"] == lock["revision"]);
        CHECK(receipt["input_files"]["model.safetensors"] == lock["files"][0]["sha256"]);
        CHECK(read_json(input / "zerocool-verification.json")["revision"] == lock["revision"]);
        const auto before = file_fingerprint(output / "experts-00.bin");
        CHECK(checkpoint_io::prepare(input, output, lock, f.control, f.geometry, true, cancel) == f.manifest);
        CHECK(file_fingerprint(output / "experts-00.bin") == before);
    }
    const auto pinned = artifact_lock(Artifact::Q4)["prepared_control"]["manifest_sha256"].get<std::string>();
    CHECK_NOTHROW(verify_prepared_compatibility(Artifact::Mixed, pinned));
    CHECK_THROWS(verify_prepared_compatibility(Artifact::Mixed, std::string(64, '0')));
}

TEST_CASE("checkpoint IO preparation rejects wrong pins and resumes only completed records") {
    PreparedFixture f;
    std::atomic<bool> cancel = false;
    const auto output = f.root.path / "prepared";
    SUBCASE("wrong source artifact fails before output") {
        CHECK_THROWS(checkpoint_io::prepare(f.mixed, output, f.control, f.control, f.geometry, false, cancel));
        CHECK_FALSE(std::filesystem::exists(output));
    }
    SUBCASE("output manifest pin is mandatory") {
        auto wrong = f.control;
        wrong["prepared_control"]["manifest_sha256"] = std::string(64, '0');
        CHECK_THROWS(checkpoint_io::prepare(f.mixed, output, f.consumer, wrong, f.geometry, false, cancel));
        CHECK_FALSE(std::filesystem::exists(output / "manifest.json"));
        CHECK_FALSE(read_json(output / "verification.json").contains("manifest_sha256"));
    }
    SUBCASE("cancel and resume through the other equivalent artifact") {
        const auto stop = [&](const FetchProgress&) { cancel = true; };
        CHECK_THROWS(checkpoint_io::prepare(f.q4, output, f.control, f.control, f.geometry, false, cancel, stop));
        CHECK_FALSE(std::filesystem::exists(output / "manifest.json"));
        const auto before = file_fingerprint(output / "experts-00.bin");
        cancel = false;
        CHECK(checkpoint_io::prepare(f.mixed, output, f.consumer, f.control, f.geometry, false, cancel) == f.manifest);
        CHECK(file_fingerprint(output / "experts-00.bin") == before);
        CHECK(read_json(output / "verification.json")["input_revision"] == f.consumer["revision"]);
    }
}
