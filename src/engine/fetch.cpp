#include "engine/fetch.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <curl/curl.h>
#include <cstdio>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <print>
#include <sys/stat.h>
#include <unistd.h>

namespace zerocool::engine {
namespace {

constexpr uint64_t HashBlock = 8 * MiB;
constexpr uint64_t DiskReserve = 5 * GiB;

void check(bool ok, const std::string& message) {
    if(!ok) throw std::runtime_error(message);
}

// The stat identity the receipt records. A file whose size, location or
// timestamps moved is reread rather than trusted.
Json fingerprint(const std::filesystem::path& path) {
    struct stat st{};
    check(::stat(path.c_str(), &st) == 0, "missing file: " + path.string());
    return Json{{"size", uint64_t(st.st_size)}, {"device", uint64_t(st.st_dev)}, {"inode", uint64_t(st.st_ino)},
                {"mtime_ns", uint64_t(st.st_mtimespec.tv_sec) * 1000000000 + uint64_t(st.st_mtimespec.tv_nsec)},
                {"ctime_ns", uint64_t(st.st_ctimespec.tv_sec) * 1000000000 + uint64_t(st.st_ctimespec.tv_nsec)}};
}

std::string hash_file(const std::filesystem::path& path, const std::atomic<bool>& cancel) {
    struct Fd {
        int value;
        ~Fd() { if(value >= 0) ::close(value); }
    } fd{::open(path.c_str(), O_RDONLY)};
    check(fd.value >= 0, "cannot open: " + path.string());
    // Verification reads the whole checkpoint; leaving 104GB in the page cache
    // would evict everything the engine is about to want.
    ::fcntl(fd.value, F_NOCACHE, 1);
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    std::vector<std::byte> block(HashBlock);
    for(uint64_t offset = 0;;) {
        check(!cancel.load(), "cancelled");
        const auto got = ::pread(fd.value, block.data(), block.size(), off_t(offset));
        check(got >= 0, "read failed: " + path.string());
        if(got == 0) break;
        CC_SHA256_Update(&context, block.data(), CC_LONG(got));
        offset += uint64_t(got);
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    for(auto c : digest) { out += hex[c >> 4]; out += hex[c & 15]; }
    return out;
}

void write_atomic(const std::filesystem::path& path, const std::string& text) {
    const auto temporary = std::filesystem::path(path).concat(".tmp");
    { std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
      check(bool(out), "cannot write: " + temporary.string());
      out << text;
      check(bool(out), "cannot write: " + temporary.string()); }
    std::filesystem::rename(temporary, path);
}

void require_safe_name(const std::string& name) {
    check(std::filesystem::path(name).filename() == name && name != "" && name != "." && name != "..",
          "unsafe artifact filename: " + name);
}

struct Sink {
    std::FILE* file = nullptr;
    CURL* easy = nullptr;
    uint64_t resumed = 0, received = 0, total = 0;
    bool checked_code = false;
    std::string name;
    const ProgressFn* progress = nullptr;
    uint64_t reported = 0;
    std::exception_ptr error;
};

size_t on_body(char* data, size_t size, size_t count, void* raw) {
    auto& sink = *static_cast<Sink*>(raw);
    const auto bytes = size * count;
    try {
        if(!sink.checked_code) {
            long code = 0;
            curl_easy_getinfo(sink.easy, CURLINFO_RESPONSE_CODE, &code);
            // A ranged request answered with 200 carries the whole file, so the
            // bytes already on disk are not a prefix of what follows.
            if(sink.resumed && code == 200) {
                check(std::fseek(sink.file, 0, SEEK_SET) == 0, "cannot rewind partial download");
                check(::ftruncate(::fileno(sink.file), 0) == 0, "cannot reset partial download");
                sink.received = 0;
                sink.resumed = 0;
            }
            check(code == 200 || code == 206, "HTTP " + std::to_string(code) + " fetching " + sink.name);
            sink.checked_code = true;
        }
        check(sink.resumed + sink.received + bytes <= sink.total, "server sent more than the pinned size for " + sink.name);
        check(std::fwrite(data, 1, bytes, sink.file) == bytes, "short write for " + sink.name);
        sink.received += bytes;
        const auto done = sink.resumed + sink.received;
        if(sink.progress && *sink.progress && (done - sink.reported >= 64 * MiB || done == sink.total)) {
            sink.reported = done;
            (*sink.progress)(FetchProgress{sink.name, sink.resumed, sink.received, sink.total});
        }
        return bytes;
    } catch(...) { sink.error = std::current_exception(); return 0; }
}

void fetch_file(const std::string& url, const std::filesystem::path& part, const std::string& name,
                uint64_t total, const std::atomic<bool>& cancel, const ProgressFn& progress) {
    static std::once_flag initialized;
    std::call_once(initialized, [] { check(curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK, "curl initialization failed"); });

    uint64_t resume = std::filesystem::exists(part) ? uint64_t(std::filesystem::file_size(part)) : 0;
    // A stale part longer than the pinned size cannot be a prefix of it.
    if(resume > total) { std::filesystem::remove(part); resume = 0; }
    if(resume == total) return;

    struct Handle {
        std::FILE* value;
        ~Handle() { if(value) std::fclose(value); }
    } file{std::fopen(part.c_str(), resume ? "r+b" : "wb")};
    check(file.value != nullptr, "cannot open: " + part.string());
    check(std::fseek(file.value, long(resume), SEEK_SET) == 0, "cannot seek: " + part.string());

    using Easy = std::unique_ptr<CURL, decltype(&curl_easy_cleanup)>;
    Easy easy(curl_easy_init(), curl_easy_cleanup);
    check(bool(easy), "cannot create HTTP client");
    Sink sink{file.value, easy.get(), resume, 0, total, false, name, &progress, resume, {}};
    char error[CURL_ERROR_SIZE] = {};

    curl_easy_setopt(easy.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(easy.get(), CURLOPT_FOLLOWLOCATION, 1L);   // the hub redirects to a CDN
    curl_easy_setopt(easy.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(easy.get(), CURLOPT_CONNECTTIMEOUT_MS, 20000L);
    curl_easy_setopt(easy.get(), CURLOPT_LOW_SPEED_LIMIT, 1L);  // fail a stalled transfer rather than hang
    curl_easy_setopt(easy.get(), CURLOPT_LOW_SPEED_TIME, 120L);
    curl_easy_setopt(easy.get(), CURLOPT_ERRORBUFFER, error);
    curl_easy_setopt(easy.get(), CURLOPT_WRITEFUNCTION, &on_body);
    curl_easy_setopt(easy.get(), CURLOPT_WRITEDATA, &sink);
    if(resume) curl_easy_setopt(easy.get(), CURLOPT_RESUME_FROM_LARGE, curl_off_t(resume));
    curl_easy_setopt(easy.get(), CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(easy.get(), CURLOPT_XFERINFODATA, const_cast<std::atomic<bool>*>(&cancel));
    curl_easy_setopt(easy.get(), CURLOPT_XFERINFOFUNCTION,
        +[](void* raw, curl_off_t, curl_off_t, curl_off_t, curl_off_t) -> int {
            return static_cast<std::atomic<bool>*>(raw)->load() ? 1 : 0;
        });

    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(nullptr, curl_slist_free_all);
    if(const char* token = std::getenv("HF_TOKEN"); token && *token) {
        headers.reset(curl_slist_append(nullptr, ("Authorization: Bearer " + std::string(token)).c_str()));
        check(bool(headers), "cannot allocate HTTP headers");
        curl_easy_setopt(easy.get(), CURLOPT_HTTPHEADER, headers.get());
    }

    const auto result = curl_easy_perform(easy.get());
    if(sink.error) std::rethrow_exception(sink.error);
    check(!cancel.load(), "cancelled");
    check(result == CURLE_OK, std::string("fetch failed for ") + name + ": " + (error[0] ? error : curl_easy_strerror(result)));
    std::fflush(file.value);
    const auto have = sink.resumed + sink.received;
    check(have == total, "short transfer for " + name + ": " + std::to_string(have) + " of " + std::to_string(total) +
                         " bytes; rerun to resume");
}

} // namespace

Json verify_checkpoint(const std::filesystem::path& directory, Artifact artifact, bool cached,
                       const std::atomic<bool>& cancel, const ProgressFn& progress) {
    const auto lock = artifact_lock(artifact);
    const auto receipt_path = directory / "zerocool-verification.json";
    Json old = cached && std::filesystem::exists(receipt_path) ? read_json(receipt_path) : Json::object();
    Json result{{"schema", 1}, {"revision", lock.at("revision")},
                {"verified_at", uint64_t(std::time(nullptr))}, {"files", Json::object()}};
    for(const auto& entry : lock.at("files")) {
        if(entry.value("optional", false)) continue;
        check(!cancel.load(), "cancelled");
        const auto name = entry.at("path").get<std::string>();
        require_safe_name(name);
        const auto path = directory / name;
        const auto before = fingerprint(path);
        const auto size = entry.at("size").get<uint64_t>();
        check(before.at("size").get<uint64_t>() == size, "wrong size: " + path.string());
        auto expected = before;
        expected["sha256"] = entry.at("sha256");
        std::string digest;
        if(cached && old.value("revision", std::string()) == lock.at("revision").get<std::string>() &&
           old.contains("files") && old.at("files").contains(name) && old.at("files").at(name) == expected)
            digest = expected.at("sha256").get<std::string>();
        else
            digest = hash_file(path, cancel);
        check(digest == entry.at("sha256").get<std::string>() && fingerprint(path) == before,
              "hash mismatch or file changed during verification: " + path.string());
        auto record = before;
        record["sha256"] = digest;
        result["files"][name] = record;
        if(progress) progress(FetchProgress{name, size, 0, size});
    }
    write_atomic(receipt_path, result.dump(2) + "\n");
    return result;
}

Json download_checkpoint(const std::filesystem::path& directory, Artifact artifact,
                         const std::atomic<bool>& cancel, const ProgressFn& progress) {
    const auto lock = artifact_lock(artifact);
    const auto repo = lock.at("repo").get<std::string>();
    const auto revision = lock.at("revision").get<std::string>();
    std::filesystem::create_directories(directory);

    // Refuse to mix a second pinned artifact into a directory that already
    // holds one, and refuse to overwrite a file that is not simply absent.
    const auto receipt_path = directory / "zerocool-verification.json";
    if(std::filesystem::exists(receipt_path)) {
        const auto receipt = read_json(receipt_path);
        check(receipt.value("revision", std::string()) == revision,
              "refusing to replace a different verified artifact; choose its own directory");
    }
    uint64_t remaining = 0;
    struct Want { std::string name; uint64_t size; };
    std::vector<Want> wanted;
    for(const auto& entry : lock.at("files")) {
        if(entry.value("optional", false)) continue;
        const auto name = entry.at("path").get<std::string>();
        require_safe_name(name);
        const auto size = entry.at("size").get<uint64_t>();
        const auto path = directory / name;
        if(std::filesystem::exists(path)) {
            check(uint64_t(std::filesystem::file_size(path)) == size,
                  "refusing to replace a different existing artifact file: " + name);
            continue;
        }
        const auto part = std::filesystem::path(path).concat(".part");
        const auto have = std::filesystem::exists(part) ? uint64_t(std::filesystem::file_size(part)) : 0;
        remaining += size - std::min(have, size);
        wanted.push_back({name, size});
    }
    const auto space = std::filesystem::space(directory);
    check(space.available >= remaining + DiskReserve,
          "insufficient disk space for the pinned checkpoint plus a 5GiB reserve");

    for(const auto& want : wanted) {
        check(!cancel.load(), "cancelled");
        const auto path = directory / want.name;
        const auto part = std::filesystem::path(path).concat(".part");
        fetch_file("https://huggingface.co/" + repo + "/resolve/" + revision + "/" + want.name,
                   part, want.name, want.size, cancel, progress);
        std::filesystem::rename(part, path);
    }
    return verify_checkpoint(directory, artifact, false, cancel, progress);
}

} // namespace zerocool::engine
