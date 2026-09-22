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

} // namespace

Json file_fingerprint(const std::filesystem::path& path) {
    struct stat st{};
    check(::stat(path.c_str(), &st) == 0, "missing file: " + path.string());
    return Json{{"size", uint64_t(st.st_size)}, {"device", uint64_t(st.st_dev)}, {"inode", uint64_t(st.st_ino)},
                {"mtime_ns", uint64_t(st.st_mtimespec.tv_sec) * 1000000000 + uint64_t(st.st_mtimespec.tv_nsec)},
                {"ctime_ns", uint64_t(st.st_ctimespec.tv_sec) * 1000000000 + uint64_t(st.st_ctimespec.tv_nsec)}};
}

std::string file_digest(const std::filesystem::path& path, const std::atomic<bool>& cancel) {
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

void write_json_atomic(const std::filesystem::path& path, const Json& value) {
    const auto temporary = std::filesystem::path(path).concat(".tmp");
    { std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
      if(!out) throw std::runtime_error("cannot write: " + temporary.string());
      out << value.dump(2) << "\n";
      if(!out) throw std::runtime_error("cannot write: " + temporary.string()); }
    std::filesystem::rename(temporary, path);
}

namespace {

void require_safe_name(const std::string& name) {
    check(std::filesystem::path(name).filename() == name && name != "" && name != "." && name != "..",
          "unsafe artifact filename: " + name);
}

// One in-flight transfer. Each owns its own part file and easy handle, so a
// failure or a resume decision on one says nothing about the others.
struct Transfer {
    std::string url, name;
    std::filesystem::path part, target;
    uint64_t total = 0, resumed = 0, received = 0;
    std::FILE* file = nullptr;
    CURL* easy = nullptr;
    bool checked_code = false;
    std::exception_ptr error;
    ~Transfer() { if(file) std::fclose(file); }
};

// `settled` counts bytes that are already on disk for good: completed files
// plus the part each in-flight transfer resumed from. Live progress adds what
// the running transfers have received since.
struct Fleet {
    uint64_t settled = 0, done = 0, total = 0, reported = 0;
    unsigned active = 0;
    const ProgressFn* progress = nullptr;
};

void report(Fleet& fleet, const Transfer& transfer, bool force) {
    if(!fleet.progress || !*fleet.progress) return;
    if(!force && fleet.done - fleet.reported < 128 * MiB) return;
    fleet.reported = fleet.done;
    (*fleet.progress)(FetchProgress{transfer.name, fleet.done, fleet.total, fleet.active, transfer.resumed > 0, "fetched"});
}

size_t on_body(char* data, size_t size, size_t count, void* raw) {
    auto& transfer = *static_cast<Transfer*>(raw);
    const auto bytes = size * count;
    try {
        if(!transfer.checked_code) {
            long code = 0;
            curl_easy_getinfo(transfer.easy, CURLINFO_RESPONSE_CODE, &code);
            // A ranged request answered with 200 carries the whole file, so the
            // bytes already on disk are not a prefix of what follows.
            if(transfer.resumed && code == 200) {
                check(std::fseek(transfer.file, 0, SEEK_SET) == 0, "cannot rewind partial download");
                check(::ftruncate(::fileno(transfer.file), 0) == 0, "cannot reset partial download");
                transfer.resumed = 0;
            }
            check(code == 200 || code == 206, "HTTP " + std::to_string(code) + " fetching " + transfer.name);
            transfer.checked_code = true;
        }
        check(transfer.resumed + transfer.received + bytes <= transfer.total,
              "server sent more than the pinned size for " + transfer.name);
        check(std::fwrite(data, 1, bytes, transfer.file) == bytes, "short write for " + transfer.name);
        transfer.received += bytes;
        return bytes;
    } catch(...) { transfer.error = std::current_exception(); return 0; }
}

void start(Transfer& transfer, const std::atomic<bool>& cancel, curl_slist* headers) {
    uint64_t resume = std::filesystem::exists(transfer.part) ? uint64_t(std::filesystem::file_size(transfer.part)) : 0;
    // A stale part longer than the pinned size cannot be a prefix of it.
    if(resume > transfer.total) { std::filesystem::remove(transfer.part); resume = 0; }
    transfer.resumed = resume;
    transfer.file = std::fopen(transfer.part.c_str(), resume ? "r+b" : "wb");
    check(transfer.file != nullptr, "cannot open: " + transfer.part.string());
    check(std::fseek(transfer.file, long(resume), SEEK_SET) == 0, "cannot seek: " + transfer.part.string());

    transfer.easy = curl_easy_init();
    check(transfer.easy != nullptr, "cannot create HTTP client");
    auto* easy = transfer.easy;
    curl_easy_setopt(easy, CURLOPT_URL, transfer.url.c_str());
    curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);   // the hub redirects to a CDN
    curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT_MS, 20000L);
    curl_easy_setopt(easy, CURLOPT_LOW_SPEED_LIMIT, 1L);  // fail a stalled transfer rather than hang
    curl_easy_setopt(easy, CURLOPT_LOW_SPEED_TIME, 120L);
    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, &on_body);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA, &transfer);
    curl_easy_setopt(easy, CURLOPT_PRIVATE, &transfer);
    if(resume) curl_easy_setopt(easy, CURLOPT_RESUME_FROM_LARGE, curl_off_t(resume));
    curl_easy_setopt(easy, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(easy, CURLOPT_XFERINFODATA, const_cast<std::atomic<bool>*>(&cancel));
    curl_easy_setopt(easy, CURLOPT_XFERINFOFUNCTION,
        +[](void* raw, curl_off_t, curl_off_t, curl_off_t, curl_off_t) -> int {
            return static_cast<std::atomic<bool>*>(raw)->load() ? 1 : 0;
        });
    if(headers) curl_easy_setopt(easy, CURLOPT_HTTPHEADER, headers);
}

// Run every transfer, at most `jobs` at a time. A finished file is renamed only
// once its full pinned length is present, so an interrupted run leaves parts
// that the next run continues rather than a short file that looks complete.
void fetch_all(std::vector<std::unique_ptr<Transfer>>& queue, unsigned jobs,
               const std::atomic<bool>& cancel, const ProgressFn& progress) {
    static std::once_flag initialized;
    std::call_once(initialized, [] { check(curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK, "curl initialization failed"); });

    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(nullptr, curl_slist_free_all);
    if(const char* token = std::getenv("HF_TOKEN"); token && *token) {
        headers.reset(curl_slist_append(nullptr, ("Authorization: Bearer " + std::string(token)).c_str()));
        check(bool(headers), "cannot allocate HTTP headers");
    }

    Fleet fleet;
    fleet.progress = &progress;
    for(const auto& transfer : queue) fleet.total += transfer->total;

    using Multi = std::unique_ptr<CURLM, decltype(&curl_multi_cleanup)>;
    Multi multi(curl_multi_init(), curl_multi_cleanup);
    check(bool(multi), "cannot create HTTP client");

    size_t next = 0;
    std::vector<Transfer*> running;
    struct Guard {
        CURLM* multi;
        std::vector<Transfer*>& running;
        ~Guard() { for(auto* t : running) { curl_multi_remove_handle(multi, t->easy); curl_easy_cleanup(t->easy); t->easy = nullptr; } }
    } guard{multi.get(), running};

    while(next < queue.size() || !running.empty()) {
        check(!cancel.load(), "cancelled");
        while(running.size() < std::max(1u, jobs) && next < queue.size()) {
            auto& transfer = *queue[next++];
            if(std::filesystem::exists(transfer.target)) { fleet.settled += transfer.total; continue; }
            start(transfer, cancel, headers.get());
            if(transfer.resumed == transfer.total) {   // already complete on disk
                std::fclose(transfer.file); transfer.file = nullptr;
                curl_easy_cleanup(transfer.easy); transfer.easy = nullptr;
                std::filesystem::rename(transfer.part, transfer.target);
                fleet.settled += transfer.total;
                continue;
            }
            fleet.settled += transfer.resumed;
            check(curl_multi_add_handle(multi.get(), transfer.easy) == CURLM_OK, "cannot schedule transfer");
            running.push_back(&transfer);
            fleet.active = unsigned(running.size());
            report(fleet, transfer, true);
        }
        if(running.empty()) continue;

        int alive = 0;
        check(curl_multi_perform(multi.get(), &alive) == CURLM_OK, "transfer progress failed");
        fleet.done = fleet.settled;
        for(const auto* t : running) fleet.done += t->received;
        if(!running.empty()) report(fleet, *running.front(), false);
        if(alive) { int fds = 0; check(curl_multi_poll(multi.get(), nullptr, 0, 100, &fds) == CURLM_OK, "transfer poll failed"); }

        int left = 0;
        while(CURLMsg* message = curl_multi_info_read(multi.get(), &left)) {
            if(message->msg != CURLMSG_DONE) continue;
            Transfer* transfer = nullptr;
            curl_easy_getinfo(message->easy_handle, CURLINFO_PRIVATE, &transfer);
            const auto result = message->data.result;
            curl_multi_remove_handle(multi.get(), message->easy_handle);
            running.erase(std::remove(running.begin(), running.end(), transfer), running.end());
            curl_easy_cleanup(transfer->easy);
            transfer->easy = nullptr;
            if(transfer->error) std::rethrow_exception(transfer->error);
            check(result == CURLE_OK, "fetch failed for " + transfer->name + ": " + curl_easy_strerror(result));
            std::fflush(transfer->file);
            std::fclose(transfer->file);
            transfer->file = nullptr;
            const auto have = transfer->resumed + transfer->received;
            check(have == transfer->total, "short transfer for " + transfer->name + ": " + std::to_string(have) +
                                           " of " + std::to_string(transfer->total) + " bytes; rerun to resume");
            std::filesystem::rename(transfer->part, transfer->target);
            fleet.settled += transfer->received;
            fleet.done = fleet.settled;
            fleet.active = unsigned(running.size());
            report(fleet, *transfer, true);
        }
    }
}

} // namespace

Json verify_checkpoint(const std::filesystem::path& directory, Artifact artifact, bool cached,
                       const std::atomic<bool>& cancel, const ProgressFn& progress) {
    const auto lock = artifact_lock(artifact);
    const auto receipt_path = directory / "zerocool-verification.json";
    Json old = cached && std::filesystem::exists(receipt_path) ? read_json(receipt_path) : Json::object();
    Json result{{"schema", 1}, {"revision", lock.at("revision")},
                {"verified_at", uint64_t(std::time(nullptr))}, {"files", Json::object()}};
    uint64_t hashed = 0, pinned = 0;
    for(const auto& entry : lock.at("files"))
        if(!entry.value("optional", false)) pinned += entry.at("size").get<uint64_t>();
    for(const auto& entry : lock.at("files")) {
        if(entry.value("optional", false)) continue;
        check(!cancel.load(), "cancelled");
        const auto name = entry.at("path").get<std::string>();
        require_safe_name(name);
        const auto path = directory / name;
        const auto before = file_fingerprint(path);
        const auto size = entry.at("size").get<uint64_t>();
        check(before.at("size").get<uint64_t>() == size, "wrong size: " + path.string());
        auto expected = before;
        expected["sha256"] = entry.at("sha256");
        std::string digest;
        if(cached && old.value("revision", std::string()) == lock.at("revision").get<std::string>() &&
           old.contains("files") && old.at("files").contains(name) && old.at("files").at(name) == expected)
            digest = expected.at("sha256").get<std::string>();
        else
            digest = file_digest(path, cancel);
        check(digest == entry.at("sha256").get<std::string>() && file_fingerprint(path) == before,
              "hash mismatch or file changed during verification: " + path.string());
        auto record = before;
        record["sha256"] = digest;
        result["files"][name] = record;
        hashed += size;
        if(progress) progress(FetchProgress{name, hashed, pinned, 0, false, "verified"});
    }
    write_json_atomic(receipt_path, result);
    return result;
}

Json download_checkpoint(const std::filesystem::path& directory, Artifact artifact,
                         const std::atomic<bool>& cancel, const ProgressFn& progress, unsigned jobs) {
    check(jobs >= 1 && jobs <= MaxFetchJobs, "concurrent jobs must be between 1 and " + std::to_string(MaxFetchJobs));
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
    std::vector<std::unique_ptr<Transfer>> queue;
    for(const auto& entry : lock.at("files")) {
        if(entry.value("optional", false)) continue;
        const auto name = entry.at("path").get<std::string>();
        require_safe_name(name);
        const auto size = entry.at("size").get<uint64_t>();
        const auto target = directory / name;
        if(std::filesystem::exists(target)) {
            check(uint64_t(std::filesystem::file_size(target)) == size,
                  "refusing to replace a different existing artifact file: " + name);
            continue;
        }
        const auto part = std::filesystem::path(target).concat(".part");
        const auto have = std::filesystem::exists(part) ? uint64_t(std::filesystem::file_size(part)) : 0;
        remaining += size - std::min(have, size);
        auto transfer = std::make_unique<Transfer>();
        transfer->url = "https://huggingface.co/" + repo + "/resolve/" + revision + "/" + name;
        transfer->name = name;
        transfer->part = part;
        transfer->target = target;
        transfer->total = size;
        queue.push_back(std::move(transfer));
    }
    const auto space = std::filesystem::space(directory);
    check(space.available >= remaining + DiskReserve,
          "insufficient disk space for the pinned checkpoint plus a 5GiB reserve");

    fetch_all(queue, jobs, cancel, progress);
    return verify_checkpoint(directory, artifact, false, cancel, progress);
}

} // namespace zerocool::engine
