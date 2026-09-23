#include "checkpoint_io.hpp"
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
    return Json{{"size", uint64_t(st.st_size)},
                {"device", uint64_t(st.st_dev)},
                {"inode", uint64_t(st.st_ino)},
                {"mtime_ns", uint64_t(st.st_mtimespec.tv_sec) * 1000000000 + uint64_t(st.st_mtimespec.tv_nsec)},
                {"ctime_ns", uint64_t(st.st_ctimespec.tv_sec) * 1000000000 + uint64_t(st.st_ctimespec.tv_nsec)}};
}

std::string file_digest(const std::filesystem::path& path, const std::atomic<bool>& cancel) {
    struct Fd {
        int value;
        ~Fd() {
            if(value >= 0) ::close(value);
        }
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
    for(auto c : digest) {
        out += hex[c >> 4];
        out += hex[c & 15];
    }
    return out;
}

void write_json_atomic(const std::filesystem::path& path, const Json& value) {
    const auto temporary = std::filesystem::path(path).concat(".tmp");
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        if(!out) throw std::runtime_error("cannot write: " + temporary.string());
        out << value.dump(2) << "\n";
        if(!out) throw std::runtime_error("cannot write: " + temporary.string());
    }
    std::filesystem::rename(temporary, path);
}

namespace {

void require_safe_name(const std::string& name) {
    check(std::filesystem::path(name).filename() == name && name != "" && name != "." && name != "..",
          "unsafe artifact filename: " + name);
}

// A receipt is an integrity cache. Only the pinned digest with the current
// fingerprint can avoid a rehash; equal length alone never qualifies a file.
Json verified_file(const std::filesystem::path& path, uint64_t size, const std::string& digest, const Json& saved,
                   const std::atomic<bool>& cancel) {
    const auto before = file_fingerprint(path);
    if(before.at("size").get<uint64_t>() != size) return nullptr;
    auto expected = before;
    expected["sha256"] = digest;
    const bool valid = saved == expected || file_digest(path, cancel) == digest;
    check(file_fingerprint(path) == before, "file changed during verification: " + path.string());
    return valid ? expected : Json(nullptr);
}

// One in-flight transfer owns its part and easy handle, including startup errors.
struct Transfer {
    std::string url, name, digest;
    std::filesystem::path part, target;
    uint64_t total = 0, resumed = 0, received = 0;
    std::FILE* file = nullptr;
    CURL* easy = nullptr;
    Json complete_part;
    bool checked_code = false;
    std::exception_ptr error;
    ~Transfer() {
        if(easy) curl_easy_cleanup(easy);
        if(file) std::fclose(file);
    }
};

// Settled bytes belong to verified complete files. Partial bytes are counted
// only while their transfer is live, so restarting a range cannot double count.
struct Fleet {
    uint64_t settled = 0, done = 0, total = 0, reported = 0;
    unsigned active = 0;
    const ProgressFn* progress = nullptr;
};

void close_part(Transfer& transfer) {
    check(std::fflush(transfer.file) == 0 && ::fsync(::fileno(transfer.file)) == 0,
          "cannot flush partial download: " + transfer.name);
    auto* file = transfer.file;
    transfer.file = nullptr;
    check(std::fclose(file) == 0, "cannot close partial download: " + transfer.name);
}

void report(Fleet& fleet, const Transfer& transfer, bool force) {
    if(!fleet.progress || !*fleet.progress) return;
    if(!force && fleet.done >= fleet.reported && fleet.done - fleet.reported < 128 * MiB) return;
    fleet.reported = fleet.done;
    (*fleet.progress)(
        FetchProgress{transfer.name, fleet.done, fleet.total, fleet.active, transfer.resumed > 0, "fetched"});
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
    } catch(...) {
        transfer.error = std::current_exception();
        return 0;
    }
}

void start(Transfer& transfer, const std::atomic<bool>& cancel, curl_slist* headers) {
    uint64_t resume = std::filesystem::exists(transfer.part) ? uint64_t(std::filesystem::file_size(transfer.part)) : 0;
    // A stale part longer than the pinned size cannot be a prefix of it.
    if(resume > transfer.total) {
        std::filesystem::remove(transfer.part);
        resume = 0;
    }
    transfer.resumed = resume;
    transfer.file = std::fopen(transfer.part.c_str(), resume ? "r+b" : "wb");
    check(transfer.file != nullptr, "cannot open: " + transfer.part.string());
    check(std::fseek(transfer.file, long(resume), SEEK_SET) == 0, "cannot seek: " + transfer.part.string());

    transfer.easy = curl_easy_init();
    check(transfer.easy != nullptr, "cannot create HTTP client");
    auto* easy = transfer.easy;
    curl_easy_setopt(easy, CURLOPT_URL, transfer.url.c_str());
    curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L); // the hub redirects to a CDN
    curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT_MS, 20000L);
    curl_easy_setopt(easy, CURLOPT_LOW_SPEED_LIMIT, 1L); // fail a stalled transfer rather than hang
    curl_easy_setopt(easy, CURLOPT_LOW_SPEED_TIME, 120L);
    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, &on_body);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA, &transfer);
    curl_easy_setopt(easy, CURLOPT_PRIVATE, &transfer);
    if(resume) curl_easy_setopt(easy, CURLOPT_RESUME_FROM_LARGE, curl_off_t(resume));
    curl_easy_setopt(easy, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(easy, CURLOPT_XFERINFODATA, const_cast<std::atomic<bool>*>(&cancel));
    curl_easy_setopt(
        easy, CURLOPT_XFERINFOFUNCTION, +[](void* raw, curl_off_t, curl_off_t, curl_off_t, curl_off_t) -> int {
            return static_cast<std::atomic<bool>*>(raw)->load() ? 1 : 0;
        });
    if(headers) curl_easy_setopt(easy, CURLOPT_HTTPHEADER, headers);
}

// Publish only after a full hash. In particular a corrupt final file remains
// intact until its replacement is verified, and a bad part cannot poison retries.
void publish(Transfer& transfer, Json& receipt, const std::filesystem::path& receipt_path) {
    auto fingerprint = transfer.complete_part;
    fingerprint.erase("sha256");
    check(file_fingerprint(transfer.part) == fingerprint, "partial file changed before publication: " + transfer.name);
    std::filesystem::rename(transfer.part, transfer.target);
    auto record = file_fingerprint(transfer.target);
    record["sha256"] = transfer.digest;
    receipt["files"][transfer.name] = record;
    write_json_atomic(receipt_path, receipt);
}

// Run at most `jobs` transfers at a time. Interrupted files remain resumable;
// completed ones are verified before receiving their final names and receipts.
void fetch_all(std::vector<std::unique_ptr<Transfer>>& queue, unsigned jobs, const std::atomic<bool>& cancel,
               const ProgressFn& progress, Json& receipt, const std::filesystem::path& receipt_path, uint64_t total,
               uint64_t settled) {
    static std::once_flag initialized;
    std::call_once(initialized,
                   [] { check(curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK, "curl initialization failed"); });

    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(nullptr, curl_slist_free_all);
    if(const char* token = std::getenv("HF_TOKEN");
       token && *token && !queue.empty() && queue.front()->url.starts_with("https://huggingface.co/")) {
        headers.reset(curl_slist_append(nullptr, ("Authorization: Bearer " + std::string(token)).c_str()));
        check(bool(headers), "cannot allocate HTTP headers");
    }

    Fleet fleet;
    fleet.progress = &progress;
    fleet.total = total;
    fleet.settled = settled;
    fleet.done = settled;
    using Multi = std::unique_ptr<CURLM, decltype(&curl_multi_cleanup)>;
    Multi multi(curl_multi_init(), curl_multi_cleanup);
    check(bool(multi), "cannot create HTTP client");

    size_t next = 0;
    std::vector<Transfer*> running;
    struct Guard {
        CURLM* multi;
        std::vector<Transfer*>& running;
        ~Guard() {
            for(auto* t : running) {
                curl_multi_remove_handle(multi, t->easy);
                curl_easy_cleanup(t->easy);
                t->easy = nullptr;
            }
        }
    } guard{multi.get(), running};
    const auto update = [&] {
        fleet.done = fleet.settled;
        for(const auto* t : running) fleet.done += t->resumed + t->received;
        fleet.active = unsigned(running.size());
    };

    while(next < queue.size() || !running.empty()) {
        check(!cancel.load(), "cancelled");
        while(running.size() < std::max(1u, jobs) && next < queue.size()) {
            auto& transfer = *queue[next++];
            if(!transfer.complete_part.is_null()) {
                publish(transfer, receipt, receipt_path);
                fleet.settled += transfer.total;
                update();
                report(fleet, transfer, true);
                continue;
            }
            start(transfer, cancel, headers.get());
            check(curl_multi_add_handle(multi.get(), transfer.easy) == CURLM_OK, "cannot schedule transfer");
            running.push_back(&transfer);
            update();
            report(fleet, transfer, true);
        }
        if(running.empty()) continue;

        int alive = 0;
        check(curl_multi_perform(multi.get(), &alive) == CURLM_OK, "transfer progress failed");
        update();
        report(fleet, *running.front(), false);
        if(alive) {
            int fds = 0;
            check(curl_multi_poll(multi.get(), nullptr, 0, 100, &fds) == CURLM_OK, "transfer poll failed");
        }

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
            if(result == CURLE_RANGE_ERROR && transfer->resumed > 0) {
                // Some servers ignore Range; libcurl can reject their 200 before
                // invoking on_body. Retry once from zero, never append that body.
                close_part(*transfer);
                std::filesystem::remove(transfer->part);
                transfer->received = 0;
                transfer->checked_code = false;
                start(*transfer, cancel, headers.get());
                check(curl_multi_add_handle(multi.get(), transfer->easy) == CURLM_OK, "cannot restart transfer");
                running.push_back(transfer);
                update();
                report(fleet, *transfer, true);
                continue;
            }
            check(result == CURLE_OK, "fetch failed for " + transfer->name + ": " + curl_easy_strerror(result));
            close_part(*transfer);
            const auto have = transfer->resumed + transfer->received;
            check(have == transfer->total, "short transfer for " + transfer->name + ": " + std::to_string(have) +
                                               " of " + std::to_string(transfer->total) + " bytes; rerun to resume");
            transfer->complete_part = verified_file(transfer->part, transfer->total, transfer->digest, nullptr, cancel);
            if(transfer->complete_part.is_null()) {
                std::filesystem::remove(transfer->part);
                throw std::runtime_error("hash mismatch for " + transfer->name +
                                         "; discarded corrupt partial file; rerun to retry");
            }
            publish(*transfer, receipt, receipt_path);
            fleet.settled += transfer->total;
            update();
            report(fleet, *transfer, true);
        }
    }
}

} // namespace

Json checkpoint_io::verify(const std::filesystem::path& directory, const Json& lock, bool cached,
                           const std::atomic<bool>& cancel, const ProgressFn& progress) {
    const auto receipt_path = directory / "zerocool-verification.json";
    Json old = cached && std::filesystem::exists(receipt_path) ? read_json(receipt_path) : Json::object();
    Json result{{"schema", 1},
                {"revision", lock.at("revision")},
                {"verified_at", uint64_t(std::time(nullptr))},
                {"files", Json::object()}};
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

Json checkpoint_io::download(const std::filesystem::path& directory, const Json& lock, const std::string& base_url,
                             const std::atomic<bool>& cancel, const ProgressFn& progress, unsigned jobs) {
    check(jobs >= 1 && jobs <= MaxFetchJobs, "concurrent jobs must be between 1 and " + std::to_string(MaxFetchJobs));
    check(!cancel.load(), "cancelled");
    const auto revision = lock.at("revision").get<std::string>();
    std::filesystem::create_directories(directory);
    const auto receipt_path = directory / "zerocool-verification.json";
    const auto old = std::filesystem::exists(receipt_path) ? read_json(receipt_path) : Json::object();
    if(!old.empty())
        check(old.value("revision", std::string()) == revision,
              "refusing to replace a different verified artifact; choose its own directory");
    Json receipt{{"schema", 1}, {"revision", revision}, {"files", Json::object()}};
    uint64_t remaining = 0, total = 0, settled = 0;
    for(const auto& entry : lock.at("files"))
        if(!entry.value("optional", false)) total += entry.at("size").get<uint64_t>();
    std::vector<std::unique_ptr<Transfer>> queue;
    for(const auto& entry : lock.at("files")) {
        if(entry.value("optional", false)) continue;
        check(!cancel.load(), "cancelled");
        const auto name = entry.at("path").get<std::string>();
        require_safe_name(name);
        const auto size = entry.at("size").get<uint64_t>();
        const auto digest = entry.at("sha256").get<std::string>();
        const auto target = directory / name;
        if(std::filesystem::exists(target)) {
            check(uint64_t(std::filesystem::file_size(target)) == size,
                  "refusing to replace a different existing artifact file: " + name);
            const Json saved =
                old.contains("files") && old.at("files").contains(name) ? old.at("files").at(name) : Json(nullptr);
            const auto verified = verified_file(target, size, digest, saved, cancel);
            if(!verified.is_null()) {
                receipt["files"][name] = verified;
                settled += size;
                if(progress) progress(FetchProgress{name, settled, total, 0, false, "verified"});
                continue;
            }
        }
        auto transfer = std::make_unique<Transfer>();
        transfer->url = base_url + name;
        transfer->name = name;
        transfer->part = std::filesystem::path(target).concat(".part");
        transfer->target = target;
        transfer->total = size;
        transfer->digest = digest;
        auto have = std::filesystem::exists(transfer->part) ? uint64_t(std::filesystem::file_size(transfer->part)) : 0;
        if(have == size && std::filesystem::exists(transfer->part))
            transfer->complete_part = verified_file(transfer->part, size, digest, nullptr, cancel);
        if(have > size || (have == size && transfer->complete_part.is_null())) {
            std::filesystem::remove(transfer->part);
            have = 0;
        }
        remaining += size - have;
        queue.push_back(std::move(transfer));
    }
    const auto space = std::filesystem::space(directory);
    check(space.available >= remaining + DiskReserve,
          "insufficient disk space for the pinned checkpoint plus a 5GiB reserve");
    // Preserve completed hashes through interruptions, including later files
    // failing. Final verification checks every current fingerprint against pins.
    write_json_atomic(receipt_path, receipt);
    fetch_all(queue, jobs, cancel, progress, receipt, receipt_path, total, settled);
    return checkpoint_io::verify(directory, lock, true, cancel, progress);
}

Json verify_checkpoint(const std::filesystem::path& directory, Artifact artifact, bool cached,
                       const std::atomic<bool>& cancel, const ProgressFn& progress) {
    return checkpoint_io::verify(directory, artifact_lock(artifact), cached, cancel, progress);
}

Json download_checkpoint(const std::filesystem::path& directory, Artifact artifact, const std::atomic<bool>& cancel,
                         const ProgressFn& progress, unsigned jobs) {
    const auto lock = artifact_lock(artifact);
    const auto base = "https://huggingface.co/" + lock.at("repo").get<std::string>() + "/resolve/" +
                      lock.at("revision").get<std::string>() + "/";
    return checkpoint_io::download(directory, lock, base, cancel, progress, jobs);
}

} // namespace zerocool::engine
