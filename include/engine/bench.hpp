#pragma once
#include "engine/model.hpp"
namespace zerocool::engine {
std::vector<Json> read_route_trace(const std::filesystem::path& path, Artifact artifact = Artifact::Q4);
Json dependency_bench(const Options& options, const std::filesystem::path& routes,
                      int repetitions, int hit_count);
Json kernel_bench(const Options& options,int repetitions);
Json fixture_kernel_bench(const Options& options,int repetitions);
}
