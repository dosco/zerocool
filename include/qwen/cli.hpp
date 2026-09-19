#pragma once
#include "qwen/session.hpp"

namespace freellm::qwen {
// The command line is parsed and validated apart from execution so the cross
// option rules stay checkable without a model, a GPU or a benchmark run.
struct Cli {
    std::string command;
    Options options;
    std::string prompt,json_path,io_path,tokens_path,logits_path,tokenize,render_path,workloads_path;
    std::string replay_routes,phase_profile,cached_progress,bench_progress_path;
    int replay_hits=-1,soak_seconds=0,repetitions=3,port=8080,control_fd=-1;
    bool probe=false,storage=false,kernel_probe=false,raw=false,thinking=false;
};
// argv[1] selects the command; options follow. Both throw std::invalid_argument
// with a message naming the option at fault.
Cli parse_cli(int argc,char** argv);
void validate_cli(Cli& cli);
} // namespace freellm::qwen
