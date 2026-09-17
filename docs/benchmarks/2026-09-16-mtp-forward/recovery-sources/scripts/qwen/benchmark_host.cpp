// Read benchmark prerequisites without constructing Metal or loading a model.
#include "qwen/metal.hpp"
#include <fstream>
#include <iostream>
using namespace freellm::qwen;
int main(int argc,char** argv) {
    try {
        if(argc!=2 || std::filesystem::exists(argv[1]))
            throw std::runtime_error("usage: benchmark-host NEW_REPORT");
        const Json report={{"kind","benchmark_host_preflight_v1"},{"complete",true},
            {"build_fingerprint",build_fingerprint()},{"host",host_conditions()},
            {"process",process_memory()},{"model_loaded",false},{"gpu_used",false}};
        std::ofstream out(argv[1]);out<<report.dump(2)<<'\n';
        if(!out) throw std::runtime_error("cannot write host preflight");
        std::cout<<report.at("host").dump()<<'\n';return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 2;}
}
