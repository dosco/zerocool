#pragma once
#include "engine/storage.hpp"
#include <sstream>

namespace zerocool::engine {
// Validate completed coordinator records, not the watcher's intended timing.
inline Json recovery_trace_evidence(const std::string& text,size_t target,size_t observed,
                                    const std::string& build,const std::string& artifact) {
    if((target!=1 && target!=97) || text.empty() || text.back()!='\n')
        throw std::invalid_argument("incomplete recovery trace");
    std::istringstream input(text);std::string line;size_t count=0;
    while(std::getline(input,line)) {
        const auto row=Json::parse(line);
        if(row.at("layer")!=count%48 || row.at("tokens")!=1 || row.at("offset")!=(count<48?0:1) ||
           row.at("build")!=build || row.at("artifact_revision")!=artifact)
            throw std::invalid_argument("recovery trace sequence or identity differs");
        ++count;
    }
    const size_t last=target==1?47:143;
    if(count<target || count>last || observed<target || observed>count)
        throw std::invalid_argument("missed recovery cancellation window");
    return {{"target_records",target},{"observed_records",observed},{"completed_records",count},
            {"window",target==1?"reference_setup":"warmup_control"},{"validated",true}};
}
} // namespace zerocool::engine
