#pragma once
#include "qwen/storage.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace freellm::qwen::q3_probe {
// Experimental fixture format only. Runtime checkpoints still require Q4/Q8.
inline constexpr size_t Group=64,GroupBytes=28;
inline uint16_t bf16(float x) {return uint16_t(std::bit_cast<uint32_t>(round_bf16(x))>>16);}
inline float value(uint16_t x) {return std::bit_cast<float>(uint32_t(x)<<16);}
inline unsigned code(std::span<const std::byte> packed,size_t index) {
    if(packed.size()%GroupBytes || index>=packed.size()/GroupBytes*Group) throw std::out_of_range("Q3 code bounds");
    const auto group=index/Group,bit=(index%Group)*3,at=group*GroupBytes+bit/8;
    unsigned word=std::to_integer<unsigned>(packed[at]);
    if(bit%8>5) word|=std::to_integer<unsigned>(packed[at+1])<<8;
    return (word>>(bit%8))&7;
}
inline std::vector<std::byte> quantize(std::span<const float> input) {
    if(input.empty() || input.size()%Group) throw std::invalid_argument("Q3 requires complete groups of 64");
    std::vector<std::byte> out(input.size()/Group*GroupBytes);
    for(size_t start=0;start<input.size();start+=Group) {
        auto row=input.subspan(start,Group);
        for(float x:row) if(!std::isfinite(x)) throw std::invalid_argument("non-finite Q3 source");
        const auto [lo,hi]=std::minmax_element(row.begin(),row.end());
        const auto scale=bf16((*hi-*lo)/7),bias=bf16(*lo);
        const auto s=value(scale),b=value(bias);
        if(!std::isfinite(s) || !std::isfinite(b) || (*hi!=*lo && s==0)) throw std::invalid_argument("unrepresentable Q3 metadata");
        auto* dst=out.data()+start/Group*GroupBytes;
        for(size_t j=0;j<Group;++j) {
            // Deterministic nearest integer, ties to even; independent of fenv.
            const float v=s==0?0:std::clamp((row[j]-b)/s,0.0f,7.0f),floor=std::floor(v);
            const auto q=unsigned(floor)+unsigned(v-floor>.5f || (v-floor==.5f && (unsigned(floor)&1)));
            const auto bit=j*3;dst[bit/8]|=std::byte((q<<(bit%8))&255);
            if(bit%8>5) dst[bit/8+1]|=std::byte(q>>(8-bit%8));
        }
        dst[24]=std::byte(scale&255);dst[25]=std::byte(scale>>8);
        dst[26]=std::byte(bias&255);dst[27]=std::byte(bias>>8);
    }
    return out;
}
inline std::vector<float> decode(std::span<const std::byte> input) {
    if(input.empty() || input.size()%GroupBytes) throw std::invalid_argument("malformed Q3 record");
    std::vector<float> out(input.size()/GroupBytes*Group);
    for(size_t i=0;i<out.size();++i) {
        const auto* p=input.data()+i/Group*GroupBytes;
        const auto s=value(uint16_t(std::to_integer<unsigned>(p[24])|(std::to_integer<unsigned>(p[25])<<8)));
        const auto b=value(uint16_t(std::to_integer<unsigned>(p[26])|(std::to_integer<unsigned>(p[27])<<8)));
        if(!std::isfinite(s)||s<0||!std::isfinite(b)) throw std::invalid_argument("invalid Q3 metadata");
        out[i]=s*float(code(input,i))+b;
    }
    return out;
}
} // namespace freellm::qwen::q3_probe
