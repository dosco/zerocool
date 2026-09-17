// Source-copy replacement of Metal::capture_linear. Capture six tiny inputs,
// hashing existing GPU-visible weight bytes without allocating weight copies.
void Metal::capture_linear(const Linear& l,const Linear* up,const Buf& x,uint32_t tokens,const Buf& rows) {
    auto& p=*impl_;const auto& dir=p.config.operator_capture;
    if(dir.empty() || !l.quantized || l.bits!=8 || l.group!=64 || up || rows || tokens!=4 ||
       p.request_phase!="decode" || p.context.value("offset",0u)!=72) return;
    const auto stage=p.context.value("stage","");const auto layer=p.context.value("layer",-2);
    const bool wanted=(stage=="gdn" && layer==0 &&
        ((l.input==2560 && (l.output==10240 || l.output==6144)) || (l.input==6144 && l.output==2560))) ||
        (stage=="attention" && layer==3 && ((l.input==2560 && l.output==12288) || (l.input==6144 && l.output==2560))) ||
        (stage=="logits" && layer==-1 && l.input==2560 && l.output==248320);
    if(!wanted) return;
    Json matrix={{"K",l.input},{"N",l.output},{"rows",tokens},{"group",l.group},{"bits",l.bits},
        {"fused",false},{"gathered",false}};
    const auto key=stage+"/"+std::to_string(layer)+"/"+matrix.dump();
    if(p.captured_shapes.contains(key)) return;
    const uint64_t input_bytes=uint64_t(tokens)*l.input*4;
    if(p.capture.size()>=6 || x->bytes!=input_bytes || input_bytes>MiB-p.captured_bytes)
        throw std::runtime_error("expanded capture exceeded input bound");
    finish();
    if(p.capture.empty()) {
        if(std::filesystem::exists(dir) && !std::filesystem::is_empty(dir)) throw std::runtime_error("expanded input directory must be new");
        std::filesystem::create_directories(dir);
    }
    auto digest=[](const Binding& binding,uint64_t bytes) {
        if(!binding.buffer || binding.offset>binding.buffer->bytes || bytes>binding.buffer->bytes-binding.offset || bytes>UINT32_MAX)
            throw std::runtime_error("expanded capture tensor bounds");
        unsigned char hash[CC_SHA256_DIGEST_LENGTH];CC_SHA256(binding.buffer->data+binding.offset,CC_LONG(bytes),hash);
        std::string hex;constexpr char digits[]="0123456789abcdef";
        for(auto c:hash){hex+=digits[c>>4];hex+=digits[c&15];}return hex;
    };
    const auto filename=std::to_string(p.capture.size())+"-x.bin";
    std::ofstream stream(dir/filename,std::ios::binary);
    stream.write(reinterpret_cast<const char*>(x->data),std::streamsize(input_bytes));stream.close();
    if(!stream) throw std::runtime_error("cannot save expanded input");
    const uint64_t weight_bytes=uint64_t(l.input)*l.output,meta_bytes=weight_bytes/64*2;
    Json tensors={{"w",{{"bytes",weight_bytes},{"sha256",digest(l.weight,weight_bytes)}}},
        {"s",{{"bytes",meta_bytes},{"sha256",digest(l.scales,meta_bytes)}}},
        {"b",{{"bytes",meta_bytes},{"sha256",digest(l.biases,meta_bytes)}}},
        {"x",{{"file",filename},{"bytes",input_bytes},{"sha256",digest({x},input_bytes)}}}};
    p.capture.push_back({{"matrix",matrix},{"phase",p.request_phase},{"context",p.context},{"tensors",tensors}});
    p.captured_shapes.insert(key);p.captured_bytes+=input_bytes;
    Json manifest={{"kind","q8_expanded_inputs_v1"},{"build_fingerprint",BuildFingerprint},
        {"artifact_revision",p.config.artifact_revision},{"bytes",p.captured_bytes},{"byte_limit",MiB},{"case_limit",6},
        {"weight_payloads_copied",false},{"normal_request_latency_qualified",false},{"cases",p.capture}};
    std::ofstream file(dir/"manifest.json");file<<manifest.dump(2)<<'\n';
    if(!file) throw std::runtime_error("cannot save expanded input manifest");
}
