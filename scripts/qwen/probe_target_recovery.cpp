// Developer-only saved-state fixture. Payloads never enter the production binary.
constexpr uint64_t FixtureLimit=2*GiB;
// A fixture is larger than a gigabyte. Do not leave its writes in the filesystem
// cache alongside the live model; the reader already uses uncached I/O.
class FixtureWriter {
    int fd_=-1;
public:
    explicit FixtureWriter(const std::filesystem::path& path) {
        fd_=::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if(fd_<0) throw std::system_error(errno,std::generic_category(),"create fixture payload");
        if(::fcntl(fd_,F_NOCACHE,1)!=0) {
            const int error=errno;::close(fd_);fd_=-1;
            throw std::system_error(error,std::generic_category(),"uncached fixture writes");
        }
    }
    FixtureWriter(const FixtureWriter&)=delete;
    ~FixtureWriter(){if(fd_>=0)::close(fd_);}
    void write(std::span<const std::byte> bytes) {
        check(fd_>=0,"fixture writer is closed");
        while(!bytes.empty()) {
            const auto n=::write(fd_,bytes.data(),std::min<size_t>(bytes.size(),32*MiB));
            if(n<0 && errno==EINTR)continue;
            if(n<0)throw std::system_error(errno,std::generic_category(),"write fixture payload");
            check(n>0,"short fixture write");bytes=bytes.subspan(size_t(n));
        }
    }
    void close() {
        check(fd_>=0,"fixture writer is closed");const int fd=fd_;fd_=-1;
        if(::close(fd)!=0)throw std::system_error(errno,std::generic_category(),"close fixture payload");
    }
};
void capture_memory(Json& report,const char* event,uint32_t prefix=0) {
    auto& samples=report["capture_memory"];
    if(samples.is_null())samples=Json::array();
    check(samples.size()<12,"too many capture memory boundaries");
    samples.push_back({{"event",event},{"prefix",prefix},{"monotonic_ns",monotonic_ns()},{"process",process_memory()}});
}
std::string stream_hash(const std::filesystem::path& path) {
    File file(path);check(file.size()<=FixtureLimit,"fixture payload exceeds 2GiB");
    CC_SHA256_CTX context;CC_SHA256_Init(&context);std::array<std::byte,65536> bytes;
    for(uint64_t at=0;at<file.size();) {
        auto n=std::min<uint64_t>(bytes.size(),file.size()-at);file.read(at,{bytes.data(),size_t(n)});
        CC_SHA256_Update(&context,bytes.data(),CC_LONG(n));at+=n;
    }
    unsigned char out[32];CC_SHA256_Final(out,&context);std::string result;
    for(auto c:out){result+="0123456789abcdef"[c>>4];result+="0123456789abcdef"[c&15];}return result;
}
uint64_t fixture_geometry(int layer,int field) {
    if(field==5) return layer==1?9ull*Hyper*4:0;
    if((layer+1)%4) return field==0?3*10240*4:field==1?48*128*128*4:0;
    return field>=2 && field<=4?8192ull*(field==4?128:512)*4:0;
}
struct RecoveryBundle {
    std::filesystem::path path;std::unique_ptr<FixtureWriter> payload;uint64_t bytes=0;
    Json manifest={{"kind","target_recovery_fixture_v1"},{"complete",false},{"limit_bytes",FixtureLimit},
        {"reference_origin","full-target-forward"},
        {"artifact_revision",artifact_revision(Artifact::Mixed)},{"layers",Layers},{"context",8192},
        {"arithmetic","original-gdn_scan-conv_update-v1"},{"expected",Json::array()}};
    explicit RecoveryBundle(const std::filesystem::path& p):path(p) {
        check(!std::filesystem::exists(p),"fixture directory exists");std::filesystem::create_directories(p);
        payload=std::make_unique<FixtureWriter>(p/"payload.bin");
    }
    Json buffer(const Buf& b) {
        if(!b)return nullptr;check(bytes+b->bytes+MiB<=FixtureLimit,"fixture bundle exceeds 2GiB");
        Json record={{"offset",bytes},{"bytes",b->bytes},{"sha256",hash({b->data,size_t(b->bytes)})}};
        payload->write({b->data,size_t(b->bytes)});
        bytes+=b->bytes;return record;
    }
    void state(const char* name,const State& state) {
        Json rows=Json::array();
        for(int l=0;l<Layers;++l) {
            Json buffers=Json::array();
            for(int f=0;f<6;++f){const auto& b=state.layers[l].*fields[f];
                check((b?b->bytes:0)==fixture_geometry(l,f),"fixture state geometry differs");buffers.push_back(buffer(b));}
            rows.push_back({{"position",state.layers[l].position},{"buffers",buffers}});
        }
        manifest[name]={{"layers",rows},{"tokens",state.tokens},{"history",state.history},
            {"trace_session_id",state.trace_session_id},{"valid",state.valid},{"digest",state_digest(state)}};
    }
    void journal(const mtp_recovery::Journal& journal) {
        Json entries=Json::array();
        for(int l=0;l<Layers;++l) if((l+1)%4) {
            const auto& e=journal.entries[l];entries.push_back({{"layer",l},{"ad",e.ad},{"dd",e.dd},
                {"convolution",buffer(e.convolution)},{"normalized",buffer(e.normalized)},
                {"a",buffer(e.a)},{"b",buffer(e.b)},{"alog",buffer(e.alog)},{"dt",buffer(e.dt)}});
        }
        manifest["journal"]={{"entries",entries},{"ple",buffer(journal.ple)},
            {"offset",journal.offset},{"ids",journal.ids},{"accounting",journal.stats()}};
    }
    void finish() {
        payload->close();manifest["payload_bytes"]=bytes;manifest["payload_sha256"]=stream_hash(path/"payload.bin");
        manifest["complete"]=true;check(manifest.dump().size()<MiB,"fixture manifest exceeds bound");
        save_report(path/"manifest.json",manifest);
    }
};
void capture_recovery_fixture(RecoveryBundle& bundle,Model& model,State& state,CheckpointCopy& checkpoint,
        mtp_recovery::Journal& journal,std::span<const int> ids,Json& report,const std::filesystem::path& output) {
    model.diagnostic_drain();journal.validate(state,4);capture_memory(report,"verified_before_write");
    bundle.state("verified",state);capture_memory(report,"verified_written");
    bundle.journal(journal);capture_memory(report,"journal_written");
    bundle.manifest["source"]={{"request_id",report.at("request_id")},{"draft_manifest_sha256",report.at("draft_manifest_sha256")},
        {"input_sha256",report.at("input_sha256")},
        {"producer_binary_sha256",report.at("producer_binary_sha256")},
        {"native_build_fingerprint",report.at("before").at("metal").at("build_fingerprint")},
        {"kernel_policy",report.at("before").at("metal").at("kernels")},
        {"target_prepared_sha256",report.at("before").at("prepared").at("manifest_sha256")}};
    // Golden states use complete target forwards, without calling Journal::apply.
    for(uint32_t keep=1;keep<=4;++keep) {
        phase(output,"capture_reference_prefix",keep);
        check(!stopped,"capture cancelled");checkpoint.restore(state);Json logits=Json::array();
        for(uint32_t i=0;i<keep;++i) {
            model.phase("decode");auto row=model.forward(ids.subspan(i,1),state,true,&stopped);logits.push_back(row_hash(row));
        }
        model.diagnostic_drain();bundle.manifest["expected"].push_back({{"keep",keep},{"state",state_digest(state)},
            {"trace_session_id",state.trace_session_id},{"row_logits_sha256",logits}});
        capture_memory(report,"reference_prefix_complete",keep);
    }
    bundle.finish();capture_memory(report,"payload_closed_and_hashed");
    report["kind"]="target_recovery_capture_v1";report["performance_measurement"]=false;
    report["capture"]={{"manifest",(bundle.path/"manifest.json").string()},{"sha256",hash_file(bundle.path/"manifest.json")},
        {"payload_bytes",bundle.bytes},{"independent_full_target_prefixes",{1,2,3,4}},{"fixture_limit_bytes",FixtureLimit}};
    report["after"]=model.stats();
}
struct FixtureReader {
    Json manifest;File file;uint64_t cursor=0;
    static uint32_t integer(const Json& value,uint32_t maximum) {
        check(value.is_number_unsigned() && value.get<uint64_t>()<=maximum,"invalid fixture integer");
        return value.get<uint32_t>();
    }
    explicit FixtureReader(const std::filesystem::path& path,const char* origin="full-target-forward"):file(path/"payload.bin") {
        check(std::filesystem::file_size(path/"manifest.json")<MiB,"fixture manifest too large");manifest=read_json(path/"manifest.json");
        check(manifest.at("kind")=="target_recovery_fixture_v1" && manifest.at("complete")==true &&
            manifest.at("reference_origin")==origin &&
            manifest.at("limit_bytes")==FixtureLimit && manifest.at("artifact_revision")==artifact_revision(Artifact::Mixed) &&
            manifest.at("layers")==Layers && manifest.at("context")==8192 &&
            manifest.at("arithmetic")=="original-gdn_scan-conv_update-v1" &&
            manifest.at("payload_bytes")==file.size() && file.size()+MiB<=FixtureLimit &&
            manifest.at("payload_sha256")==stream_hash(path/"payload.bin"),"corrupt or incompatible recovery fixture");
        // Validate every tensor range and shape before allocating or writing state.
        for(auto name:{"before","verified"}) {
            const auto& s=manifest.at(name);check(s.at("layers").size()==Layers && s.at("valid")==true,"invalid fixture state");
            const uint32_t tokens=integer(s.at("tokens"),8192);
            check(s.at("history").size()==2 && s.at("trace_session_id").is_number_unsigned(),"invalid fixture history");
            for(const auto& id:s.at("history"))integer(id,Vocab-1);
            for(int l=0;l<Layers;++l) {
                const auto& row=s.at("layers").at(l);check(row.at("position")==tokens && row.at("buffers").size()==6,"fixture positions differ");
                for(int f=0;f<6;++f) validate(row.at("buffers").at(f),fixture_geometry(l,f));
            }
        }
        const auto& j=manifest.at("journal");const uint32_t offset=integer(j.at("offset"),8188);
        check(j.at("ids").size()==4 && offset<=8188 && manifest["before"]["tokens"]==offset &&
            manifest["verified"]["tokens"]==offset+4 && j.at("entries").size()==36,"fixture journal coverage differs");
        for(const auto& id:j.at("ids"))integer(id,Vocab-1);
        int row=0;
        for(int l=0;l<Layers;++l) if((l+1)%4) {
            const auto& e=j.at("entries").at(row++);const uint32_t ad=integer(e.at("ad"),2),dd=integer(e.at("dd"),2);
            check(e.at("layer")==l && ad<=2 && dd<=2,"invalid journal layer or dtype");
            for(auto k:{"convolution","normalized"})validate(e.at(k),4*10240*4);
            for(auto k:{"a","b"})validate(e.at(k),4*48*4);
            validate(e.at("alog"),48*(ad==1?4:2));validate(e.at("dt"),48*(dd==1?4:2));
        }
        validate(j.at("ple"),4*Hyper*4);check(cursor==file.size(),"fixture has unindexed payload");
        check(manifest.at("expected").size()==4,"missing reference prefixes");
        for(int i=0;i<4;++i)check(manifest["expected"][i]["keep"]==i+1 &&
            manifest["expected"][i]["state"]["tokens"]==offset+i+1,"invalid reference prefix");
    }
    void validate(const Json& r,uint64_t bytes) {
        if(!bytes){check(r.is_null(),"unexpected fixture tensor");return;}
        check(r.at("offset").is_number_unsigned() && r.at("bytes").is_number_unsigned() &&
            r.at("offset")==cursor && r.at("bytes")==bytes && cursor+bytes<=file.size() &&
            r.at("sha256").is_string() && r.at("sha256").get<std::string>().size()==64,"invalid fixture tensor range");cursor+=bytes;
    }
    void load(const Json& record,const Buf& b) {
        check(b && record.at("bytes")==b->bytes,"fixture destination differs");
        file.read(record.at("offset"),{b->data,size_t(b->bytes)});
        check(record.at("sha256")==hash({b->data,size_t(b->bytes)}),"fixture tensor hash differs");
    }
    void state(Metal& gpu,const char* name,State& state) {
        const auto& source=manifest.at(name);state.valid=false;state.artifact=Artifact::Mixed;
        for(int l=0;l<Layers;++l) {
            for(int f=0;f<6;++f) if(auto n=fixture_geometry(l,f)) {
                auto& b=state.layers[l].*fields[f];if(!b)b=gpu.allocate(n,AllocationClass::State);
                load(source.at("layers").at(l).at("buffers").at(f),b);
            }
            state.layers[l].position=source.at("layers").at(l).at("position");
        }
        state.tokens=source.at("tokens");state.history=source.at("history").get<std::array<int,2>>();
        state.trace_session_id=source.at("trace_session_id");state.valid=true;
        check(state_digest(state)==source.at("digest"),"fixture state digest differs");
    }
    void journal(Metal& gpu,mtp_recovery::Journal& journal) {
        const auto& source=manifest.at("journal");const auto ids=source.at("ids").get<std::vector<int>>();
        journal.begin(ids,source.at("offset"));
        for(const auto& row:source.at("entries")) {
            auto& e=journal.entries.at(row.at("layer").get<size_t>());e.ad=row.at("ad");e.dd=row.at("dd");
            load(row.at("convolution"),e.convolution);load(row.at("normalized"),e.normalized);load(row.at("a"),e.a);load(row.at("b"),e.b);
            e.alog=gpu.allocate(row.at("alog").at("bytes"),AllocationClass::Resident);load(row.at("alog"),e.alog);
            e.dt=gpu.allocate(row.at("dt").at("bytes"),AllocationClass::Resident);load(row.at("dt"),e.dt);e.seen=true;
        }
        load(source.at("ple"),journal.ple);journal.ple_seen=true;journal.finish();
    }
};
Json replay_recovery_fixture(const std::filesystem::path& path,const std::string& producer_sha256,
        const char* origin="full-target-forward") {
    check(std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"),"recovery fixture requires Metal validation");
    const auto before=process_memory(),host_before=host_conditions();FixtureReader reader(path,origin);
    check(reader.manifest.at("source").at("producer_binary_sha256")==producer_sha256,"fixture producer differs");
    Metal gpu;gpu.budget(FixtureLimit);
    KernelConfig config;gpu.configure(config);gpu.prepare_pipelines();Json cases=Json::array(),accounting;
    uint64_t peak=0;
    {
        State state;reader.state(gpu,"before",state);CheckpointCopy checkpoint(state);checkpoint.save(state,4);
        mtp_recovery::Journal journal(gpu);reader.journal(gpu,journal);
        auto corrupt=reader.manifest["journal"]["entries"][0]["a"];corrupt["sha256"]=std::string(64,'0');bool corrupted=false;
        try{reader.load(corrupt,journal.entries[0].a);}catch(const std::exception&){corrupted=true;}
        check(corrupted,"corrupt tensor hash accepted");
        auto reset=[&]{gpu.finish();reader.state(gpu,"verified",state);};
        auto run=[&](uint32_t keep,const std::atomic<bool>* cancel=nullptr){
            journal.validate(state,keep);checkpoint.check_prefix(state,keep);mtp_recovery::cancelled(cancel);
            if(keep<4){checkpoint.restore_prefix(state,keep);journal.apply(gpu,state,keep,cancel);checkpoint.commit_prefix(state,std::span(journal.ids).first(keep));}
        };
        for(int repetition=0;repetition<2;++repetition) for(uint32_t keep=1;keep<=4;++keep) {
            reset();run(keep);const auto& expected=reader.manifest.at("expected").at(keep-1);
            check(state_digest(state)==expected.at("state") && state.trace_session_id==expected.at("trace_session_id"),"recovery differs from full-target prefix state");
            cases.push_back({{"keep",keep},{"repetition",repetition},{"every_buffer_exact",true},{"state",state_digest(state)}});
            peak=std::max(peak,process_memory().at("physical_footprint_peak_bytes").get<uint64_t>());check(peak<=FixtureLimit,"replay process exceeds 2GiB");
        }
        // Bad late geometry/coverage must leave earlier state and all metadata intact.
        for(int test=0;test<3;++test) {
            reset();auto old=state.layers[47].index;const bool seen=journal.entries[46].seen;
            if(test==0)state.layers[47].index=Buffer::host(16);if(test==1)journal.entries[46].seen=false;
            const auto initial=state_digest(state);bool failed=false;
            try{run(test==2?0:2);}catch(const std::exception&){failed=true;}
            check(failed && state_digest(state)==initial,"invalid recovery wrote state");state.layers[47].index=old;journal.entries[46].seen=seen;
        }
        reset();const auto initial=state_digest(state);std::atomic<bool> cancel=true;bool failed=false;
        try{run(2,&cancel);}catch(const std::exception&){failed=true;}
        check(failed && state_digest(state)==initial,"pre-cancelled recovery wrote state");
        reset();journal.test_fail_after_layer=5;failed=false;
        try{run(2);}catch(const std::exception&){failed=true;}
        check(failed && !state.valid && gpu.statistics().at("live_command_groups")==0,"failed recovery did not invalidate and drain");journal.test_fail_after_layer=-1;
        // Delay actual completion callbacks, then exercise cancellation with work in flight.
        for(bool interrupt:{false,true}) {
            reset();cancel=false;mtp_scratch::held_callbacks=0;mtp_scratch::hold_completions=true;
            std::thread releaser([&]{
                const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(2);
                while(!mtp_scratch::held_callbacks && std::chrono::steady_clock::now()<end)std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(30));cancel=interrupt;mtp_scratch::hold_completions=false;
            });
            struct Join {std::thread& t;~Join(){mtp_scratch::hold_completions=false;if(t.joinable())t.join();}} join{releaser};
            failed=false;try{run(3,&cancel);}catch(const std::exception&){failed=true;}releaser.join();
            check(mtp_scratch::held_callbacks>0 && failed==interrupt && gpu.statistics().at("live_command_groups")==0,"delayed completion handling differs");
            if(interrupt)check(!state.valid,"cancelled partial state remained valid");
            else check(state_digest(state)==reader.manifest["expected"][2]["state"],"delayed recovery state differs");
        }
        accounting=journal.stats();check(journal.peak_groups<=2,"too many recovery groups");
    }
    gpu.finish();check(gpu.allocated()==0,"recovery buffers retained after cleanup");const auto after=process_memory();
    check(after.at("physical_footprint_peak_bytes").get<uint64_t>()<=FixtureLimit,"replay process peak exceeds 2GiB");
    return {{"kind","target_recovery_replay_v1"},{"complete",true},{"performance_measurement",false},
        {"reference_origin",origin},{"host_before",host_before},{"host_after",host_conditions()},
        {"source_manifest_sha256",hash_file(path/"manifest.json")},{"cases",cases},{"journal",accounting},
        {"invalid_geometry_atomic",true},{"missing_coverage_atomic",true},{"invalid_prefix_atomic",true},
        {"corrupt_tensor_rejected",true},
        {"cancellation_drained",true},{"failure_drained",true},{"delayed_completion_safe",true},{"all_buffers_released",true},
        {"full_model_loaded",false},{"before",before},{"after",after},{"process_limit_bytes",FixtureLimit}};
}

// A low-memory operator/transaction check, explicitly synthetic. This is useful
// before real capture is admitted, and cannot satisfy the real-model gate.
Json recovery_self_test(const std::filesystem::path& fixture={},const std::string& producer_sha256={}) {
    check(std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"),"recovery self-test needs Metal validation");
    for(const auto& value:Json::array({uint64_t(1)<<32,-1,1.5,true,"1"})) {
        bool rejected=false;try{FixtureReader::integer(value,8192);}catch(const std::exception&){rejected=true;}
        check(rejected,"invalid fixture integer accepted or narrowed");
    }
    const auto before=process_memory(),host_before=host_conditions();
    Metal gpu;gpu.budget(FixtureLimit);gpu.prepare_pipelines();Json cases=Json::array();uint64_t peak=0;
    {
        State state;state.artifact=Artifact::Mixed;
        for(int l=0;l<Layers;++l)for(int f=0;f<6;++f)if(auto n=fixture_geometry(l,f))state.layers[l].*fields[f]=gpu.zeros(n/4,AllocationClass::State);
        mtp_recovery::Journal journal(gpu);const std::array<int,4> ids={17,248046,23,248044};
        for(uint32_t offset:{3u,8188u}) {
            state.tokens=offset;state.history={248044,248046};state.trace_session_id=91;state.valid=true;
            for(auto& layer:state.layers)layer.position=offset;
            std::unique_ptr<RecoveryBundle> bundle;
            if(offset==3 && !fixture.empty()) {
                bundle=std::make_unique<RecoveryBundle>(fixture);bundle->manifest["reference_origin"]="synthetic-operators";
                bundle->manifest["source"]={{"producer_binary_sha256",producer_sha256}};bundle->state("before",state);
            }
            CheckpointCopy checkpoint(state);checkpoint.save(state,4);journal.begin(ids,offset);
            auto fill=[](const Buf& b,int seed){for(size_t i=0;i<b->floats().size();++i)b->floats()[i]=float(int((i+seed)%19)-9)/128;};
            for(int l=0;l<Layers;++l)if((l+1)%4) {
                auto& e=journal.entries[l];fill(e.convolution,l);fill(e.normalized,l+1);fill(e.a,l+2);fill(e.b,l+3);
                e.ad=e.dd=1;e.alog=gpu.zeros(48,AllocationClass::Resident);e.dt=gpu.zeros(48,AllocationClass::Resident);e.seen=true;
            }
            fill(journal.ple,7);journal.ple_seen=true;journal.finish();Json expected=Json::array();
            // Reference processes one row at a time through the existing public
            // operators. It never calls either candidate restore_prefix/apply.
            for(uint32_t keep=1;keep<=4;++keep) {
                checkpoint.restore(state);
                for(uint32_t row=0;row<keep;++row) {
                    auto p=gpu.slice(journal.ple,uint64_t(row)*Hyper*4,Hyper*4);
                    auto ple=gpu.allocate(9*Hyper*4,AllocationClass::State);
                    gpu.dispatch("conv_update",{{p},{state.layers[1].ple_conv},{ple}},{Hyper,1,9},9*Hyper);state.layers[1].ple_conv=ple;
                    for(int l=0;l<Layers;++l)if((l+1)%4) {
                        auto& s=state.layers[l];const auto& e=journal.entries[l];
                        auto c=gpu.slice(e.convolution,uint64_t(row)*10240*4,10240*4);
                        auto n=gpu.slice(e.normalized,uint64_t(row)*10240*4,10240*4);
                        auto a=gpu.slice(e.a,row*48*4,48*4),b=gpu.slice(e.b,row*48*4,48*4);
                        auto conv=gpu.allocate(3*10240*4,AllocationClass::State);
                        gpu.dispatch("conv_update",{{c},{s.conv},{conv}},{10240,1,3},3*10240);s.conv=conv;
                        auto y=gpu.gdn_scan(n,a,b,e.alog,e.dt,s.recurrence,1,e.ad,e.dd);gpu.finish();
                    } else for(int f=2;f<=4;++f) {
                        auto& b=state.layers[l].*fields[f];const uint64_t stride=(f==4?128:512)*4;
                        std::memset(b->data+(offset+row)*stride,int(31+row+l),stride);
                    }
                    state.history={state.history[1],ids[row]};
                }
                gpu.finish();for(auto& layer:state.layers)layer.position=offset+keep;state.tokens=offset+keep;
                expected.push_back(state_digest(state));
                if(bundle)bundle->manifest["expected"].push_back({{"keep",keep},{"state",state_digest(state)},
                    {"trace_session_id",state.trace_session_id},{"row_logits_sha256",Json::array()}});
            }
            if(bundle){bundle->state("verified",state);bundle->journal(journal);bundle->finish();}
            auto post=snapshot_state(gpu,state);
            for(uint32_t keep=1;keep<=4;++keep) {
                restore_state(gpu,post,state);journal.validate(state,keep);checkpoint.check_prefix(state,keep);
                if(keep<4){checkpoint.restore_prefix(state,keep);journal.apply(gpu,state,keep);checkpoint.commit_prefix(state,std::span(ids).first(keep));}
                check(state_digest(state)==expected.at(keep-1),"synthetic state recovery differs from row-at-a-time reference");
                cases.push_back({{"offset",offset},{"keep",keep},{"all_persistent_buffers_exact",true}});
                peak=std::max(peak,process_memory().at("physical_footprint_peak_bytes").get<uint64_t>());check(peak<=FixtureLimit,"self-test exceeds 2GiB");
            }
        }
    }
    gpu.finish();check(gpu.allocated()==0,"synthetic recovery users leaked");
    Json serialized=nullptr;
    if(!fixture.empty()) {
        bool rejected=false;try{FixtureReader wrong_origin(fixture);}catch(const std::exception&){rejected=true;}
        check(rejected,"synthetic fixture accepted as real target evidence");
        serialized=replay_recovery_fixture(fixture,producer_sha256,"synthetic-operators");
    }
    return {{"kind","target_recovery_synthetic_v1"},{"complete",true},{"real_model_evidence",false},
        {"before",before},{"after",process_memory()},{"host_before",host_before},{"host_after",host_conditions()},
        {"invalid_integers_rejected",true},
        {"serialized_replay",serialized},{"performance_measurement",false},{"cases",cases},
        {"peak_physical_bytes",process_memory().at("physical_footprint_peak_bytes")},{"all_buffers_released",true}};
}
