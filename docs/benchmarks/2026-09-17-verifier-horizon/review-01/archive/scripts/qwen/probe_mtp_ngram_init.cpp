Json ngram_init_test(const std::filesystem::path& model,const std::filesystem::path& prepared) {
    Checkpoint cp(model,true,Artifact::Mixed);ReadPool reads(4);
    auto artifact=std::make_shared<PreparedArtifact>(prepared,cp);Json cases=Json::array();
    // A small ring forces many wraparounds; full capacity checks that unused
    // payload stays unconstructed. Both stores read the same real packed tables.
    for(uint64_t budget:{64*1024ull,64*MiB}) {
        check(!setenv("FREELLM_MTP_NGRAM_INIT","eager",1),"cannot select eager fixture");
        NgramStore eager(cp,reads,budget,artifact);
        check(!setenv("FREELLM_MTP_NGRAM_INIT","lazy",1),"cannot select lazy fixture");
        NgramStore lazy(cp,reads,budget,artifact);
        const auto initial=NgramAudit::summary(lazy);check(initial.at("constructed_rows")==0,"lazy constructor touched rows");
        check(initial.at("capacity_rows")==NgramAudit::summary(eager).at("capacity_rows"),"cache capacities differ");
        std::array<int,2> history={248044,248044};uint64_t tokens=0;Json boundaries=Json::array();
        for(uint32_t step=0;step<18;++step) {
            const uint32_t n=std::array<uint32_t,6>{1,2,3,7,11,17}[step%6];std::vector<int> ids(n);
            for(uint32_t i=0;i<n;++i) ids[i]=(step%3==0)?77091:int(100+step*97+i*29);
            if(step%4==0) ids[n/2]=248044;
            std::vector<float> a(uint64_t(n)*Hidden),b(a.size());
            check(eager.row_ids(ids,history)==lazy.row_ids(ids,history),"ngram addresses changed");
            eager.embedding(ids,history,a);lazy.embedding(ids,history,b);
            check(std::memcmp(a.data(),b.data(),a.size()*4)==0 && NgramAudit::same(eager,lazy),"ngram bytes, replacement or hit counts changed");
            // An exact repeat also checks live hits and duplicate-row fan-out.
            eager.embedding(ids,history,a);lazy.embedding(ids,history,b);
            check(std::memcmp(a.data(),b.data(),a.size()*4)==0 && NgramAudit::same(eager,lazy),"ngram repeat changed cache behavior");
            for(int id:ids) history={history[1],id};tokens+=n;
            boundaries.push_back({{"step",step},{"lazy",NgramAudit::summary(lazy)},{"output_sha256",row_hash(a)}});
        }
        const auto final=NgramAudit::summary(lazy);
        if(budget==64*1024ull) check(final.at("constructed_rows")==final.at("capacity_rows") && lazy.misses>initial.at("capacity_rows").get<uint64_t>()*2,"missing ring wraparound");
        else check(final.at("constructed_rows")==final.at("cached_rows") && final.at("initialized_row_bytes").get<uint64_t>()<MiB,"unused full cache rows initialized");
        cases.push_back({{"budget_bytes",budget},{"tokens",tokens},{"initial",initial},{"final",final},{"boundaries",boundaries}});
    }
    return {{"kind","mtp_ngram_init_fixture_v1"},{"complete",true},{"exact_outputs_and_cache",true},
        {"duplicate_rows_and_hits",true},{"eos_history",true},{"ring_wraparound",true},{"cases",cases},
        {"model_loaded",false},{"gpu_used",false},{"production_promoted",false}};
}
