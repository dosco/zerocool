Json streamed_embedding_test(const std::filesystem::path& model_path) {
    check(std::getenv("MTL_DEBUG_LAYER") && std::getenv("MTL_SHADER_VALIDATION"),"embedding check requires Metal validation");
    Json report={{"kind","streamed_embedding_test_v1"},{"performance_measurement",false},
        {"before",process_memory()},{"host_before",host_conditions()},{"cases",Json::array()}};
    {
        embedding_rows::Scope scope;Checkpoint cp(model_path,true,Artifact::Mixed);Metal gpu;gpu.budget(GiB);
        embedding_rows::store=std::make_unique<embedding_rows::Store>(cp);auto& rows=*embedding_rows::store;
        auto full=rows.descriptor();
        const auto load=[&](const char* suffix){return cp.load(std::string("model.embed_tokens.")+suffix,[&](uint64_t n){return gpu.allocate(n);});};
        full.weight={load("weight")};full.scales={load("scales")};full.biases={load("biases")};
        std::vector<std::vector<int>> cases={{0,Vocab-1,17,0,71093,17,42,1,0}};
        for(int start:{1000,1128,1256}) {auto& ids=cases.emplace_back();for(int i=0;i<128;++i)ids.push_back(start+i);}
        cases.push_back(cases.front());
        uint32_t evictions_before_submit=0;
        for(uint32_t copies:{1u,4u}) for(const auto& ids:cases) {
            auto expected=gpu.embedding(full,ids,copies),actual=gpu.embedding(rows.descriptor(),ids,copies);
            if(copies==1 && ids.size()==9) {
                for(int token=30000;token<30257;++token)rows.get(token);
                ++evictions_before_submit;
            }
            gpu.finish();check(actual->bytes==expected->bytes && std::memcmp(actual->data,expected->data,actual->bytes)==0,"streamed embedding changed values");
            report["cases"].push_back({{"tokens",ids.size()},{"copies",copies},{"exact",true},
                {"output_sha256",hash(std::span<const std::byte>(actual->data,actual->bytes))}});
        }
        const auto before=rows.stats();bool rejected=false;
        try {const std::array<int,2> bad={0,Vocab};gpu.embedding(rows.descriptor(),bad,1);}catch(const std::exception&){rejected=true;}
        check(rejected && rows.stats()==before,"invalid row request changed cache");
        const auto cache_hash=hash(std::as_bytes(std::span(rows.rows)));const auto saved=rows.tensors[1].offset;
        rows.tensors[1].offset=rows.tensors[1].file->size()-1;rejected=false;
        try {rows.get(20000);}catch(const std::exception&){rejected=true;}
        rows.tensors[1].offset=saved;
        check(rejected && hash(std::as_bytes(std::span(rows.rows)))==cache_hash && rows.next<embedding_rows::Capacity &&
            rows.stats()["misses"]==before["misses"] && rows.stats()["evictions"]==before["evictions"],"failed read published partial row");
        check(rows.hits>0 && rows.evictions>0 && rows.misses>embedding_rows::Capacity,"embedding cache transitions unexercised");
        report["cache"]=rows.stats();report["invalid_request_atomic"]=true;report["failed_read_atomic"]=true;
        report["evictions_before_gpu_submission"]=evictions_before_submit;report["peak_gpu_bytes"]=gpu.peak();
    }
    check(!embedding_rows::enabled && !embedding_rows::store,"embedding owner leaked");
    report["owner_released"]=true;report["after"]=process_memory();report["host_after"]=host_conditions();report["complete"]=true;return report;
}
