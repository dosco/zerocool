#!/usr/bin/env python3
"""Isolated exact target-state recovery; prior producers remain unchanged."""
import argparse
import json
from pathlib import Path
import subprocess

import build_mtp_direct_output as base
from build_block_cache_trace import replace
from build_identity import build_fingerprint
from cache_residency import require
from qualification_evidence import save, sha

ROOT=base.ROOT
HEADER=ROOT/'scripts/qwen/mtp_target_recovery.hpp'
FIXTURE=ROOT/'scripts/qwen/probe_target_recovery.cpp'
CHECKPOINT=r'''
    void check_prefix(const State& state,uint32_t keep) const {
        check(valid_ && state.artifact==artifact_ && keep>=1 && keep<=4 && state.tokens==tokens_+4,"invalid prefix checkpoint");
        for(const auto& r:regions_) {
            const auto& b=state.layers[r.layer].*fields[r.field];
            const bool rows=r.field>=2 && r.field<=4;const uint64_t stride=(r.field==4?128:512)*4;
            check(b && b->bytes==r.full_bytes && r.offset+r.bytes<=b->bytes && r.host && r.bytes<=r.host->bytes &&
                (!rows || (r.bytes==4*stride && r.offset==tokens_*stride)),"invalid prefix restore geometry");
        }
    }
    void restore_prefix(State& state,uint32_t keep) const {
        check_prefix(state,keep);state.valid=false;
        for(const auto& r:regions_) {
            const bool rows=r.field>=2 && r.field<=4;const uint64_t skip=rows?keep*(r.field==4?128ull:512ull)*4:0;
            std::memcpy((state.layers[r.layer].*fields[r.field])->data+r.offset+skip,r.host->data+skip,r.bytes-skip);
        }
        for(int l=0;l<Layers;++l) state.layers[l].position=positions_[l];
        state.history=history_;state.tokens=tokens_;state.trace_session_id=trace_;
    }
    void commit_prefix(State& state,std::span<const int> ids) const {
        check(!state.valid && state.tokens==tokens_ && ids.size()>=1 && ids.size()<=4,"invalid prefix commit");
        for(auto id:ids) check(id>=0 && id<Vocab,"invalid prefix history");
        for(int l=0;l<Layers;++l) state.layers[l].position=positions_[l]+uint32_t(ids.size());
        for(auto id:ids) state.history={state.history[1],id};
        state.tokens=tokens_+uint32_t(ids.size());state.valid=true;
    }
'''
REPAIR=r'''            if(keep<width) {
                const auto reads_before=TargetRecoveryAccess::reads(model);
                const auto forwards_before=recovery_target_forward_calls;
                const auto begin=monotonic_ns();model.diagnostic_drain();
                if(state_only) {
                    journal.validate(state,keep);checkpoint.check_prefix(state,keep);mtp_recovery::cancelled(&stopped);
                    checkpoint.restore_prefix(state,keep);target_restore_ns=monotonic_ns()-begin;
                    const auto repair_begin=monotonic_ns();
                    try {journal.apply(gpu,state,keep,&stopped);checkpoint.commit_prefix(state,std::span(ids).first(keep));}
                    catch(...) {state.valid=false;try{gpu.finish();}catch(...){}throw;}
                    target_repair_ns=monotonic_ns()-repair_begin;
                } else {
                    checkpoint.restore(state);target_restore_ns=monotonic_ns()-begin;
                    const auto repair_begin=monotonic_ns();
                    for(uint32_t i=0;i<keep;++i) {
                        model.phase("decode");auto row=model.forward(std::span(ids).subspan(i,1),state,true,&stopped);
                        check(row_hash(row)==row_hash(std::span(logits).subspan(i*Vocab,Vocab)),"target rejection replay changed logits");
                    }
                    target_repair_ns=monotonic_ns()-repair_begin;
                }
                target_recovery_read_bytes=TargetRecoveryAccess::reads(model)-reads_before;
                target_recovery_forward_calls=recovery_target_forward_calls-forwards_before;
                check(!state_only || (!target_recovery_read_bytes && !target_recovery_forward_calls),"state repair executed target experts");
            }
'''


def generated(output):
    s=base.generated(output);p=output/'include/qwen/model.hpp'
    s[p]=replace(s[p],'    friend struct DraftAccess;','    friend struct DraftAccess;\n    friend struct TargetRecoveryAccess;')
    p=output/'model.cpp';t='#include "mtp_target_recovery.hpp"\n'+s[p]
    t=replace(t,'std::vector<float> Model::forward(std::span<const int> ids,State& state,bool logits,const std::atomic<bool>* cancel) {',
        'std::vector<float> Model::forward(std::span<const int> ids,State& state,bool logits,const std::atomic<bool>* cancel) {\n'
        '    ++recovery_target_forward_calls;')
    t=replace(t,'    auto y=gpu_.gdn_scan(normalized,a,beta,resident_->at(b+".A_log"),resident_->at(b+".dt_bias"),',
        '    if(mtp_recovery::capturing) mtp_recovery::capturing->capture(gpu_,layer,qkv,normalized,a,beta,\n'
        '        resident_->at(b+".A_log"),resident_->at(b+".dt_bias"),resident_->dtype(b+".A_log"),resident_->dtype(b+".dt_bias"));\n'
        '    auto y=gpu_.gdn_scan(normalized,a,beta,resident_->at(b+".A_log"),resident_->at(b+".dt_bias"),')
    t=replace(t,'    auto convolved=conv(normalized,state.ple_conv,b+".conv1d.weight",Hyper,tokens,3);',
        '    if(mtp_recovery::capturing) mtp_recovery::capturing->capture_ple(gpu_,normalized);\n'
        '    auto convolved=conv(normalized,state.ple_conv,b+".conv1d.weight",Hyper,tokens,3);')
    s[p]=t;p=output/'probe.cpp';t=('#include "mtp_target_recovery.hpp"\n'
        '#include <cerrno>\n#include <fcntl.h>\n#include <system_error>\n#include <unistd.h>\n'+s[p])
    t=replace(t,'};\nJson checkpoint_self_test()',CHECKPOINT+'};\nJson checkpoint_self_test()')
    t=replace(t,'struct ContinuationInput {',FIXTURE.read_text()+'\nstruct ContinuationInput {')
    t=replace(t,'"native_mtp_continuation_v1"','"native_mtp_continuation_v2"')
    t=replace(t,'    const ContinuationInput work(input,mode);const auto& prompt=work.prompt;',
        '    const ContinuationInput work(input,mode);const auto& prompt=work.prompt;\n'
        '    const char* recovery_setting=std::getenv("FREELLM_TARGET_RECOVERY");\n'
        '    check(recovery_setting && (std::string_view(recovery_setting)=="full-replay" || std::string_view(recovery_setting)=="state-only"),"explicit target recovery arm required");\n'
        '    const bool state_only=std::string_view(recovery_setting)=="state-only";\n'
        '    const uint32_t forced_keep=input.value("force_prefix",1u);check(forced_keep>=1 && forced_keep<=4,"invalid forced prefix");\n'
        '    const std::string capture_path=input.value("capture_recovery",std::string{});\n'
        '    check(capture_path.empty() || (work.validation && !state_only),"capture requires reference validation");\n'
        '    report["target_recovery"]=recovery_setting;report["request_id"]=report.at("input_sha256").get<std::string>()+":"+std::to_string(monotonic_ns());')
    t=replace(t,'    mtp_direct::enabled=std::string_view(direct_setting)=="on";',
        '    check(std::string_view(direct_setting)=="on","recovery trial requires direct outputs");mtp_direct::enabled=true;')
    t=replace(t,'+mtp_scratch::ReserveBytes;','+mtp_scratch::ReserveBytes+mtp_recovery::ReserveBytes;')
    t=replace(t,'{"expert_scratch_reserve_bytes",mtp_scratch::ReserveBytes}',
        '{"expert_scratch_reserve_bytes",mtp_scratch::ReserveBytes},{"target_recovery_reserve_bytes",mtp_recovery::ReserveBytes}')
    t=replace(t,'\n    CheckpointCopy checkpoint(state);\n',
        '\n    mtp_recovery::Journal journal(gpu);\n    CheckpointCopy checkpoint(state);\n')
    t=replace(t,'        std::vector<int> ids{next};uint64_t draft_ns=0,verify_ns=0,recovery_ns=0;',
        '        std::vector<int> ids{next};uint64_t draft_ns=0,verify_ns=0,recovery_ns=0,checkpoint_save_ns=0;\n'
        '        uint64_t target_restore_ns=0,target_repair_ns=0,draft_restore_ns=0,draft_catchup_ns=0;\n'
        '        uint64_t target_recovery_read_bytes=0,target_recovery_forward_calls=0;')
    t=replace(t,'            checkpoint.save(state,4);draft_checkpoint.save(gpu,draft_state,4);',
        '            const auto save_start=monotonic_ns();checkpoint.save(state,4);draft_checkpoint.save(gpu,draft_state,4);\n'
        '            checkpoint_save_ns=monotonic_ns()-save_start;')
    t=replace(t,'        const bool forced=work.validation && consumed==0 && width==4;\n        if(forced) ids[1]=(work.expected[0]+1)%Vocab;',
        '        const bool forced=work.validation && consumed==0 && width==4 && forced_keep<4 && capture_path.empty();\n'
        '        if(work.validation && consumed==0 && width==4 && capture_path.empty())\n'
        '            for(uint32_t i=1;i<forced_keep;++i) ids[i]=work.expected[i-1];\n'
        '        if(forced) ids[forced_keep]=(work.expected[forced_keep-1]+1)%Vocab;\n'
        '        std::unique_ptr<RecoveryBundle> bundle;\n'
        '        if(!capture_path.empty() && consumed==0) {\n'
        '            capture_memory(report,"before_snapshot_write");bundle=std::make_unique<RecoveryBundle>(capture_path);\n'
        '            bundle->state("before",state);capture_memory(report,"before_snapshot_written");}\n')
    old='        model.phase("decode");const auto vstart=monotonic_ns();auto logits=model.forward(ids,state,true,&stopped);verify_ns=monotonic_ns()-vstart;'
    t=replace(t,old,
        '        model.phase("decode");const auto vstart=monotonic_ns();std::vector<float> logits;\n'
        '        {mtp_recovery::Capture capture(width==4 && (state_only || bundle)?&journal:nullptr,ids,state.tokens);\n'
        '         logits=model.forward(ids,state,true,&stopped);capture.finish();}\n'
        '        verify_ns=monotonic_ns()-vstart;\n'
        '        if(bundle) {capture_recovery_fixture(*bundle,model,state,checkpoint,journal,ids,report,output);return;}')
    t=replace(t,'        if(forced) check(good==0,"forced continuation rejection failed");',
        '        if(forced) check(good+1==forced_keep,"forced continuation rejection failed");')
    begin=t.index('            if(keep<width) {',t.index('void joint('));end=t.index('            draft_checkpoint.restore(gpu,draft_state);',begin)
    t=t[:begin]+REPAIR+t[end:]
    t=replace(t,'            draft_checkpoint.restore(gpu,draft_state);',
        '            const auto restore_draft=monotonic_ns();draft_checkpoint.restore(gpu,draft_state);\n'
        '            draft_restore_ns=monotonic_ns()-restore_draft;const auto catchup_begin=monotonic_ns();')
    t=replace(t,'        } else if(!work.serial) {',
        '            draft_catchup_ns=monotonic_ns()-catchup_begin;\n        } else if(!work.serial) {\n'
        '            const auto catchup_begin=monotonic_ns();')
    t=replace(t,'        recovery_ns=monotonic_ns()-recovery_start;',
        '        recovery_ns=monotonic_ns()-recovery_start;')
    t=replace(t,'            else draft.forward(ids,seed,draft_state,false,&stopped);',
        '            else draft.forward(ids,seed,draft_state,false,&stopped);\n            draft_catchup_ns=monotonic_ns()-catchup_begin;')
    t=replace(t,'        report["cycles"].push_back({{"offset",offset}',
        '        report["cycles"].push_back({{"request_id",report.at("request_id")},{"cycle_id",report["cycles"].size()},\n'
        '            {"checkpoint_save_ns",checkpoint_save_ns},{"target_restore_ns",target_restore_ns},{"target_repair_ns",target_repair_ns},\n'
        '            {"draft_restore_ns",draft_restore_ns},{"draft_catchup_ns",draft_catchup_ns},\n'
        '            {"target_recovery_read_bytes",target_recovery_read_bytes},{"target_recovery_forward_calls",target_recovery_forward_calls},{"offset",offset}')
    t=replace(t,'    report["ngram_after_decode"]=NgramAudit::summary(model);',
        '    report["target_recovery_journal"]=journal.stats();\n    report["ngram_after_decode"]=NgramAudit::summary(model);')
    t=replace(t,'        check(argc==6,"usage:',
        '        if((argc==3 || argc==4) && std::string_view(argv[1])=="--recovery-self-test") {\n'
        '            check(!std::filesystem::exists(argv[2]),"self-test output exists");\n'
        '            save_report(argv[2],recovery_self_test(argc==4?argv[3]:"",hash_file(argv[0])));return 0;\n        }\n'
        '        if(argc==4 && std::string_view(argv[1])=="--replay-recovery") {\n'
        '            check(!std::filesystem::exists(argv[3]),"replay output exists");\n'
        '            save_report(argv[3],replay_recovery_fixture(argv[2],hash_file(argv[0])));return 0;\n        }\n'
        '        check(argc==6,"usage:')
    t=replace(t,'        auto input=read_json(argv[3]);report.update(',
        '        report["producer_binary_sha256"]=hash_file(argv[0]);\n        auto input=read_json(argv[3]);report.update(')
    s[p]=t;return s


def settings(output):return base.settings(output)
def inputs(c):return [*base.inputs(c),Path(__file__).resolve(),HEADER,FIXTURE]
def proof(c):
    return dict(kind='target_recovery_producer_v1',base_native_fingerprint=build_fingerprint(ROOT),
        inputs={str(p):sha(p) for p in inputs(c)},generated={str(p):sha(p) for p in generated(c['output'])},
        compiler=c['compiler'],linker=c['linker'],production_promoted=False)
def build(output):
    c=settings(output);c['output'].mkdir(parents=True,exist_ok=False)
    for p,value in generated(c['output']).items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
    frozen=proof(c);save(c['output']/'producer.json',dict(frozen,complete=False))
    with (c['output']/'build.log').open('w') as log:
        for cmd in [*c['compiler'],c['linker']]:subprocess.run(cmd,cwd=c['native'],stdout=log,stderr=subprocess.STDOUT,timeout=180,check=True)
    require(frozen==proof(c),'Recovery build inputs changed')
    result=dict(frozen,complete=True,binary=str(c['binary']),binary_sha256=sha(c['binary']),objects={str(p):sha(p) for p in c['objects']})
    save(c['output']/'producer.json',result);return result
def verify(directory):
    c=settings(directory);saved=json.loads((c['output']/'producer.json').read_text())
    require(saved==dict(proof(c),complete=True,binary=str(c['binary']),binary_sha256=sha(c['binary']),
        objects={str(p):sha(p) for p in c['objects']}),'Changed recovery producer')
    for p,value in generated(c['output']).items():require(p.read_text()==value,'Changed recovery source copy')
    return c,saved
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    r=build(p.parse_args().output);print(json.dumps({k:r[k] for k in ('complete','binary','binary_sha256')}))
