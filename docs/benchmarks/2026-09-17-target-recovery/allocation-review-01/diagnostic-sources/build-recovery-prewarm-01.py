"""Source-copy diagnostic: prepare only observed kernels, with late-compile accounting."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path('/repo')
SNAPSHOT=ROOT/'.cache/experiment-sources/target-recovery-06'
sys.path.insert(0,str(SNAPSHOT/'scripts/qwen'))
import build_target_recovery as base
from build_block_cache_trace import replace
from qualification_evidence import save,sha

out=ROOT/'.cache/recovery-prewarm-build-01'
reports=[ROOT/'docs/benchmarks/2026-09-17-target-recovery/numerical-diagnostic-01'/f'case-{i}-{arm}.json'
         for i in range(5) for arm in ('full-replay','state-only')]
reports += [ROOT/'docs/benchmarks/2026-09-16-mtp-direct-output/long-01'/f'case-{i}-pair-0-on.json' for i in range(3)]
names=sorted({name for p in reports for name,n in json.loads(p.read_text())['after']['metal']['kernel_dispatches'].items() if n})
cfg=base.settings(out);out.mkdir(exist_ok=False);sources=base.generated(out)
p=out/'metal.mm';s=sources[p]
s=replace(s,'    uint64_t peak_groups=0;', '    uint64_t peak_groups=0,late_pipeline_creations=0;\n    bool prewarmed=false;\n    std::string prewarm_mode="unprepared";')
s=replace(s,'''    @autoreleasepool {
        for(NSString* key in impl_->library.functionNames) if(!impl_->pipelines[key]) {''', '''    const char* mode=std::getenv("FREELLM_PIPELINE_PREWARM");
    if(!mode || (std::string_view(mode)!="all" && std::string_view(mode)!="observed"))
        throw std::invalid_argument("explicit pipeline prewarm arm must be all or observed");
    impl_->prewarm_mode=mode;
    static const std::set<std::string> selected={'''+','.join(json.dumps(n) for n in names)+'''};
    @autoreleasepool {
        for(NSString* key in impl_->library.functionNames) if(!impl_->pipelines[key] &&
            (impl_->prewarm_mode=="all" || selected.contains(key.UTF8String))) {''')
s=replace(s,'            impl_->pipelines[key]=pipeline;\n        }\n    }\n}',
    '            impl_->pipelines[key]=pipeline;\n        }\n        impl_->prewarmed=true;\n    }\n}')
s=replace(s,'        if(!pipeline) {\n            NSError* error=nil;',
    '        if(!pipeline) {\n            if(p.prewarmed) ++p.late_pipeline_creations;\n            NSError* error=nil;')
s=replace(s,'{"kernel_dispatches",impl_->kernel_counts}',
    '{"kernel_dispatches",impl_->kernel_counts},{"pipeline_preparation",{{"mode",impl_->prewarm_mode},'+
    '{"prepared",impl_->prewarmed},{"available",[impl_->library.functionNames count]},'+
    '{"live",[impl_->pipelines count]},{"late_creations",impl_->late_pipeline_creations}}}')
sources[p]=s
for p,text in sources.items():p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
proof=dict(kind='mtp_pipeline_prewarm_diagnostic_producer_v1',complete=False,
    inputs={str(p):sha(p) for p in [*base.inputs(cfg),Path(__file__),*reports]},
    generated={str(p):sha(p) for p in sources},selected_kernels=names,
    compiler=cfg['compiler'],linker=cfg['linker'],base_native_fingerprint=base.build_fingerprint(SNAPSHOT),production_promoted=False)
save(out/'producer.json',proof)
with (out/'build.log').open('w') as log:
    for cmd in [*cfg['compiler'],cfg['linker']]:
        subprocess.run(cmd,cwd=cfg['native'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
proof.update(complete=True,binary=str(cfg['binary']),binary_sha256=sha(cfg['binary']),objects={str(p):sha(p) for p in cfg['objects']})
save(out/'producer.json',proof)
print(json.dumps(dict(binary_sha256=proof['binary_sha256'],selected_kernels=names)))
