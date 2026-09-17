"""Diagnostic source-copy build; return unused CPU heap pages at startup only."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/repo')
SNAPSHOT = ROOT/'.cache/experiment-sources/target-recovery-06'
sys.path.insert(0, str(SNAPSHOT/'scripts/qwen'))
import build_target_recovery as base
from build_block_cache_trace import replace
from qualification_evidence import save, sha

out = ROOT/'.cache/recovery-startup-relief-build-01'
cfg = base.settings(out)
out.mkdir(exist_ok=False)
sources = base.generated(out)
p = out/'probe.cpp'
s = sources[p]
start = s.index('void joint(const std::filesystem::path& model_path,')
end = s.index('\n}\n\n}\nint main(', start)+2
joint = s[start:end]
joint = replace(joint, '    const ContinuationInput work(input,mode);', '''    const char* relief = std::getenv("FREELLM_STARTUP_HEAP_RELIEF");
    check(relief && (std::string_view(relief)=="off" || std::string_view(relief)=="on"),"explicit heap relief arm required");
    report["startup_heap_relief"]=relief;report["startup_heap"]=Json::array();
    auto heap_boundary=[&](const char* name) {
        malloc_statistics_t a{},b{};malloc_zone_statistics(nullptr,&a);
        const auto before=process_memory();const auto start=monotonic_ns();
        const size_t released=std::string_view(relief)=="on"?malloc_zone_pressure_relief(nullptr,0):0;
        const auto elapsed=monotonic_ns()-start;const auto after=process_memory();
        malloc_zone_statistics(nullptr,&b);
        report["startup_heap"].push_back({{"phase",name},{"released_bytes",released},{"elapsed_ns",elapsed},
            {"before",before},{"after",after},
            {"heap_in_use_before",a.size_in_use},{"heap_in_use_after",b.size_in_use},
            {"heap_allocated_before",a.size_allocated},{"heap_allocated_after",b.size_allocated}});
    };
    const ContinuationInput work(input,mode);''')
joint = replace(joint, '    model.prepare_pipelines();int next=-1;',
                '    model.prepare_pipelines();heap_boundary("loaded");int next=-1;')
joint = replace(joint, '    mtp_recovery::Journal journal(gpu);',
                '    heap_boundary("primed");mtp_recovery::Journal journal(gpu);')
sources[p] = s[:start]+joint+s[end:]
for p, text in sources.items():
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
proof = dict(kind='mtp_startup_heap_diagnostic_producer_v1', complete=False,
    inputs={str(p):sha(p) for p in [*base.inputs(cfg), Path(__file__)]},
    generated={str(p):sha(p) for p in sources},
    compiler=cfg['compiler'], linker=cfg['linker'],
    base_native_fingerprint=base.build_fingerprint(SNAPSHOT), production_promoted=False)
save(out/'producer.json', proof)
with (out/'build.log').open('w') as log:
    for command in [*cfg['compiler'], cfg['linker']]:
        subprocess.run(command, cwd=cfg['native'], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=180)
proof.update(complete=True, binary=str(cfg['binary']), binary_sha256=sha(cfg['binary']),
             objects={str(p):sha(p) for p in cfg['objects']})
save(out/'producer.json', proof)
print(json.dumps(dict(complete=True, binary_sha256=proof['binary_sha256'])))
