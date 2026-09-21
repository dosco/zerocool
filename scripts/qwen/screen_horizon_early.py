#!/usr/bin/env python3
"""Reject weak horizons early; diagnostic numerics never qualify adoption."""
import argparse
import hashlib
from pathlib import Path

import build_streamed_verifier_horizon as builder
from benchmark_host import build_probe
from cache_residency import require
from combined_q4 import freeze
from perfect_draft import configuration
from qualification_evidence import ResourceBlocked, save, sha, verify_seal
from screen_mtp_continuation import PREPARED, host_check
from screen_verifier_horizon import read, validate, compare, prefix
from stage200 import Experiment
from target_recovery_checks import resource_failure, replay_resources

ROOT=builder.ROOT
PROTOCOL=ROOT/'docs/benchmarks/2026-09-17-verifier-horizon/early-protocol.md'


def diagnostic_resources(result,raw):
    # This bound permits only a numerical diagnostic, never a timing result.
    memory=[raw['before_load'],raw['before']['process'],raw['after']['process'],raw['after_destroy']]
    memory += [c[k] for c in raw['cycles'] for k in ('memory_before','memory_after')]
    require(result['clean_host'] and max(m['compressed_peak_bytes'] for m in memory)<=128*1024**2 and
        len({m['system_swap_used_bytes'] for m in memory})==1,'Numerical diagnostic exceeded its explicit resource bound')


def run(output,directory,source,tiled=False):
    if tiled:
        import build_tiled_verifier_horizon as producer_builder
    else: producer_builder=builder
    source=Path(source).resolve();verify_seal(source,sha(source/'evidence-files.json'))
    summary=read(source/'summary.json');require(summary['status']=='resource_blocked' and summary['embedding_storage']=='rows',
        'Early screen requires the preserved streamed attempt')
    parent=read(source/'producer.json');require(builder.verify(Path(parent['binary']).parent)[1]==parent,'Numerical parent producer changed')
    cfg,proof=producer_builder.verify(directory)
    require(proof['base_native_fingerprint']==parent['base_native_fingerprint'],'Native base changed')
    if not tiled: require(parent==proof,'Undeclared numerical producer change')
    work=read(source/'workload.json');serial=read(source/'validation-serial.json');short=read(source/'validation-serial.input.json')
    serial_result=validate(serial,short,sha(source/'validation-serial.input.json'),parent['binary_sha256'],1,True,True)
    diagnostic_resources(serial_result,serial)
    require(summary['embedding']['clean_memory'] is summary['embedding']['clean_host'] is True and
            summary['kernel']['clean_memory'] is summary['kernel']['clean_host'] is True,'Missing clean low-memory operator prerequisites')
    configs=[dict(configuration(4,expert_slots=1460),token_tile=w,embedding_storage='rows') for w in (4,8)]
    exp=Experiment(output,'verifier_horizon_early_screen_v1',configs,work,420)
    with exp:
        host=build_probe(ROOT/'.cache/benchmark-host'/hashlib.sha256(str(exp.out).encode()).hexdigest())
        save(exp.out/'producer.json',proof)
        freeze(exp,[*producer_builder.inputs(cfg),*producer_builder.generated(cfg['output']),*cfg['objects'],cfg['binary'],*host['files'],PROTOCOL,
            PREPARED/'manifest.json',PREPARED/'dense.bin',PREPARED/'experts.bin',
            *[p for p in source.iterdir() if p.is_file()]])
        exp.env.update(ZEROCOOL_Q8_EXPANDED='packed',ZEROCOOL_MTP_NGRAM_INIT='lazy',ZEROCOOL_MTP_EXPERT_SCRATCH='off',
                       ZEROCOOL_MTP_DIRECT_OUTPUT='on',ZEROCOOL_TARGET_RECOVERY='full-replay')
        exp.report.update(samples=[],comparisons=[],perfect_proposals_only=True,production_promoted=False,
            advance_to_real_draft=False,qualification_blocked=True,
            compute_tile_cap=4 if tiled else 8,
            numerical_parent=dict(path=str(source),seal_sha256=sha(source/'evidence-files.json'),
                                  source_sha256=sha(source/'validation-serial.json'),observation=serial_result))
        exp.guard.check_resources(initial=True)
        if tiled:
            freeze(exp,[PROTOCOL.parent/'tiled-protocol.md'])
            host_check(exp,host,'kernel',full=False)
            exp.command([cfg['binary'],'--horizon-kernel-test',exp.out/'kernel.json'],'kernel',limit=60,validation=True)
            kernel=read(exp.out/'kernel.json');resources=replay_resources(kernel)
            require(kernel['kind']=='verifier_horizon_kernel_test_v1' and kernel['complete'] is True and
                kernel['performance_measurement'] is False and kernel['peak_gpu_bytes']<=128*1024**2 and len(kernel['cases'])==4 and
                all(x['exact_reference'] is x['guards_intact'] is True for x in kernel['cases']),'Incomplete tiled kernel check')
            exp.report['kernel']=dict(resources,source='kernel.json',sha256=sha(exp.out/'kernel.json'));exp.persist()
            if not resources['clean_memory'] or not resources['clean_host']:raise ResourceBlocked(resource_failure(kernel,'kernel'))
        def sample(width,count,stem,validation):
            values=prefix(work,count);input_path=exp.out/(stem+'.input.json');save(input_path,values);freeze(exp,[input_path])
            host_check(exp,host,stem);exp.env['ZEROCOOL_VERIFIER_HORIZON']=str(width);path=exp.out/(stem+'.json')
            exp.command([cfg['binary'],exp.model,PREPARED,input_path,path,'fast-validate' if validation else 'fast-timing'],
                        stem,limit=180,validation=validation)
            raw=read(path);result=validate(raw,values,sha(input_path),proof['binary_sha256'],width,validation,True)
            require((raw.get('compute_tile_cap')==4) is tiled,'Incorrect compute tile policy')
            exp.report['samples'].append(dict(width=width,validation=validation,source=path.name,input=input_path.name,
                sha256=sha(path),observation=result));exp.persist()
            if validation:diagnostic_resources(result,raw)
            elif not result['clean_memory'] or not result['clean_host']:raise ResourceBlocked(resource_failure(raw,stem))
            return raw
        eight=sample(8,9,'numerical-eight',True)
        numerical=compare(serial,eight,numerical_across_producers=tiled)
        for key in ('ratio','control_verified_tps','candidate_verified_tps'):numerical.pop(key,None)
        exp.report['numerical_comparison']=dict(numerical,timing_used=False,qualification_passed=False);exp.persist()
        four=sample(4,64,'pair-0-width-4',False);eight=sample(8,64,'pair-0-width-8',False)
        decision=compare(four,eight);exp.report['comparisons'].append(decision)
        rejected=decision['ratio']>.85 or decision['candidate_verified_tps']<6.5
        exp.report.update(status='insufficient_verifier_headroom' if rejected else 'promising_preliminary_ceiling',
                          advance_to_acceptance_cost_model=not rejected)
        exp.report['limitations']=['One fresh perfect-proposal pair; no real draft generation or rejection recovery is measured.',
            'Numerical equality is checked separately from memory qualification. Compressed validation remains unqualified.',
            'Normal timing retains zero compression/decompression and unchanged swap. No historical timing is reused.',
            'A promising result cannot qualify adoption; a weak result stops this horizon before more expensive checks.']
    return exp.report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tiled',action='store_true')
    a=p.parse_args();run(a.output,a.build,a.source,a.tiled)
