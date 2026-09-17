#!/usr/bin/env python3
"""Confirm decode scratch reuse with five fresh pairs after exact state validation."""
import argparse
from pathlib import Path

from capacity_experiment import decide as timing_decision
from capture_routes import load
from confirm_q8_steady import revalidate as revalidate_pairs, run as run_pairs
from qualification_evidence import confined, sha
from screen_decode_scratch import configs, observations, revalidate as revalidate_screen

KIND='decode_scratch_confirmation_v1'


def screen_files(directory):
    prior=load(directory/'summary.json');files={sha(directory/'summary.json'):directory/'summary.json'}
    for row in [prior['prior_output_control'],*prior['measurements'],*prior['correctness_sources']]:
        p=confined(directory,row['source'])
        if sha(p)!=row['sha256']:raise ValueError('Changed scratch prerequisite source')
        files[row['sha256']]=p
    return prior,files


def decide(rows,pairs=5):
    if pairs!=5:raise ValueError('Scratch confirmation requires five pairs')
    result=timing_decision(rows,pairs)
    decode=[r for r in result['ratios'] if r['metric']=='decode_wall_ms']
    # Keep the complete-conversation and TTFT gates. Generation must improve
    # in both phases with paired confidence, not just a favorable point estimate.
    timing_pass=(result['candidate_for_later_qualification'] and
        all(r['confidence_95']['high']<1 and all(v<1 for v in r['ratios']) for r in decode))
    clean=all(r.get('decode_memory_disturbance') is False for row in rows for r in row['requests'])
    qualified=timing_pass and clean
    result.update(status='promising' if qualified else ('memory_disturbed' if timing_pass else 'improvement_not_demonstrated'),
        clean_decode_memory=clean,candidate_for_later_qualification=qualified,
        target_ms_per_token=200,normal_request_latency_qualified=False,production_promoted=False)
    return result


def revalidate(summary,resolve):
    return revalidate_pairs(summary,resolve,kind=KIND,configs=configs,
        observations=observations,revalidate_screen=revalidate_screen,decision=decide)


def run(args):
    return run_pairs(args,kind=KIND,screen=args.screen.resolve(),configs=configs,screen_files=screen_files,
        revalidate_screen=revalidate_screen,observations=observations,revalidate=revalidate,decision=decide)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True,help='Sealed passing screen, including both state/failure reports')
    p.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
