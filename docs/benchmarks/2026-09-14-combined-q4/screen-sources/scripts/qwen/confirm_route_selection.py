#!/usr/bin/env python3
"""Five fresh router pairs on the unchanged build; no pooling or early success."""
import argparse
from pathlib import Path

from capture_routes import load
from confirm_q8_steady import revalidate as revalidate_pairs, run as run_pairs
from qualification_evidence import confined, sha
from screen_route_selection import configs, observations, revalidate as revalidate_screen

ROOT=Path(__file__).resolve().parents[2]
KIND='route_selection_confirmation_v1'
SCREEN=ROOT/'docs/benchmarks/2026-09-11-route-selection/raw'


def screen_files(directory):
    prior=load(directory/'summary.json');files={sha(directory/'summary.json'):directory/'summary.json'}
    for row in [prior['operator_source'],*prior['correctness_sources'],*prior['measurements']]:
        p=confined(directory,row['source'])
        if sha(p)!=row['sha256']:raise ValueError('Changed router prerequisite source')
        files[row['sha256']]=p
    return prior,files


def revalidate(summary,resolve):
    return revalidate_pairs(summary,resolve,kind=KIND,configs=configs,
        observations=observations,revalidate_screen=revalidate_screen)


def run(args):
    return run_pairs(args,kind=KIND,screen=SCREEN,configs=configs,screen_files=screen_files,
        revalidate_screen=revalidate_screen,observations=observations,revalidate=revalidate)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    raise SystemExit(run(parser.parse_args()))
