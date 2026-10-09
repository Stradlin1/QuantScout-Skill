#!/usr/bin/env python3
"""Offline validation of an Agent-authored evidence sidecar; no source fetching."""
import argparse
import json
from pathlib import Path
from rdkx5_doctor.official_knowledge import KnowledgeLookup

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('path',type=Path);args=p.parse_args()
    try:
        if args.path.stat().st_size>2*1024*1024:raise ValueError('sidecar exceeds bounded size')
        lookup=KnowledgeLookup.model_validate(json.loads(args.path.read_text()))
    except (ValueError,OSError) as exc:
        p.exit(2,f'Evidence validation failed: {exc}\n')
    print(f'Validated {len(lookup.records)} records; provenance contract only, not source truth or runtime support.')
if __name__=='__main__':main()
