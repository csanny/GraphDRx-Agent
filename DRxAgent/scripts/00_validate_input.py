#!/usr/bin/env python3
from __future__ import annotations

# Allow direct execution as `python scripts/<name>.py` without package installation.
import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from pathlib import Path

from graphdrx_agent_eval.common import atomic_write_json, read_jsonl
from graphdrx_agent_eval.schemas import normalize_packet, validate_packets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--normalized-output", required=True)
    ap.add_argument("--expected-diseases", type=int, default=15)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    packets = [normalize_packet(x) for x in read_jsonl(args.input)]
    errors = validate_packets(packets, args.expected_diseases, args.top_k)
    out = Path(args.normalized_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in sorted(packets, key=lambda x: (x["disease"].lower(), x["rank"])):
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    atomic_write_json(out.with_suffix(".validation.json"), {"valid": not errors, "errors": errors, "n_packets": len(packets)})
    if errors:
        print("INPUT INVALID")
        for e in errors:
            print("-", e)
        raise SystemExit(2)
    print(f"PASS: {len(packets)} pairs; {args.expected_diseases} diseases x top-{args.top_k}")


if __name__ == "__main__":
    main()
