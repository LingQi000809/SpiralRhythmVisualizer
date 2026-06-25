#!/usr/bin/env python3
"""
Read a debug JSON written by dump_debug_json and render note sequences to WAV.

Usage:
    python render_debug.py debug_similarity.json [output_dir]

Writes four files into output_dir (defaults to same folder as the JSON):
    <stem>_input_raw.wav
    <stem>_input_flat.wav
    <stem>_output_raw.wav
    <stem>_output_flat.wav
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from test_melody_similarity import _render_notes_to_wav


def _load_notes(lst):
    return [(float(o), float(f), int(p), float(a)) for o, f, p, a in lst]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    json_path = sys.argv[1]
    out_dir   = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(json_path))
    os.makedirs(out_dir, exist_ok=True)

    with open(json_path) as fh:
        data = json.load(fh)

    stem = os.path.splitext(os.path.basename(json_path))[0]

    print(f"Source JSON : {json_path}")
    print(f"Output dir  : {out_dir}")
    print(f"Matches in JSON: {len(data.get('matches', []))}")
    print()

    pairs = [
        ("raw_notes_in",   f"{stem}_input_raw.wav"),
        ("flat_notes_in",  f"{stem}_input_flat.wav"),
        ("raw_notes_out",  f"{stem}_output_raw.wav"),
        ("flat_notes_out", f"{stem}_output_flat.wav"),
    ]
    for key, filename in pairs:
        notes = _load_notes(data[key])
        _render_notes_to_wav(notes, os.path.join(out_dir, filename))


if __name__ == "__main__":
    main()
