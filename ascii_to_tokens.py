#!/usr/bin/env python3
"""
ascii_to_tokens.py
===================
Converts an MM2 ASCII level grid (.txt, one row per line) into a simplified
token grid where every tile character becomes a "T<NN>" token, with NN being
that character's numeric tile ID in a tileset JSON (default: mm2_tileset_we.json).

This token form is easier for LLMs to read and count reliably than the raw
extended-ASCII characters used by the MM2 tilesets.

Usage:
    python ascii_to_tokens.py level.txt level_tokens.txt
    python ascii_to_tokens.py ascii_levels/ token_levels/ --tileset mm2_tileset_we.json
"""

import argparse
import json
import os
import sys

UNKNOWN_TOKEN = "T??"


def build_id_to_char(tileset_path):
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tile_chars = sorted(data["tiles"].keys())
    if "_" not in tile_chars:
        tile_chars.append("_")
    return {idx: char for idx, char in enumerate(tile_chars)}


def build_char_to_token(id_to_char):
    """Map each tile character to a 'T<NN>' token, NN = its numeric tile ID."""
    width = max(2, len(str(len(id_to_char) - 1)))
    return {char: f"T{idx:0{width}d}" for idx, char in id_to_char.items()}


def convert_lines(lines, char_to_token):
    out_lines = []
    unknown = set()
    for line in lines:
        tokens = []
        for ch in line:
            token = char_to_token.get(ch)
            if token is None:
                token = UNKNOWN_TOKEN
                unknown.add(ch)
            tokens.append(token)
        out_lines.append(" ".join(tokens))
    return out_lines, unknown


def convert_file(in_path, out_path, char_to_token):
    with open(in_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    out_lines, unknown = convert_lines(lines, char_to_token)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    return unknown


def main():
    parser = argparse.ArgumentParser(
        description="Convert MM2 ASCII level grids into T0x token grids based on a tileset."
    )
    parser.add_argument("input", help="Input ASCII grid .txt file, or a folder of .txt files.")
    parser.add_argument("output", help="Output .txt file, or folder (if input is a folder).")
    parser.add_argument(
        "--tileset",
        default="mm2_tileset_we.json",
        help="Tileset JSON defining the T0x token numbering. Default: mm2_tileset_we.json",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.tileset):
        print(f"Error: tileset not found: {args.tileset}")
        sys.exit(1)

    id_to_char = build_id_to_char(args.tileset)
    char_to_token = build_char_to_token(id_to_char)

    all_unknown = set()

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        count = 0
        for fname in sorted(os.listdir(args.input)):
            if not fname.endswith(".txt"):
                continue
            unknown = convert_file(
                os.path.join(args.input, fname),
                os.path.join(args.output, fname),
                char_to_token,
            )
            all_unknown |= unknown
            count += 1
        print(f"Converted {count} file(s) -> {args.output}")
    else:
        all_unknown = convert_file(args.input, args.output, char_to_token)
        print(f"Converted {args.input} -> {args.output}")

    if all_unknown:
        print(
            f"WARNING: {len(all_unknown)} character(s) not found in "
            f"'{os.path.basename(args.tileset)}', mapped to {UNKNOWN_TOKEN}: "
            + " ".join(repr(c) for c in sorted(all_unknown))
        )


if __name__ == "__main__":
    main()
