#!/usr/bin/env python3
"""
build_dataset_with_ascii.py
===========================
Custom local pipeline designed to parse consolidated text files featuring
tags and mixed structural brackets.
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW_H = 20
WINDOW_W = 20
EXTRA_TILE = "_"

def load_tileset(path):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: Tileset file not found at {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chars = sorted(data["tiles"].keys())
    if EXTRA_TILE not in chars:
        chars.append(EXTRA_TILE)
    return {ch: idx for idx, ch in enumerate(chars)}

def _pad_rows(rows, width, empty_char):
    pad_rows = max(0, WINDOW_H - len(rows))
    padded = [empty_char * width] * pad_rows + list(rows)
    return [r.ljust(width, empty_char) for r in padded]

def extract_best_window(rows, tile_to_id, empty_char="-"):
    extra_id = tile_to_id.get(EXTRA_TILE, 0)
    empty_id = tile_to_id.get(empty_char, 0)

    height = len(rows)
    width = max((len(r) for r in rows), default=0)

    if width < WINDOW_W or height == 0:
        return None

    padded = _pad_rows(rows, width, empty_char)

    best_score = -1
    best_scene = None

    for x in range(width - WINDOW_W + 1):
        scene = []
        score = 0
        for y in range(WINDOW_H):
            row_slice = padded[y][x : x + WINDOW_W]
            id_row = []
            for ch in row_slice:
                tid = tile_to_id.get(ch, extra_id)
                id_row.append(tid)
                if tid not in (empty_id, extra_id):
                    score += 1
            scene.append(id_row)

        if score > best_score:
            best_score = score
            best_scene = scene

    return best_scene

def extract_all_windows(rows, tile_to_id, stride=1, empty_char="-"):
    """Slide a WINDOW_H x WINDOW_W window across the level and return every window."""
    extra_id = tile_to_id.get(EXTRA_TILE, 0)

    width = max((len(r) for r in rows), default=0)

    if width < WINDOW_W or not rows:
        return []

    padded = _pad_rows(rows, width, empty_char)

    scenes = []
    for x in range(0, width - WINDOW_W + 1, stride):
        scene = []
        for y in range(WINDOW_H):
            row_slice = padded[y][x : x + WINDOW_W]
            scene.append([tile_to_id.get(ch, extra_id) for ch in row_slice])
        scenes.append(scene)

    return scenes

def parse_source_file(file_path):
    """
    Parses a file containing raw '' annotations, isolating blocks
    of map data and discarding structural wrappers (e.g., {{{, }}}).
    """
    text = Path(file_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    
    levels = {}
    current_source = None
    current_rows = []

    for line in lines:
        cleaned_line = line.strip()
        
        # Skip pure structural markers or blank separator lines
        if not cleaned_line or cleaned_line.startswith("{{{") or cleaned_line.startswith("}}}"):
            continue
            
        # FIXED: Correctly matches using escaped square brackets \[ and \]. No parentheses mismatch.
        match = re.match(r'^\s*\(([^)]*)\)(.*)', line)
        
        if match:
            # Save the prior map source tracking if valid
            if current_source and current_rows:
                levels[current_source] = current_rows
                
            source_num = match.group(1)
            current_source = f"source_{source_num}"
            
            # Grab only the remaining map section explicitly trailing the header tag
            map_part = match.group(2)
            # If there's map data on the tag line, keep it, preserving structural spacing
            if map_part.strip():
                current_rows = [map_part]
            else:
                current_rows = []
        else:
            # Continue appending lines to the active tracking source
            if current_source is not None:
                current_rows.append(line)

    # Save the final trailing segment remaining at EOF
    if current_source and current_rows:
        levels[current_source] = current_rows

    if not levels:
        rows = []
        for line in lines:
            cleaned = line.strip()
            if cleaned.startswith("{{{") or cleaned.startswith("}}}"):
                continue
            if cleaned:
                rows.append(line)

        if rows:
            levels["source_0"] = rows

    return levels

def collect_input_files(input_path):
    p = Path(input_path)
    if p.is_dir():
        files = sorted(p.glob("*.txt"))
        if not files:
            sys.exit(f"ERROR: No .txt files found in folder {input_path}")
        return files
    if p.is_file():
        return [p]
    sys.exit(f"ERROR: Input path not found: {input_path}")

def load_converter(filename, module_name):
    import importlib.util
    path = os.path.join(HERE, filename)
    if not os.path.isfile(path):
        sys.exit(f"ERROR: Converter module missing from {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description="Build dataset from custom tagged text files.")
    parser.add_argument("--input_file", required=True, help="Path to a .txt file or a folder of .txt files.")
    parser.add_argument("--output", required=True, help="Output JSON filename.")
    parser.add_argument("--tileset", default=os.path.join(HERE, "smb.json"), help="Path to tileset JSON.")
    convert_group = parser.add_mutually_exclusive_group()
    convert_group.add_argument("--convert_to_vglc", action="store_true",
                               help="Convert layout to VGLC structure (ascii_to_vglc.py).")
    convert_group.add_argument("--convert_to_extended", action="store_true",
                               help="Convert layout to extended tile format (mm2view_to_extended.py).")
    convert_group.add_argument("--include_all", action="store_true",
                               help="Brute-force mode: skip all converters and windowing, include every level as-is.")
    parser.add_argument("--sliding_window", action="store_true",
                        help="Collect every window position as a separate sample instead of keeping only the best window.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Step size (in tiles) between windows when --sliding_window is active. Default: 1.")
    args = parser.parse_args()

    tile_to_id = load_tileset(args.tileset)

    converter_mod = None
    if args.convert_to_vglc:
        converter_mod = load_converter("ascii_to_vglc.py", "ascii_to_vglc")
    elif args.convert_to_extended:
        converter_mod = load_converter("mm2view_to_extended.py", "mm2view_to_extended")
    elif args.include_all:
        print("Brute-force mode: all levels included without conversion or windowing.")

    input_files = collect_input_files(args.input_file)
    dataset = []
    processed = 0
    skipped = 0

    for input_file in input_files:
        raw_levels = parse_source_file(input_file)
        file_stem = input_file.stem
        print(f"Parsing content from {input_file}...")

        for name, rows in raw_levels.items():
            # Prefix with the source filename so names stay unique across files
            full_name = f"{file_stem}/{name}" if len(input_files) > 1 else name
            try:
                if args.include_all:
                    rows = [r.rstrip('\r\n') for r in rows]
                    # Strip leading blank rows (raw MM2 ASCII sky is all spaces)
                    while rows and not rows[0].strip():
                        rows.pop(0)
                    # Take the bottom WINDOW_H rows so ground is always included
                    if len(rows) > WINDOW_H:
                        rows = rows[-WINDOW_H:]
                    empty_char = " "
                elif converter_mod is not None:
                    rows = converter_mod.convert_level(rows)
                    empty_char = "-"
                else:
                    rows = [r.rstrip('\r\n') for r in rows]
                    empty_char = "-"

                if args.sliding_window:
                    scenes = extract_all_windows(rows, tile_to_id, stride=args.stride, empty_char=empty_char)
                    if not scenes:
                        print(f"  [SKIP] {full_name} (empty)")
                        skipped += 1
                        continue
                    for i, scene in enumerate(scenes):
                        dataset.append({"name": f"{full_name}_{i}", "scene": scene})
                    processed += len(scenes)
                    print(f"  [OK] {full_name} ({len(scenes)} windows)")
                else:
                    scene = extract_best_window(rows, tile_to_id, empty_char=empty_char)
                    if scene is None:
                        print(f"  [SKIP] {full_name} (empty)")
                        skipped += 1
                        continue
                    dataset.append({"name": full_name, "scene": scene})
                    processed += 1
                    print(f"  [OK] {full_name}")

            except Exception as e:
                print(f"  [ERROR] Failed processing {full_name}: {e}")
                skipped += 1

    # Save output dataset
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nCompleted! Packaged {processed} items into {output_file} ({skipped} skipped).")

if __name__ == "__main__":
    main()

#num_tiles = 138 for mm2_tileset_full
#num_tiles = 