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

def extract_best_window(rows, tile_to_id):
    empty_id = tile_to_id.get("-", 0)
    extra_id = tile_to_id.get(EXTRA_TILE, 0)

    height = len(rows)
    width = max((len(r) for r in rows), default=0)

    if width < WINDOW_W or height == 0:
        return None

    pad_rows = max(0, WINDOW_H - height)
    padded = ["-" * width] * pad_rows + list(rows)
    padded = [r.ljust(width, "-") for r in padded]

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

def main():
    parser = argparse.ArgumentParser(description="Build dataset from custom tagged text files.")
    parser.add_argument("--input_file", required=True, help="Path to the custom text file (e.g., d.txt).")
    parser.add_argument("--output", required=True, help="Output JSON filename.")
    parser.add_argument("--convert_to_vglc", action="store_true", help="Convert layout to VGLC structure.")
    parser.add_argument("--tileset", default=os.path.join(HERE, "smb.json"), help="Path to smb.json tileset.")
    args = parser.parse_args()

    tile_to_id = load_tileset(args.tileset)

    if args.convert_to_vglc:
        import importlib.util
        # FILENAME FIXED PERMANENTLY: Points exactly to ascii_to_vglc.py
        vglc_filename = "ascii_to_vglc.py"
        vglc_path = os.path.join(HERE, vglc_filename)
        if not os.path.isfile(vglc_path):
            sys.exit(f"ERROR: Custom VGLC converter missing from {vglc_path}")
        spec = importlib.util.spec_from_file_location("ascii_to_vglc", vglc_path)
        vglc_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vglc_mod)

    # Extract clean blocks from the source file
    raw_levels = parse_source_file(args.input_file)
    
    dataset = []
    processed = 0
    skipped = 0

    print(f"Parsing content from {args.input_file}...")

    for name, rows in raw_levels.items():
        try:
            # Convert to VGLC structure if requested
            if args.convert_to_vglc:
                rows = vglc_mod.convert_level(rows)
            else:
                # Fallback trailing-strip if using manual layouts
                rows = [r.rstrip('\r\n') for r in rows]

            # Slide window and extract token grid
            scene = extract_best_window(rows, tile_to_id)
            if scene is None:
                print(f"  [SKIP] {name} (too narrow or empty)")
                skipped += 1
                continue

            dataset.append({
                "name": name,
                "scene": scene
            })
            processed += 1
            print(f"  [OK] {name}")

        except Exception as e:
            print(f"  [ERROR] Failed processing {name}: {e}")
            skipped += 1

    # Save output dataset
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"\nCompleted! Packaged {processed} items into {output_file} ({skipped} skipped).")

if __name__ == "__main__":
    main()