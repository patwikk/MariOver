#!/usr/bin/env python3
"""
build_dataset.py
================
Headless end-to-end pipeline:

  1. Stream levels from the TheGreatRambler/mm2_level HuggingFace dataset,
     filtered by a keyword in the level name.
  2. Convert each level to MM2 ASCII format (same logic as mm2_viewer's
     _build_ascii_grid / Export ASCII, but without any tkinter/GUI).
  3. Convert each ASCII level to VGLC format via mm2view_to_vglc.convert_level.
  4. Slide a 20×20 window across each VGLC level and emit one scene per
     level (the single window with the most non-empty content).
  5. Write everything to a JSON dataset file ready for diffusion model
     training, using integer tile IDs from smb.json (VGLC tileset).

Usage
-----
    python build_dataset.py --keyword mario --max_levels 100 --output dataset.json

    # Save intermediate ASCII files too:
    python build_dataset.py --keyword mario --max_levels 50 \
        --output dataset.json --save_ascii ./ascii_levels --save_vglc ./vglc_levels

Required files (same directory as this script)
---------------
    level.py          Kaitai-compiled MM2 level parser
    mm2_viewer.py     Source of level_to_dict and ASCII_MAP
    mm2view_to_vglc.py    Our ASCII→VGLC converter
    smb.json          VGLC tileset (defines tile→id mapping)
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import zlib
from io import BytesIO
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate sibling files relative to this script
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

def _load_module(name, filename):
    path = os.path.join(HERE, filename)
    if not os.path.isfile(path):
        sys.exit(f"ERROR: required file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------
# VGLC tileset  (smb.json)
# ---------------------------------------------------------------------------
VGLC_TILESET_PATH = os.path.join(HERE, "smb.json")
EXTRA_TILE = "_"   # padding / void token

def load_vglc_tileset(path=VGLC_TILESET_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chars = sorted(data["tiles"].keys())
    if EXTRA_TILE not in chars:
        chars.append(EXTRA_TILE)
    return {ch: idx for idx, ch in enumerate(chars)}

# ---------------------------------------------------------------------------
# ASCII grid builder  (pure replication of mm2_viewer._build_ascii_grid,
#                      no tkinter dependency)
# ---------------------------------------------------------------------------
def _obj_id_to_str(obj_id, OBJID_INT_TO_STR, obj_id_to_str_fn):
    return obj_id_to_str_fn(obj_id)

def build_ascii_grid(lvl: dict, ASCII_MAP: dict, obj_id_to_str) -> list[str]:
    """
    Convert a level dict (as produced by level_to_dict) to a list of strings
    in MM2 ASCII format.  Mirrors mm2_viewer._build_ascii_grid exactly.
    Returns a list of row strings (top row first).
    """
    objects = lvl.get("objects", [])
    ground  = lvl.get("ground",  [])
    theme     = lvl.get("theme",     "overworld")
    gamestyle = lvl.get("gamestyle", "smb1")
    is_castle_axe = (theme == "castle" and gamestyle != "sm3dw")

    max_tx = 40
    max_ty = 20
    for o in objects:
        max_tx = max(max_tx, o["x"] // 160 + 2)
        max_ty = max(max_ty, o["y"] // 160 + 2)
    for g in ground:
        max_tx = max(max_tx, g["x"] + 2)
        max_ty = max(max_ty, g["y"] + 2)

    max_tx = min(max_tx, 240) - 1
    max_ty = min(max_ty, 28)

    goal_x_raw  = int(lvl.get("goal_x_raw", 0))
    goal_x_tile = math.ceil(goal_x_raw // 10) if goal_x_raw > 0 else 0

    if is_castle_axe:
        goal_base_col = goal_x_tile if goal_x_tile > 0 else max_tx - 10
        max_tx = max(max_tx, goal_base_col + 2)
    else:
        goal_base_col = goal_x_tile if goal_x_tile > 0 else max_tx - 9
        max_tx = goal_base_col + 11
        max_ty -= 1

    start_ygame     = lvl.get("start_y", 1)
    goal_base_ygame = int(lvl.get("goal_y_raw", 0))
    GOAL_W          = 11 if not is_castle_axe else 10

    grid = [["-"] * max_tx for _ in range(max_ty)]

    def set_cell(col, row_game, ch):
        if 0 <= col < max_tx and 0 <= row_game < max_ty:
            grid[max_ty - 1 - row_game][col] = ch

    for g in ground:
        set_cell(g["x"], g["y"], "#")
    for obj in objects:
        ch = ASCII_MAP.get(obj_id_to_str(obj["id"]), "?")
        set_cell(obj["x"] // 160, obj["y"] // 160, ch)

    # Start ground (7-wide implicit platform)
    for col in range(7):
        for row in range(0, start_ygame):
            set_cell(col, row, "#")
    set_cell(3, start_ygame, "S")

    # Goal ground
    for gc in range(GOAL_W):
        for row in range(0, goal_base_ygame):
            set_cell(goal_base_col + gc, row, "#")
    if is_castle_axe:
        for b in range(14):
            set_cell(goal_base_col - 14 + b, goal_base_ygame - 1, "=")
        set_cell(goal_base_col, goal_base_ygame, "X")
    else:
        set_cell(goal_base_col, goal_base_ygame, "G")

    return ["".join(row) for row in grid]

# ---------------------------------------------------------------------------
# scene extraction  (one best 20×20 window per level)
# ---------------------------------------------------------------------------
WINDOW_H = 20
WINDOW_W = 20

def best_window(vglc_rows: list[str], tile_to_id: dict) -> list[list[int]] | None:
    """
    Slide a WINDOW_H × WINDOW_W window across the VGLC level and return
    the single window that has the most non-empty (non-'-') tiles.
    The level is padded to WINDOW_H rows tall if needed.
    Returns a 2-D list of tile IDs, or None if the level is empty.
    """
    empty_id = tile_to_id.get("-", 0)
    extra_id = tile_to_id.get(EXTRA_TILE, 0)

    height = len(vglc_rows)
    width  = max((len(r) for r in vglc_rows), default=0)

    if width < WINDOW_W or height == 0:
        return None

    # Pad height to WINDOW_H (top-pad with empty rows, same as create_level_json_data)
    pad_rows = max(0, WINDOW_H - height)
    padded   = ["-" * width] * pad_rows + list(vglc_rows)
    # Pad each row to full width
    padded   = [r.ljust(width, "-") for r in padded]

    best_score  = -1
    best_scene = None

    for x in range(width - WINDOW_W + 1):
        scene = []
        score  = 0
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
            best_score  = score
            best_scene = scene

    return best_scene

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    keyword:    str,
    max_levels: int,
    output:     str,
    save_ascii: str | None,
    save_vglc:  str | None,
):
    # -- imports that require optional dependencies -------------------------
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("ERROR: 'datasets' not installed.  Run: pip install datasets")
    try:
        from kaitaistruct import KaitaiStream
    except ImportError:
        sys.exit("ERROR: 'kaitaistruct' not installed.  Run: pip install kaitaistruct")

    # -- load sibling modules -----------------------------------------------
    viewer_mod  = _load_module("mm2_viewer",  "mm2_viewer.py")
    vglc_mod    = _load_module("mm2view_to_vglc", "mm2view_to_vglc.py")
    level_mod   = _load_module("level",       "level.py")

    Level         = level_mod.Level
    level_to_dict = viewer_mod.level_to_dict
    ASCII_MAP     = viewer_mod.ASCII_MAP
    obj_id_to_str = viewer_mod.obj_id_to_str

    tile_to_id = load_vglc_tileset()

    if save_ascii:
        Path(save_ascii).mkdir(parents=True, exist_ok=True)
    if save_vglc:
        Path(save_vglc).mkdir(parents=True, exist_ok=True)

    # -- stream from HuggingFace --------------------------------------------
    print(f"Streaming dataset  keyword={repr(keyword)}  max={max_levels}")
    ds = load_dataset("TheGreatRambler/mm2_level", streaming=True, split="train")
    if keyword:
        ds = ds.filter(lambda ex: keyword.lower() in ex["name"].lower())

    dataset   = []
    processed = 0
    skipped   = 0

    for ex in ds:
        if processed >= max_levels:
            break

        name = ex.get("name", f"level_{processed}")
        print(f"  [{processed+1}/{max_levels}] {name}", end="  ", flush=True)

        # 1. Parse binary level data
        try:
            raw = zlib.decompress(ex["level_data"])
            lv  = Level(KaitaiStream(BytesIO(raw)))
            lvl_dict = level_to_dict(lv, name)
        except Exception as e:
            print(f"PARSE ERROR: {e}")
            skipped += 1
            continue

        # 2. Build MM2 ASCII grid
        try:
            ascii_rows = build_ascii_grid(lvl_dict, ASCII_MAP, obj_id_to_str)
        except Exception as e:
            print(f"ASCII ERROR: {e}")
            skipped += 1
            continue

        if save_ascii:
            safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
            ascii_path = os.path.join(save_ascii, f"{safe_name}.txt")
            Path(ascii_path).write_text("\n".join(ascii_rows), encoding="utf-8")

        # 3. Convert ASCII → VGLC
        try:
            vglc_rows = vglc_mod.convert_level(ascii_rows)
        except Exception as e:
            print(f"VGLC ERROR: {e}")
            skipped += 1
            continue

        if save_vglc:
            safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
            vglc_path = os.path.join(save_vglc, f"{safe_name}_VGLC.txt")
            Path(vglc_path).write_text("\n".join(vglc_rows), encoding="utf-8")

        # 4. Extract best 20×20 window
        scene = best_window(vglc_rows, tile_to_id)
        if scene is None:
            print("SKIP (level too narrow for 20×20 window)")
            skipped += 1
            continue

        dataset.append({
            "name":   name,
            "scene": scene,
        })
        processed += 1
        print("OK")

    # -- write output -------------------------------------------------------
    output_path = Path("datasets") / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(dataset)} scenes to {output_path}  ({skipped} skipped)")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print("Done.")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download MM2 levels from HuggingFace and build a VGLC JSON dataset."
    )
    parser.add_argument("--keyword",    default="",    help="Filter levels by name keyword (case-insensitive). Empty = no filter.")
    parser.add_argument("--max_levels", type=int, default=50, help="Maximum number of levels to include.")
    parser.add_argument("--output",     required=True, help="Output JSON file path.")
    parser.add_argument("--save_ascii", default=None,  help="Optional: directory to save intermediate MM2 ASCII .txt files.")
    parser.add_argument("--save_vglc",  default=None,  help="Optional: directory to save intermediate VGLC .txt files.")
    args = parser.parse_args()

    run_pipeline(
        keyword    = args.keyword,
        max_levels = args.max_levels,
        output     = args.output,
        save_ascii = args.save_ascii,
        save_vglc  = args.save_vglc,
    )