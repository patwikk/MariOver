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
  4. Slide a 20x20 window across each VGLC level and emit one scene per
     level (the single window with the most non-empty content).
  5. Write everything to a JSON dataset file ready for diffusion model
     training, using integer tile IDs from smb.json (VGLC tileset).
  6. Optionally: generate text captions, split into train/val/test, and create
     random test captions -- producing a dataset ready for train_diffusion.py.

Usage
-----
    # Plain scene dataset only:
    python build_dataset.py --keyword mario --max_levels 100 --output dataset.json

    # Full captioned pipeline (caption -> split -> random test captions):
    python build_dataset.py --keyword mario --max_levels 100 --output dataset.json \
        --caption --exclude_upside_down_pipes --seed 0

    # Save intermediate ASCII/VGLC files too:
    python build_dataset.py --keyword mario --max_levels 50 --output dataset.json \
        --caption --save_ascii ./ascii_levels --save_vglc ./vglc_levels

Required files (same directory as this script)
---------------
    level.py                        Kaitai-compiled MM2 level parser
    mm2_viewer.py                   Source of level_to_dict and ASCII_MAP
    mm2view_to_vglc.py              Our ASCII->VGLC converter
    smb.json                        VGLC tileset (defines tile->id mapping)
    create_ascii_captions.py        Caption generator          (--caption only)
    create_random_test_captions.py  Random test captions       (--caption only)
    captions/                       Caption utility package    (--caption only)
    util/common_settings.py         Shared settings            (--caption only)
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
# Caption generation  (create_ascii_captions.py)
# ---------------------------------------------------------------------------
def generate_captions_inline(
    dataset:                   list,
    tileset_path:              str,
    exclude_upside_down_pipes: bool = False,
    include_broken:            bool = False,
) -> list:
    """
    Run create_ascii_captions.assign_caption on each scene in the dataset.

    Always uses regular (non-absence) captions, matching the SMB1-data.bat
    regular pipeline. Returns a new list of
      {"name":..., "scene":..., "caption":...}
    dicts, with broken-pipe/cannon scenes excluded unless include_broken=True.
    """
    captions_mod = _load_module("create_ascii_captions", "create_ascii_captions.py")

    from captions.util import extract_tileset
    tile_chars, id_to_char, char_to_id, tile_descriptors = extract_tileset(tileset_path)

    exclude_broken = not include_broken
    captioned = []
    num_excluded = 0

    for entry in dataset:
        scene   = entry["scene"]
        caption = captions_mod.assign_caption(
            scene,
            id_to_char,
            char_to_id,
            tile_descriptors,
            describe_locations=False,
            describe_absence=False,               # always regular captions
            exclude_upside_down_pipes=exclude_upside_down_pipes,
        )

        if exclude_broken and "broken" in caption:
            if "broken pipe" in caption:
                print(f"  [caption] broken pipe -- excluding: {entry['name']}")
            if "broken cannon" in caption:
                print(f"  [caption] broken cannon -- excluding: {entry['name']}")
            num_excluded += 1
            continue

        captioned.append({
            "name":    entry["name"],
            "scene":   scene,
            "caption": caption,
        })

    print(f"  [caption] {len(captioned)} captioned, {num_excluded} excluded (broken artifacts).")
    return captioned


# ---------------------------------------------------------------------------
# Tokenizer step  (tokenizer.py)
# ---------------------------------------------------------------------------
def build_tokenizer(captioned_json_path: str, pkl_path: str):
    """
    Build and save a vocabulary tokenizer from a captioned JSON file,
    equivalent to: python tokenizer.py save --json_file ... --pkl_file ...
    """
    tokenizer_mod = _load_module("tokenizer", "tokenizer.py")
    tok = tokenizer_mod.Tokenizer()
    tok.build_vocab(captioned_json_path)
    tok.save(pkl_path)
    print(f"  [tokenizer] vocab size {tok.get_vocab_size()}, saved to {pkl_path}")
    return tok


# ---------------------------------------------------------------------------
# Split step
# ---------------------------------------------------------------------------
def split_dataset(captioned_json_path: str, seed: int, train_pct: float = 0.9,
                  val_pct: float = 0.05, test_pct: float = 0.05):
    """
    Split the captioned JSON into train/validate/test files with mario coverage
    verification. Inlined from split_data.py to avoid its LR imports.
    Produces <stem>-train.json, <stem>-validate.json, <stem>-test.json.
    Returns (train_path, val_path, test_path) as strings.
    """
    import random as _random
    from captions.caption_match import TOPIC_KEYWORDS as MARIO_TOPIC_KEYWORDS

    _random.seed(seed)

    with open(captioned_json_path, "r") as f:
        full_dataset = json.load(f)

    # Determine required structures for coverage check (mario/regular only)
    required_structures = [kw for kw in MARIO_TOPIC_KEYWORDS if "broken" not in kw]
    has_upside_down = any(
        "upside down pipe" in entry.get("caption", "").lower()
        for entry in full_dataset
    )
    if not has_upside_down:
        required_structures = [kw for kw in required_structures if "upside down pipe" not in kw]

    # Initial shuffle + split
    def do_split(data):
        _random.shuffle(data)
        n = len(data)
        n_train = int(train_pct * n)
        n_val   = int(val_pct * n)
        return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]

    def check_coverage(split):
        found = {s: False for s in required_structures}
        for entry in split:
            caption = entry.get("caption", "").lower()
            for s in required_structures:
                if s in caption:
                    found[s] = True
        return found

    def find_and_swap(source, target, missing):
        for i, entry in enumerate(source):
            if missing in entry.get("caption", "").lower():
                target.append(source.pop(i))
                return True
        return False

    train_split, val_split, test_split = do_split(list(full_dataset))
    splits = {"train": train_split, "val": val_split, "test": test_split}

    # Coverage verification: single pass per split.
    # A while loop risks infinite cycling when swapping depletes the donor split,
    # so we do one forward pass only -- each structure just needs to exist once.
    for split_name, split in splits.items():
        for structure, present in check_coverage(split).items():
            if not present:
                for other_name, other in splits.items():
                    if other_name != split_name:
                        if find_and_swap(other, split, structure):
                            print(f"  [split] swapped '{structure}' from {other_name} -> {split_name}")
                            break

    base, ext = os.path.splitext(captioned_json_path)
    train_path = f"{base}-train{ext}"
    val_path   = f"{base}-validate{ext}"
    test_path  = f"{base}-test{ext}"

    with open(train_path, "w") as f:
        json.dump(splits["train"], f, indent=2)
    with open(val_path, "w") as f:
        json.dump(splits["val"], f, indent=2)
    with open(test_path, "w") as f:
        json.dump(splits["test"], f, indent=2)

    print(f"  [split] train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")
    print(f"    {train_path}")
    print(f"    {val_path}")
    print(f"    {test_path}")
    return train_path, val_path, test_path


# ---------------------------------------------------------------------------
# Random test captions step  (create_random_test_captions.py)
# ---------------------------------------------------------------------------
def create_random_test_captions(captioned_json_path: str, save_path: str,
                                seed: int, exclude_upside_down_pipes: bool,
                                n: int = 100):
    """
    Generate n grammar-based random captions that don't duplicate any caption
    already in the dataset, equivalent to:
      python create_random_test_captions.py --json ... --save_file ... \
          --seed ... --no_upside_down_pipes
    """
    rtc_mod = _load_module("create_random_test_captions", "create_random_test_captions.py")
    from captions.caption_generator import GrammarGenerator
    from captions.caption_match import compare_captions
    from level_dataset import LevelDataset

    generator = GrammarGenerator(
        seed=seed,
        describe_absence=False,
        no_upside_down_pipes=exclude_upside_down_pipes,
    )

    dataset = LevelDataset(
        json_path=captioned_json_path,
        tokenizer=None,
        shuffle=True,
        mode="text",
        augment=False,
    )

    validation_captions = []
    while len(validation_captions) < n:
        new_caption = generator.generate_sentence()
        caption_is_new = True
        for caption in dataset:
            if compare_captions(caption, new_caption) == 1.0:
                caption_is_new = False
                break
        if not caption_is_new:
            continue
        validation_captions.append(new_caption)
        if len(validation_captions) % 10 == 0:
            print(f"  [random test captions] {len(validation_captions)}/{n}")

    out = [{"caption": c} for c in validation_captions]
    with open(save_path, "w") as f:
        json.dump(out, f, indent=4)
    print(f"  [random test captions] saved {len(out)} captions to {save_path}")


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
# scene extraction  (one best 20x20 window per level)
# ---------------------------------------------------------------------------
WINDOW_H = 20
WINDOW_W = 20

def best_window(vglc_rows: list[str], tile_to_id: dict) -> list[list[int]] | None:
    """
    Slide a WINDOW_H x WINDOW_W window across the VGLC level and return
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

    pad_rows = max(0, WINDOW_H - height)
    padded   = ["-" * width] * pad_rows + list(vglc_rows)
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
    caption:    bool = False,
    tileset:    str | None = None,
    exclude_upside_down_pipes: bool = False,
    include_broken:            bool = False,
    seed:       int  = 0,
    train_pct:  float = 0.9,
    val_pct:    float = 0.05,
    test_pct:   float = 0.05,
    random_test_captions_n: int = 100,
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

        # 3. Convert ASCII -> VGLC
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

        # 4. Extract best 20x20 window
        scene = best_window(vglc_rows, tile_to_id)
        if scene is None:
            print("SKIP (level too narrow for 20x20 window)")
            skipped += 1
            continue

        dataset.append({
            "name":  name,
            "scene": scene,
        })
        processed += 1
        print("OK")

    # -- write plain output -------------------------------------------------
    output_path = Path("datasets") / output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting {len(dataset)} scenes to {output_path}  ({skipped} skipped)")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    # -- optional: full captioned pipeline ----------------------------------
    if caption:
        tileset_path = tileset or VGLC_TILESET_PATH
        if not os.path.isfile(tileset_path):
            sys.exit(f"ERROR: tileset file not found at {tileset_path!r}")

        # Derive base name for all caption-pipeline outputs
        stem         = output_path.stem
        caption_path = output_path.with_name(f"{stem}_captioned.json")
        pkl_path     = str(output_path.with_name(f"{stem}_Tokenizer-regular.pkl"))
        rtc_path     = str(output_path.with_name(f"{stem}_RandomTest-regular.json"))

        # Step 6a: caption
        print(f"\nGenerating captions -> {caption_path}")
        captioned = generate_captions_inline(
            dataset,
            tileset_path=tileset_path,
            exclude_upside_down_pipes=exclude_upside_down_pipes,
            include_broken=include_broken,
        )
        with open(caption_path, "w", encoding="utf-8") as f:
            json.dump(captioned, f, indent=2)
        print(f"  Captioned dataset: {caption_path}")

        # Step 6b: tokenize
        print(f"\nBuilding tokenizer -> {pkl_path}")
        build_tokenizer(str(caption_path), pkl_path)

        # Step 6c: split into train / validate / test
        print(f"\nSplitting dataset (train={train_pct} val={val_pct} test={test_pct}) ...")
        train_path, val_path, test_path = split_dataset(
            str(caption_path), seed=seed,
            train_pct=train_pct, val_pct=val_pct, test_pct=test_pct,
        )

        # Step 6d: random test captions
        print(f"\nGenerating {random_test_captions_n} random test captions -> {rtc_path}")
        create_random_test_captions(
            captioned_json_path=str(caption_path),
            save_path=rtc_path,
            seed=seed,
            exclude_upside_down_pipes=exclude_upside_down_pipes,
            n=random_test_captions_n,
        )

        print(f"\nCaption pipeline complete.")
        print(f"  Captioned JSON : {caption_path}")
        print(f"  Tokenizer pkl  : {pkl_path}")
        print(f"  Train split    : {train_path}")
        print(f"  Val split      : {val_path}")
        print(f"  Test split     : {test_path}")
        print(f"  Random test    : {rtc_path}")

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
    parser.add_argument("--output",     required=True, help="Output JSON filename (placed in datasets/ folder).")
    parser.add_argument("--save_ascii", default=None,  help="Optional: directory to save intermediate MM2 ASCII .txt files.")
    parser.add_argument("--save_vglc",  default=None,  help="Optional: directory to save intermediate VGLC .txt files.")
    parser.add_argument("--seed",       type=int, default=0, help="Random seed (used for dataset split).")

    # -- captioning ----------------------------------------------------------
    cap = parser.add_argument_group(
        "captioning pipeline",
        "When --caption is set, runs the full pipeline: caption -> "
        "train/val/test split -> random test captions."
    )
    cap.add_argument(
        "--caption", action="store_true", default=False,
        help="Enable the full captioning pipeline.",
    )
    cap.add_argument(
        "--tileset", default=None,
        help="Path to the VGLC tileset JSON (default: smb.json next to this script).",
    )
    cap.add_argument(
        "--exclude_upside_down_pipes", action="store_true", default=False,
        help="Omit upside-down pipe mentions from captions (recommended for MM2 data).",
    )
    cap.add_argument(
        "--include_broken", action="store_true", default=False,
        help="Keep scenes with broken pipes/cannons in captions (excluded by default).",
    )
    cap.add_argument(
        "--train_pct", type=float, default=0.9,
        help="Fraction of captioned data for training split (default: 0.9).",
    )
    cap.add_argument(
        "--val_pct", type=float, default=0.05,
        help="Fraction of captioned data for validation split (default: 0.05).",
    )
    cap.add_argument(
        "--test_pct", type=float, default=0.05,
        help="Fraction of captioned data for test split (default: 0.05).",
    )
    cap.add_argument(
        "--random_test_captions_n", type=int, default=100,
        help="Number of grammar-based random test captions to generate (default: 100).",
    )

    args = parser.parse_args()

    run_pipeline(
        keyword    = args.keyword,
        max_levels = args.max_levels,
        output     = args.output,
        save_ascii = args.save_ascii,
        save_vglc  = args.save_vglc,
        seed       = args.seed,
        caption                   = args.caption,
        tileset                   = args.tileset,
        exclude_upside_down_pipes = args.exclude_upside_down_pipes,
        include_broken            = args.include_broken,
        train_pct  = args.train_pct,
        val_pct    = args.val_pct,
        test_pct   = args.test_pct,
        random_test_captions_n = args.random_test_captions_n,
    )