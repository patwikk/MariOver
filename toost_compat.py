"""
Best-effort "toost compatibility" pass for the JSON consumed by
json_to_bcd.py.

Background
----------
toost's renderer (src/LevelDrawer.cpp in TheGreatRambler/toost) looks up
object sprites in LevelData::ObjectLocation, an unordered_map keyed by
`OBJ_<id> | <gamestyle>`. That table doesn't have an entry for every
(object id, gamestyle) combination, and a handful of object types also
draw extra "continuation" sprites for tall/wide instances (e.g. Tree /
id 106 draws OBJ_106A for every tile of height beyond 4, and ALWAYS
draws OBJ_106B regardless of size). When toost hits a missing
combination it calls unordered_map::at() and crashes with
std::out_of_range.

This script reads the constant values and ObjectLocation table straight
out of LevelData.hpp and uses them to adjust an *_overworld.json /
*_subworld.json so this toost build won't crash trying to render it:
  - Objects whose base sprite (OBJ_<id> | gamestyle) isn't in the atlas
    are dropped (toost has no sprite for them at all here).
  - Trees (id 106): OBJ_106B is drawn unconditionally for every Tree, so
    if (OBJ_106B | gamestyle) isn't in the atlas the Tree is dropped
    entirely (this build's LevelData.hpp only has OBJ_106A/106B for
    SM3DW - SMB1/SMB3/SMW/NSMBU trees can't be rendered at all here).
    Otherwise, trees taller than 4 tiles are clamped to height 4 if the
    "extra trunk" sprite (OBJ_106A | gamestyle) isn't in the atlas.

This only affects how the level previews in THIS toost build - it does
not change anything about how SMM2 itself would play the level, and it
is not an exhaustive compatibility check (only LevelData::ObjectLocation
and the Tree special-case are covered).

Usage
-----
    # Inspect what would change (dry run):
    python toost_compat.py bcd_levels/json/foo_overworld.json

    # Write a sanitized copy:
    python toost_compat.py bcd_levels/json/foo_overworld.json -o fixed_overworld.json

    # Or just pass --toost-compat to json_to_bcd.py to apply this
    # automatically before building the .bcd.
"""

import argparse
import json
import re
from pathlib import Path

# Sibling checkout of TheGreatRambler/toost, e.g.
#   Documents/GitHub/MariOver   (this repo)
#   Documents/GitHub/toost      (toost source)
DEFAULT_LEVELDATA = Path(__file__).resolve().parent.parent / "toost" / "src" / "LevelData.hpp"

CONST_RE = re.compile(r"^\s*((?:OBJ_[A-Za-z0-9_]+)|SMB1|SMB3|SMW|NSMBU|SM3DW)\s*=\s*(\d+),", re.M)
LOCATION_SECTION_RE = re.compile(r"ObjectLocation\s*=\s*\{(.*?)\n\t\};", re.S)
LOCATION_ENTRY_RE = re.compile(r"\{\s*([A-Za-z0-9_| ]+?)\s*,\s*\{")

TREE_ID = 106
TREE_BASE_CONST = "OBJ_106"
TREE_TRUNK_CONST = "OBJ_106A"   # extra trunk sprite, drawn per tile of height beyond 4
TREE_LEAVES_CONST = "OBJ_106B"  # leaves sprite, drawn unconditionally for every tree
TREE_BASE_HEIGHT = 4

# This toost build allocates the canvas as (right_boundary/16)*16 pixels and
# segfaults (fixed-size buffer overrun in the grid/ground drawing code) once
# that exceeds ~4096px (256 tiles). Some JSON exporters report
# right_boundary/left_boundary/bottom_boundary in "object-coordinate" units
# (160 per tile, same as objects[].x/y) instead of toost's native pixel units
# (16 per tile) - dividing by 10 converts between the two scales. We only do
# this when the raw value would otherwise crash toost.
MAX_RENDER_WIDTH = 4096
BOUNDARY_SCALE_FACTOR = 10


def parse_leveldata(path):
    """Return (constants, location_keys) parsed from LevelData.hpp."""
    text = Path(path).read_text(encoding="utf-8")

    constants = {name: int(value) for name, value in CONST_RE.findall(text)}

    section = LOCATION_SECTION_RE.search(text).group(1)
    location_keys = set()
    for expr in LOCATION_ENTRY_RE.findall(section):
        names = [n.strip() for n in expr.split("|")]
        try:
            key = 0
            for n in names:
                key |= constants[n]
            location_keys.add(key)
        except KeyError:
            continue

    return constants, location_keys


def sanitize_map_json(map_json, constants, location_keys, label="map"):
    """Return a copy of map_json with objects toost can't draw fixed up."""
    if map_json is None:
        return None

    gamestyle_raw = map_json.get("gamestyle_raw", 0)
    out = dict(map_json)
    fixed_objects = []
    dropped = clamped = 0

    right_boundary = out.get("right_boundary", 0)
    if right_boundary > MAX_RENDER_WIDTH:
        if right_boundary % BOUNDARY_SCALE_FACTOR == 0 and right_boundary // BOUNDARY_SCALE_FACTOR <= MAX_RENDER_WIDTH:
            new_rb = right_boundary // BOUNDARY_SCALE_FACTOR
            print(f"  [{label}] toost-compat: rescaling right_boundary {right_boundary} -> {new_rb} "
                  f"(this toost build can't render courses wider than {MAX_RENDER_WIDTH}px)")
            out["right_boundary"] = new_rb
            out["left_boundary"] = out.get("left_boundary", 0) // BOUNDARY_SCALE_FACTOR
            out["bottom_boundary"] = out.get("bottom_boundary", 0) // BOUNDARY_SCALE_FACTOR
        else:
            print(f"  [{label}] toost-compat: clamping right_boundary {right_boundary} -> {MAX_RENDER_WIDTH} "
                  f"(this toost build can't render courses wider than {MAX_RENDER_WIDTH}px)")
            out["right_boundary"] = MAX_RENDER_WIDTH

    for obj in out.get("objects", []):
        obj_id = obj.get("id")
        base_name = f"OBJ_{obj_id}"

        if base_name in constants and (constants[base_name] | gamestyle_raw) not in location_keys:
            dropped += 1
            continue

        if obj_id == TREE_ID:
            # OBJ_106B (leaves) is drawn unconditionally for every tree; if
            # this gamestyle has no sprite for it, the tree can't be drawn
            # at all by this toost build.
            if (constants[TREE_LEAVES_CONST] | gamestyle_raw) not in location_keys:
                dropped += 1
                continue

            # OBJ_106A (extra trunk) is drawn once per tile of height beyond 4.
            if obj.get("h", 0) > TREE_BASE_HEIGHT and (constants[TREE_TRUNK_CONST] | gamestyle_raw) not in location_keys:
                obj = dict(obj)
                obj["h"] = TREE_BASE_HEIGHT
                clamped += 1

        fixed_objects.append(obj)

    if dropped or clamped:
        print(f"  [{label}] toost-compat: dropped {dropped} object(s), clamped {clamped} object(s)")

    out["objects"] = fixed_objects
    out["object_count"] = len(fixed_objects)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_path", help="Path to a *_overworld.json or *_subworld.json file")
    p.add_argument("-o", "--output", help="Where to write the sanitized JSON (default: dry run, no output)")
    p.add_argument("--leveldata", default=str(DEFAULT_LEVELDATA),
                   help=f"Path to toost's LevelData.hpp (default: {DEFAULT_LEVELDATA})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    leveldata_path = Path(args.leveldata)
    if not leveldata_path.exists():
        raise SystemExit(f"Could not find LevelData.hpp at {leveldata_path} (pass --leveldata)")

    constants, location_keys = parse_leveldata(leveldata_path)

    in_path = Path(args.json_path)
    map_json = json.loads(in_path.read_text(encoding="utf-8"))
    fixed = sanitize_map_json(map_json, constants, location_keys, label=in_path.stem)

    if args.output:
        Path(args.output).write_text(json.dumps(fixed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print("Dry run (pass -o to write a sanitized copy)")
