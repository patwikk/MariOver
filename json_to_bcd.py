"""
Rebuild a .bcd course file from the JSON exported by Toost
(toost.exe --overworldJson / --subworldJson, see toost_stuff/batch_convert.py).

Background
----------
Toost's JSON export flattens two things into one object per file:
  - the level-wide header (name, description, gamestyle, clear conditions,
    timer, goal position, etc.)
  - the header + entity arrays for ONE map (overworld or subworld)

To rebuild a full Level (per level.ksy) we need both the *_overworld.json
and *_subworld.json for a level (the level-wide fields are duplicated in
both, so either can supply them). If only one is given, the other map is
written out as an empty/default map.

Field layout follows level.ksy exactly (376768-byte payload =
512-byte level header + two 188128-byte maps).

Caveats (lossy fields)
----------------------
Toost's JSON export does not include everything in the binary format, so
the following are written back as zero and will NOT match the original
.bcd byte-for-byte:
  - level header: year/month/day/hour/minute (creation date), unk1 (189
    bytes), unk2 (1 byte)
  - per object: unk1 (s2)
  - per map: unk_flag bits other than the "night_time" bit, unk1 (s4),
    unk2 padding (3516 bytes)
  - sounds, exclamation blocks, track blocks, icicles (no JSON arrays
    exist for these; their counts are written as 0)
  - clear pipe "unk" marker word (set to 1 for any pipe present in the
    JSON, 0 otherwise)

The result is a structurally valid, encrypted .bcd that Toost / SMM2 can
load, but it is a "best effort" reconstruction, not a perfect round trip.

Usage
-----
    python json_to_bcd.py bcd_levels/json/3000009_overworld.json
    python json_to_bcd.py bcd_levels/json/3000009_overworld.json -o out/3000009.bcd

    # Drop/clamp objects this build of toost can't render (see toost_compat.py),
    # so the resulting .bcd can be previewed with toost without crashing:
    python json_to_bcd.py bcd_levels/json/3000009_overworld.json --toost-compat
"""

import argparse
import json
import struct
from pathlib import Path

from extract_mm2_bcd import build_bcd, PAYLOAD_SIZE

# ---------------------------------------------------------------------------
# Fixed array sizes / element sizes, per level.ksy
# ---------------------------------------------------------------------------

OBJ_MAX           = 2600
SOUND_MAX         = 300
SNAKE_MAX         = 5
CLEAR_PIPE_MAX    = 200
PIRANHA_MAX       = 10
EXCLAMATION_MAX   = 10
TRACK_BLOCK_MAX   = 10
GROUND_MAX        = 4000
TRACK_MAX         = 1500
ICICLE_MAX        = 300

OBJ_SIZE          = 32   # x,y(s4*2) unk1(s2) w,h(u1*2) flag,cflag,ex(s4*3) id,cid,lid,sid(s2*4)
SOUND_SIZE        = 4
SNAKE_NODE_SIZE   = 8    # index,direction(u2*2) unk1(u4)
SNAKE_SIZE        = 4 + 120 * SNAKE_NODE_SIZE
CLEAR_PIPE_NODE_SIZE = 8  # type,index,x,y,width,height,unk1,direction (u1*8)
CLEAR_PIPE_SIZE   = 4 + 36 * CLEAR_PIPE_NODE_SIZE
PIRANHA_NODE_SIZE = 4    # unk1,direction(u1*2) unk2(u2)
PIRANHA_SIZE      = 4 + 20 * PIRANHA_NODE_SIZE
EXCLAMATION_SIZE  = 4 + 10 * 4
TRACK_BLOCK_SIZE  = 4 + 10 * 4
GROUND_SIZE       = 4
TRACK_SIZE        = 12   # unk1(u2) flags,x,y,type(u1*4) lid,unk2,unk3(u2*3)
ICICLE_SIZE       = 4
MAP_UNK2_SIZE     = 0xDBC  # 3516

LEVEL_HEADER_SIZE = 512
MAP_HEADER_SIZE   = 72


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def pack_str_utf16(s, size_bytes):
    raw = (s or "").encode("utf-16-le")
    # Toost reads these fields as null-terminated char16_t* strings, so
    # always leave room for a trailing u"\x00" even when truncating.
    max_bytes = size_bytes - 2
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
        if len(raw) % 2:
            raw = raw[:-1]
    return raw.ljust(size_bytes, b"\x00")


# ---------------------------------------------------------------------------
# Level header (512 bytes)
# ---------------------------------------------------------------------------

def pack_level_header(j):
    goal_x = j.get("goal_x", 0)
    goal_y = j.get("goal_y", 0)
    right_boundary = j.get("right_boundary", 0)
    # toost exports goal_x/goal_y as the raw s2/u1 binary values (goal_y in
    # whole tiles, goal_x in TENTHS of a tile). Some other exporters instead
    # report both in object-coordinate units (160 per tile), which overflows
    # goal_y's u1 field. Detect that case and rescale both back to raw units.
    if goal_y > 255:
        goal_x //= 160
        goal_y //= 160
    elif (
        right_boundary
        and goal_x // 10 > right_boundary // 16
        and goal_x // 160 <= right_boundary // 16
    ):
        # Some exporters leave goal_y in raw tile units but still report
        # goal_x in object-coordinate units (160 per tile, like
        # objects[].x) instead of tenths of a tile. If treating goal_x as
        # tenths-of-a-tile would place the goal past the level's right
        # boundary, but treating it as object-coordinate units would not,
        # rescale 160-per-tile -> 10-per-tile. (Otherwise this build's
        # toost reads a wildly out-of-range goal column and crashes, with
        # or without --toost-compat.)
        goal_x = (goal_x // 160) * 10

    fixed = struct.pack(
        "<BBhhhhbbbbBBiiiiiIqi",
        j.get("start_y", 0),
        goal_y,
        goal_x,
        j.get("timer", 0),
        j.get("clear_condition_magnitude", 0),
        0,  # year
        0,  # month
        0,  # day
        0,  # hour
        0,  # minute
        j.get("autoscroll_speed_raw", 0),
        j.get("clear_condition_category_raw", 0),
        j.get("clear_condition_type_raw", 0),
        j.get("gamever", 0),
        j.get("management_flags", 0),
        j.get("clear_attempts", 0),
        j.get("clear_time", 0),
        j.get("creation_id", 0),
        j.get("upload_id", 0),
        j.get("game_version_raw", 0),
    )
    out = (
        fixed
        + b"\x00" * 189  # unk1
        + struct.pack("<h", j.get("gamestyle_raw", 0))
        + b"\x00"  # unk2
        + pack_str_utf16(j.get("name", ""), 66)
        + pack_str_utf16(j.get("description", ""), 202)
    )
    assert len(out) == LEVEL_HEADER_SIZE, len(out)
    return out


# ---------------------------------------------------------------------------
# Map entities
# ---------------------------------------------------------------------------

def pack_obj(o):
    return struct.pack(
        "<iihBBiiihhhh",
        o.get("x", 0),
        o.get("y", 0),
        0,  # unk1, not exported by toost
        o.get("w", 0),
        o.get("h", 0),
        o.get("flag", 0),
        o.get("cflag", 0),
        o.get("ex", 0),
        o.get("id", 0),
        o.get("cid", -1),
        o.get("lid", -1),
        o.get("sid", -1),
    )


def pack_ground(g):
    return struct.pack("<BBBB", g.get("x", 0), g.get("y", 0), g.get("id", 0), g.get("bid", 0))


def pack_snake(s):
    nodes = s.get("nodes", [])
    out = struct.pack("<BBH", s.get("index", 0), s.get("node_count", len(nodes)), 0)
    for i in range(120):
        if i < len(nodes):
            n = nodes[i]
            out += struct.pack("<HHI", n.get("index", 0), n.get("direction", 0), 0)
        else:
            out += b"\x00" * SNAKE_NODE_SIZE
    return out


def pack_clear_pipe(cp):
    nodes = cp.get("nodes", [])
    # "unk" is used by toost as an "is this slot populated" marker; any
    # non-zero value works, so use 1 for pipes present in the JSON.
    out = struct.pack("<BBH", cp.get("index", 0), cp.get("node_count", len(nodes)), 1)
    for i in range(36):
        if i < len(nodes):
            n = nodes[i]
            out += struct.pack(
                "<BBBBBBBB",
                n.get("type", 0),
                n.get("index", 0),
                n.get("x", 0),
                n.get("y", 0),
                n.get("w", 0),
                n.get("h", 0),
                0,  # unk1
                n.get("direction", 0),
            )
        else:
            out += b"\x00" * CLEAR_PIPE_NODE_SIZE
    return out


def pack_piranha_creeper(c):
    nodes = c.get("nodes", [])
    out = struct.pack("<BBBB", 0, c.get("index", 0), c.get("node_count", len(nodes)), 0)
    for i in range(20):
        if i < len(nodes):
            out += struct.pack("<BBH", 0, int(nodes[i]), 0)
        else:
            out += b"\x00" * PIRANHA_NODE_SIZE
    return out


def pack_track(t):
    x = t.get("x", 0)
    y = t.get("y", 0)
    # Inverse of toost's TX==255 -> 0, else TX+1 transform.
    raw_x = 255 if x == 0 else (x - 1) & 0xFF
    raw_y = 255 if y == 0 else (y - 1) & 0xFF
    return struct.pack(
        "<HBBBBHHH",
        t.get("un", 0),
        t.get("flag", 0),
        raw_x,
        raw_y,
        t.get("type", 0),
        t.get("lid", 0),
        t.get("k0", 0),
        t.get("k1", 0),
    )


# ---------------------------------------------------------------------------
# Map (188128 bytes)
# ---------------------------------------------------------------------------

def _pack_array(items, max_count, size, pack_fn, label):
    if len(items) > max_count:
        raise ValueError(f"too many {label}: {len(items)} > {max_count}")
    out = bytearray()
    for i in range(max_count):
        out += pack_fn(items[i]) if i < len(items) else b"\x00" * size
    return bytes(out)


def _fix_object_anchors(objects, label="map"):
    """Real .bcd objects store x/y as the CENTER of their tile footprint:
        x = (left_col + w/2) * 160   (x % 160 == 80 for odd w, == 0 for even w)
        y = bottom_row * 160 + 80    (always, regardless of h)
    (Verified against bcd_levels/json/*_overworld.json: 1x1/2x2/4x4/8x1
    objects all follow this.)

    Some JSON exporters/generators instead place objects on a naive
    "x = col*160, y = row*160" grid with no center offset. toost still
    reads x/y -> tile via x//160 (-w//2) / y//160, so the object lands in
    the "right" tile, but is drawn 8px (half a tile) off the ground grid -
    visually "floating between tiles" instead of sitting on them.

    Detect the naive X convention from odd-width objects (where naive vs.
    real differ mod 160) and correct both axes.
    """
    odd_w = [o for o in objects if o.get("w", 1) % 2 == 1]
    x_naive = bool(odd_w) and sum(1 for o in odd_w if o.get("x", 0) % 160 == 0) > len(odd_w) // 2

    fixed = []
    n_x = n_y = 0
    for o in objects:
        o = dict(o)
        if x_naive:
            o["x"] = o.get("x", 0) + o.get("w", 1) * 80
            n_x += 1
        if o.get("y", 0) % 160 == 0:
            o["y"] = o.get("y", 0) + 80
            n_y += 1
        fixed.append(o)

    if n_x or n_y:
        print(f"  [{label}] toost-anchor fix: shifted {n_x} object(s) on X, "
              f"{n_y} object(s) on Y onto toost's tile-center grid")
    return fixed


def pack_map(j, label="map"):
    objects          = _fix_object_anchors(j.get("objects", []), label)
    ground           = j.get("ground", [])
    snakes           = j.get("snakes", [])
    clear_pipes      = j.get("clear_pipes", [])
    piranha_creepers = j.get("piranha_creepers", [])
    tracks           = j.get("track", [])

    night_time = j.get("night_time", False)

    header = struct.pack(
        "<BBBBBBBB",
        j.get("theme_raw", 0),
        j.get("autoscroll_type_raw", 0),
        j.get("boundary_type_raw", 0),
        j.get("orientation_raw", 0),
        j.get("liquid_end_height", 0),
        j.get("liquid_mode_raw", 0),
        j.get("liquid_speed_raw", 0),
        j.get("liquid_start_height", 0),
    ) + struct.pack(
        "<iiiiiiiiiiiiiiii",
        j.get("right_boundary", 0),
        j.get("top_boundary", 0),
        j.get("left_boundary", 0),
        j.get("bottom_boundary", 0),
        1 if night_time else 0,  # unk_flag
        len(objects),
        0,  # sound_effect_count (no sounds array in JSON)
        len(snakes),
        j.get("clear_pipe_count", len(clear_pipes)),
        len(piranha_creepers),
        0,  # exclamation_mark_block_count (no array in JSON)
        0,  # track_block_count (no array in JSON)
        0,  # unk1
        len(ground),
        len(tracks),
        0,  # ice_count (no icicles array in JSON)
    )
    assert len(header) == MAP_HEADER_SIZE, len(header)

    out = bytearray(header)
    out += _pack_array(objects, OBJ_MAX, OBJ_SIZE, pack_obj, "objects")
    out += b"\x00" * (SOUND_SIZE * SOUND_MAX)
    out += _pack_array(snakes, SNAKE_MAX, SNAKE_SIZE, pack_snake, "snakes")
    out += _pack_array(clear_pipes, CLEAR_PIPE_MAX, CLEAR_PIPE_SIZE, pack_clear_pipe, "clear_pipes")
    out += _pack_array(piranha_creepers, PIRANHA_MAX, PIRANHA_SIZE, pack_piranha_creeper, "piranha_creepers")
    out += b"\x00" * (EXCLAMATION_SIZE * EXCLAMATION_MAX)
    out += b"\x00" * (TRACK_BLOCK_SIZE * TRACK_BLOCK_MAX)
    out += _pack_array(ground, GROUND_MAX, GROUND_SIZE, pack_ground, "ground")
    out += _pack_array(tracks, TRACK_MAX, TRACK_SIZE, pack_track, "tracks")
    out += b"\x00" * (ICICLE_SIZE * ICICLE_MAX)
    out += b"\x00" * MAP_UNK2_SIZE

    return bytes(out)


# ---------------------------------------------------------------------------
# Top level: combine header + two maps into the full payload
# ---------------------------------------------------------------------------

def build_payload(overworld_json, subworld_json):
    header_source = overworld_json or subworld_json or {}
    payload = (
        pack_level_header(header_source)
        + pack_map(overworld_json or {}, label="overworld")
        + pack_map(subworld_json or {}, label="subworld")
    )
    assert len(payload) == PAYLOAD_SIZE, len(payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_companion(path: Path):
    """Given an *_overworld.json or *_subworld.json path, return
    (overworld_path, subworld_path, base_stem), with paths set to None
    if that file doesn't exist."""
    stem = path.stem
    if stem.endswith("_overworld"):
        base = stem[: -len("_overworld")]
        ow, sub = path, path.with_name(base + "_subworld" + path.suffix)
    elif stem.endswith("_subworld"):
        base = stem[: -len("_subworld")]
        ow, sub = path.with_name(base + "_overworld" + path.suffix), path
    else:
        base = stem
        ow, sub = path, None

    ow = ow if ow and ow.exists() else None
    sub = sub if sub and sub.exists() else None
    return ow, sub, base


def parse_args():
    p = argparse.ArgumentParser(
        description="Rebuild a .bcd course file from toost's JSON export."
    )
    p.add_argument("json_path", help="Path to a *_overworld.json or *_subworld.json file")
    p.add_argument("-o", "--output", help="Output .bcd path (default: <stem>.bcd next to the input)")
    p.add_argument("--toost-compat", action="store_true",
                   help="Drop/clamp objects toost's local sprite atlas can't render (see toost_compat.py)")
    p.add_argument("--leveldata", help="Path to toost's LevelData.hpp (used with --toost-compat)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    in_path = Path(args.json_path)
    ow_path, sub_path, base = find_companion(in_path)

    overworld_json = load_json(ow_path) if ow_path else None
    subworld_json = load_json(sub_path) if sub_path else None

    if overworld_json is None and subworld_json is None:
        raise SystemExit(f"Could not find JSON data at {in_path}")

    if overworld_json is None:
        print(f"  [WARN] no overworld JSON found, writing an empty overworld map")
    if subworld_json is None:
        print(f"  [WARN] no subworld JSON found, writing an empty subworld map")

    if args.toost_compat:
        import toost_compat

        leveldata_path = Path(args.leveldata) if args.leveldata else toost_compat.DEFAULT_LEVELDATA
        if not leveldata_path.exists():
            raise SystemExit(f"Could not find LevelData.hpp at {leveldata_path} (pass --leveldata)")

        constants, location_keys = toost_compat.parse_leveldata(leveldata_path)
        overworld_json = toost_compat.sanitize_map_json(overworld_json, constants, location_keys, "overworld")
        subworld_json = toost_compat.sanitize_map_json(subworld_json, constants, location_keys, "subworld")

    payload = build_payload(overworld_json, subworld_json)
    bcd_bytes = build_bcd(payload)

    out_path = Path(args.output) if args.output else in_path.with_name(base + ".bcd")
    out_path.write_bytes(bcd_bytes)
    print(f"Wrote {out_path} ({len(bcd_bytes)} bytes)")
