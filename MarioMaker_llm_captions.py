#!/usr/bin/env python3
"""
MarioMaker_llm_captions.py
==========================
Uses a local Ollama LLM to generate rich captions for Mario Maker 2
ASCII level scenes, replacing the simple tile-presence captions.

Supports resume: if the output file already exists, previously captioned
items are skipped and the run continues from where it left off.
Progress is saved every 10 new captions so an interrupted run loses minimal work.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

# ── Tile character → human-readable name ─────────────────────────────────────

EXTENDED_CHAR_NAMES = {
    "#": "Ground",
    "B": "Brick",
    "?": "Question Block",
    "c": "Coin",
    "g": "Enemy",
    "K": "Koopa",
    "P": "Piranha Plant",
    "t": "Thwomp",
    "^": "Spike",
    "N": "Block",
    "T": "Mushroom Platform",
    "=": "Bridge",
    "k": "Semisolid Platform",
    "S": "Stone",
    "i": "Fire Flower",
    "V": "Cannon",
    "|": "Pipe",
    "↑": "Pipe",
    "↓": "Pipe",
    "←": "Pipe",
    "→": "Pipe",
}

MM2_CHAR_NAMES = {
    " ": "Air",
    "#": "Ground",
    "B": "Brick Block",
    "H": "Hard Block",
    "?": "Question Block",
    "h": "Hidden Block",
    "N": "Note Block",
    "d": "Donut Block",
    "I": "Ice Block",
    "p": "P Block",
    "O": "On/Off Block",
    ".": "Dotted-Line Block",
    "*": "Blinking Block",
    "^": "Spike Block",
    "C": "Crate",
    "S": "Stone Block",
    "{": "Starting Brick",
    "=": "Castle Bridge",
    "T": "Tree",
    "/": "Slight Slope",
    "\\": "Steep Slope",
    "|": "Pipe",
    "↑": "Pipe (Up)",
    "↓": "Pipe (Down)",
    "←": "Pipe (Left)",
    "→": "Pipe (Right)",
    "D": "Door",
    "W": "Warp Box",
    "k": "Key",
    "f": "Checkpoint Flag",
    "G": "Goal",
    "c": "Clear Pipe",
    "g": "Goomba",
    "K": "Koopa Troopa",
    "P": "Piranha Plant",
    "m": "Hammer Bro",
    "t": "Thwomp",
    "o": "Bob-omb",
    "s": "Spiny",
    "b": "Buzzy Beetle",
    "L": "Lakitu",
    "l": "Lakitu's Cloud",
    "Z": "Banzai Bill",
    "V": "Bullet Bill Blaster",
    "y": "Magikoopa",
    "<": "Spike Top",
    "u": "Boo",
    "X": "Bowser",
    "x": "Bowser Jr.",
    "@": "Chain Chomp",
    "~": "Cheep Cheep",
    "q": "Blooper",
    "w": "Wiggler",
    "Y": "Pokey",
    "e": "Piranha Creeper",
    "F": "Porcupuffer",
    "%": "Fish Bone",
    "&": "Lava Bubble",
    "r": "Rocky Wrench",
    ",": "Muncher",
    "a": "Ant Trooper",
    "n": "Monty Mole",
    "R": "Mechakoopa",
    "!": "Boom Boom",
    "9": "Dry Bones",
    "j": "Skipsqueak",
    "+": "Cinobio",
    "\xa1": "Cinobic",
    ";": "Stingby",
    "A": "Angry Sun",
    "v": "Charvaargh",
    "[": "Bully",
    "1": "Lemmy Koopa",
    "2": "Morton Koopa Jr.",
    "3": "Larry Koopa",
    "4": "Wendy O. Koopa",
    "5": "Iggy Koopa",
    "6": "Roy Koopa",
    "7": "Ludwig von Koopa",
    # Style Ride slot (id 45): gamestyle-dependent (Yoshi's Egg in
    # SMW/NSMBU); scenes carry no gamestyle, so the SMB1/SMB3 name is
    # used here as a baseline. See mm2_json_field_dictionary.txt §6.
    "\xb5": "Goomba's Shoe",
    "\xa2": "Coin",
    "$": "Red Coin",
    "\xa3": "Big Coin",
    "U": "1-Up Mushroom",
    "i": "Fire Flower",
    "\xa4": "Super Star",
    "M": "Super Mushroom",
    # Style Power-up slots A/B: gamestyle-dependent (Super Leaf/Cape Feather/
    # Propeller Mushroom, Frog Suit/Power Balloon/Super Acorn); scenes carry
    # no gamestyle, so the SMB1 names are used here as a baseline. See
    # mm2_json_field_dictionary.txt §5.
    "\xb6": "Big Mushroom",
    "\xa7": "SMB2 Mushroom",
    "\xac": "Super Hammer",
    "\xa6": "P Switch",
    "\xaf": "POW Block",
    "\xb1": "Spring",
    "]": "Cannon Box",
    "}": "Propeller Box",
    ")": "Goomba Mask",
    "\xb0": "Bullet Bill Mask",
    "\xb2": "Red POW Box",
    "-": "Lift",
    "\xb3": "Mushroom Platform",
    "\xb4": "Semisolid Platform",
    "\xb7": "Bridge",
    "\xb8": "Lava Lift",
    "\xb9": "Snake Block",
    "\xba": "Track Block",
    "\xbb": "Conveyor Belt",
    "\xbc": "Fast Conveyor Belt",
    "\xbd": "Sprint Platform",
    "\xbe": "Seesaw",
    "\xbf": "Swinging Claw",
    "\xc0": "On/Off Trampoline",
    "\xc1": "Trampoline",
    "J": "Jumping Machine",
    "\xc2": "Half-Collision Platform",
    "\xc3": "Donut Block Platform",
    "\xc4": "Fire Bar",
    "\xc5": "Saw",
    "\xc6": "Burner",
    "\xc7": "Spike Trap",
    "\xc8": "Spike Ball",
    "\xc9": "Skewer",
    "\xca": "Twister",
    "\xcb": "Icicle",
    "\xd8": "Cannon",
    "\xcc": "Cloud",
    "\xcd": "Vine",
    "\xce": "Water",
    "\xcf": "Arrow",
    "\xd0": "One-Way Wall",
    "\xd1": "Reel Camera",
    "\xd2": "Sound Effect",
    "\xd3": "Player Spawn",
    "\xd4": "Clown Car",
    "\xd5": "Koopa Car",
    "\xd6": "Track",
    "\xd7": "Starting Arrow",
    "\xd9": "Exclamation Block",
}

# ── Terrain character sets (solid tiles used for height/gap analysis) ─────────

TERRAIN_CHARS_MM2 = frozenset({"#", "H", "B", "S", "I", "C", "/", "\\"})
TERRAIN_CHARS_EXT = frozenset({"#", "B", "N", "S"})
TERRAIN_CHARS_WE = frozenset({"#", "B", "N", "?", "H", "I", "p", "O"})

# ── Prompt template ───────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are an expert Mario Maker 2 level describer.

You will receive three inputs:
1. A symbol dictionary mapping level grid symbols to Mario Maker 2 objects.
2. Pre-computed level metadata (RAW object tile counts, terrain column heights, floor/ceiling analysis, region boundaries).
3. A level grid (read top-to-bottom, left-to-right).

The metadata's object tile counts are raw occupied-cell counts, NOT deduplicated object counts. Many objects are single placed instances that render as a block of several identical cells — counting those cells directly will overcount how many objects are actually there. See MULTI-TILE OBJECTS below for how to interpret these counts and the grid correctly. Trust the metadata for terrain heights. Do not re-count terrain tiles from the grid, and never mention a feature that is not backed by the metadata or grid.

Your output trains a text-to-level diffusion model. The model needs many different human-style ways of describing the same level, so it can recognize a level from a short tag list, a casual sentence, or a detailed paragraph alike.

MULTI-TILE OBJECTS

Many object types occupy more than one grid cell per placed instance, appearing as a contiguous block of identical cells. Treat one contiguous block of the same tile type as ONE object, not one object per cell — unless the block's shape clearly looks like several same-size chunks repeated side by side, in which case count each repeated chunk as its own object.

This applies to (at least): pipes, bullet bill blasters, bridges, lifts, mushroom platforms, semisolid platforms, snake blocks, track blocks, conveyor belts, sprint platforms, half-collision platforms, donut block platforms, clouds, lava lifts, thwomps, angry suns, skewers, goal poles/flagpoles, vines, and fire bars.

If two blocks of the same tile type are not touching, they are separate objects. If you are uncertain whether a block is one object or several placed side by side, use your own knowledge of what sizes/dimensions that object can actually take in the Mario Maker 2 editor to judge the most plausible grouping, rather than guessing blindly. When still in doubt, prefer the more conservative (smaller) count.

WHAT YOU MAY DESCRIBE

Only describe concepts that are actually present in the metadata's object tile counts or terrain analysis, or that you can directly see in the grid using the symbol dictionary. Do not invent features. Concepts you may draw on, when present:

- terrain shape (floor, ceiling, gaps, hills, staircases, slopes, platforms, bridges, walls, pillars, towers, chambers, enclosed rooms, etc.) and its direction (ascending/descending) when relevant
- region structure (separate sections, dividers, left/center/right layout)
- block materials (ground, hard block, brick, ice block, note block, stone, etc.) when a structure is clearly made of one recognizable material
- pipes, doors, warps, and other traversal elements
- enemies and hazards, ideally with where they are (on the ground, on a platform, on a pipe, in the air, or which part of the level)
- collectibles and power-ups, ideally with placement
- rough quantities (one, two, three, a few, several, many) rather than exact tile counts for anything above three

You have free rein over how to organize and phrase this — which features to lead with, how to group them, and how much structural vocabulary to use. You do not need to follow a fixed checklist or priority order; just describe what's actually there in a way a person might.

WHAT TO AVOID

- Do not discuss gameplay, difficulty, quality, fun, or designer intent.
- Do not address the player or use second person.
- Do not describe exact coordinates or tile-by-tile detail.
- Do not mention features that are absent.

YOUR TASK

Write 5 different captions for this same level. All 5 must be accurate, but they must vary WIDELY from each other in length, level of detail, and register, so that together they cover the range of ways a human might describe this level:

- one or two should be terse, tag-like phrases separated by periods (similar to keyword lists), covering only the 2-4 most prominent features
- one or two should be a plain casual sentence or two, in normal prose, that a person might type quickly
- one or two should be a more detailed, descriptive paragraph that walks through the level's layout and notable features in order

Across the 5 captions, vary which features get emphasized — they don't all need to mention everything, but none should contradict another or invent something not present. Keep all 5 lowercase except for proper nouns inherent to object names if any.

OUTPUT FORMAT

Output a single JSON array of exactly 5 strings, and nothing else — no markdown fences, no commentary, no keys other than the array itself.

Example shape (do not reuse this content, it is only to show the format):
["ascending staircase. two goombas. one pipe right.", "a short level with a rising staircase, a couple of goombas, and a pipe near the end.", "the level begins on flat ground before climbing a set of steps toward the right side. two goombas patrol the lower section, and a pipe sits near the far right edge of the level.", "...", "..."]

Symbol dictionary:
{dict_string}


Metadata:
{metadata}


{grid_label}:
{ascii_grid}

Write the JSON array of 5 captions now. DO NOT INCLUDE ANY NON-ENGLISH CHARACTERS, and do not include anything outside the JSON array."""


# ── Core helpers ──────────────────────────────────────────────────────────────

def build_id_to_char(tileset_path):
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tile_chars = sorted(data["tiles"].keys())
    if "_" not in tile_chars:
        tile_chars.append("_")
    return {idx: char for idx, char in enumerate(tile_chars)}


# Category tags that describe how a tile behaves, not what it is. When
# picking a display name from a tileset's tag list, the last tag that is
# NOT one of these is used, since the truly identifying tag (e.g. "pipe")
# is not always written last (e.g. '|' = ["solid", "pipe", "warp"] would
# otherwise resolve to "Warp" instead of "Pipe").
GENERIC_TILE_TAGS = frozenset({
    "passable", "solid", "empty", "enemy", "damaging", "hazard", "moving",
    "flying", "boss", "explosive", "projectile", "falling", "slippery",
    "togglable", "collectable", "power-up", "platform", "vehicle",
    "style ride", "style power-up", "interactive", "climbable", "shooter",
    "warp", "decoration", "spawn",
})


def derive_char_names(tileset_path):
    """Build a char->name dict straight from a tileset's own tags.

    Used for tilesets (like mm2_tileset_we.json) that don't have a curated
    *_CHAR_NAMES dict, since their characters don't line up with MM2_CHAR_NAMES
    (e.g. "c" = coin here vs. "Clear Pipe" in mm2_tileset_full.json's char set).
    """
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    names = {}
    for char, tags in data["tiles"].items():
        if not tags:
            continue
        specific = [t for t in tags if t not in GENERIC_TILE_TAGS]
        chosen = specific[-1] if specific else tags[-1]
        names[char] = chosen.replace("-", " ").title()
    return names


def get_char_names(tileset_path):
    basename = os.path.basename(tileset_path)
    if "extended_tiles" in basename:
        return EXTENDED_CHAR_NAMES
    if "tileset_we" in basename:
        return derive_char_names(tileset_path)
    return MM2_CHAR_NAMES


def build_dict_string(tileset_path, char_names):
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for char, tags in data["tiles"].items():
        name = char_names.get(char)
        if not name:
            name = tags[-1].replace("-", " ").title() if tags else "Unknown"
        lines.append(f"'{char}' = {name}")
    return "\n".join(lines)


def compute_metadata(scene, id_to_char, char_names, tileset_path):
    """Pre-compute level metadata to anchor the LLM's terrain and object descriptions."""
    if not scene or not scene[0]:
        return "No metadata available."

    grid = [[id_to_char.get(tid, " ") for tid in row] for row in scene]
    nrows = len(grid)
    ncols = len(grid[0])

    basename = os.path.basename(tileset_path)
    if "extended_tiles" in basename:
        terrain = TERRAIN_CHARS_EXT
    elif "tileset_we" in basename:
        terrain = TERRAIN_CHARS_WE
    else:
        terrain = TERRAIN_CHARS_MM2

    parts = []

    # Raw occupied-cell counts per tile type (skip Air and Ground). Multi-tile
    # objects are deduplicated in the prompt, not here — see MULTI-TILE OBJECTS.
    counts = {}
    for row in grid:
        for ch in row:
            if ch == " ":
                continue
            name = char_names.get(ch)
            if not name or name in ("Air", "Ground"):
                continue
            counts[name] = counts.get(name, 0) + 1

    if counts:
        parts.append("Object tile counts (raw occupied cells per type, see prompt notes on multi-tile objects):")
        for name, cnt in sorted(counts.items(), key=lambda x: -x[1])[:20]:
            parts.append(f"  {name}: {cnt}")

    # Terrain column height profile: for each column, height of topmost solid tile from bottom
    col_heights = []
    for c in range(ncols):
        h = 0
        for r in range(nrows - 1, -1, -1):
            if grid[r][c] in terrain:
                h = nrows - r  # 1 = bottom row, nrows = top row
                break
        col_heights.append(h)

    parts.append("\nTerrain top-of-column heights (left to right, 0=no terrain in column):")
    for start in range(0, ncols, 10):
        chunk = col_heights[start: start + 10]
        end = start + len(chunk)
        parts.append(f"  cols {start + 1:02d}-{end:02d}: {' '.join(str(h) for h in chunk)}")

    # Floor analysis: does the level have a continuous floor and where are the gaps?
    def col_has_floor(c):
        return any(grid[r][c] in terrain for r in range(nrows - 3, nrows))

    floor_mask = [col_has_floor(c) for c in range(ncols)]
    has_floor = sum(floor_mask) > ncols * 0.35

    gaps = []
    in_gap = False
    gap_start = 0
    for c in range(ncols):
        if not floor_mask[c] and not in_gap:
            in_gap = True
            gap_start = c + 1  # 1-indexed
        elif floor_mask[c] and in_gap:
            in_gap = False
            gaps.append(f"cols {gap_start}-{c}")
    if in_gap:
        gaps.append(f"cols {gap_start}-{ncols}")

    floor_str = "present" if has_floor else "absent"
    if gaps:
        floor_str += f", gaps at: {', '.join(gaps)}"
    parts.append(f"\nFloor: {floor_str}")

    # Ceiling analysis
    ceiling_count = sum(1 for c in range(ncols) if grid[0][c] in terrain)
    parts.append(f"Ceiling: {'present' if ceiling_count > ncols * 0.2 else 'absent'}")

    # Explicit region boundaries for position labeling
    t = ncols // 3
    parts.append(
        f"\nRegion boundaries (use these when assigning left/center/right):"
        f" left=cols 1-{t}, center=cols {t+1}-{2*t}, right=cols {2*t+1}-{ncols}"
    )

    return "\n".join(parts)


def scene_to_ascii(scene, id_to_char):
    return "\n".join(
        "".join(id_to_char.get(tid, "?") for tid in row)
        for row in scene
    )


# ── T0x token format (based on --tileset-we) ──────────────────────────────────

def build_char_to_token(id_to_char):
    """Map each tile character to a 'T<NN>' token, NN = its numeric tile ID."""
    width = max(2, len(str(len(id_to_char) - 1)))
    return {char: f"T{idx:0{width}d}" for idx, char in id_to_char.items()}


def build_token_dict_string(tileset_path, char_to_token, char_names):
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lines = []
    for char, tags in data["tiles"].items():
        token = char_to_token[char]
        name = char_names.get(char)
        if not name:
            name = tags[-1].replace("-", " ").title() if tags else "Unknown"
        lines.append(f"{token} = {name}")
    return "\n".join(lines)


def scene_to_tokens(scene, id_to_char, char_to_token):
    unknown = char_to_token.get("?", "T??")
    return "\n".join(
        " ".join(char_to_token.get(id_to_char.get(tid, "?"), unknown) for tid in row)
        for row in scene
    )


def load_api_key(api_key_path):
    with open(api_key_path, "r", encoding="utf-8") as f:
        return f.readline().strip()


def call_claude(prompt, model, api_key, max_tokens, timeout, retries):
    payload = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                parts = result.get("content", [])
                return "".join(p.get("text", "") for p in parts).strip()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [RETRY {attempt + 1}/{retries - 1}] {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Claude request failed after {retries} attempts: {e}"
                ) from e


def call_ollama(prompt, model, url, timeout, retries):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42
        },
    }).encode("utf-8")

    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < retries - 1:
                wait = min(2 ** attempt * 5, 60)
                print(f"  [RETRY {attempt + 1}/{retries - 1}] {e} (waiting {wait}s)")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Ollama request failed after {retries} attempts: {e}"
                ) from e


def parse_captions(raw_response):
    """Parse the LLM's JSON array of 5 captions, with a line-based fallback.

    Returns a list of caption strings (possibly empty if parsing fails entirely).
    """
    text = raw_response.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return [c.strip() for c in parsed if c.strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: try to find the first [...] block in the response.
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return [c.strip() for c in parsed if c.strip()]
        except json.JSONDecodeError:
            pass

    # Last resort: treat each non-empty line as its own caption.
    lines = [ln.strip().lstrip("0123456789.-) ").strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def load_existing(output_path):
    if not os.path.isfile(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["name"]: item for item in data if "captions" in item or "caption" in item}


def _write(output_path, data):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _validate_tileset_match(dataset, id_to_char, tileset_path):
    """Abort early if the dataset's tile IDs don't fit the loaded tileset.

    A common mistake is pairing a dataset built with extended_tiles.json (max ID ~22)
    against mm2_tileset_full.json (138 tiles, different sort order), or vice versa.
    When the IDs don't match, the ASCII fed to the LLM is completely wrong.
    """
    tileset_size = len(id_to_char)
    sample_count = min(50, len(dataset))
    max_seen = 0
    unknown_count = 0

    for item in dataset[:sample_count]:
        scene = item["scene"] if isinstance(item, dict) else item
        for row in scene:
            for tid in row:
                if tid > max_seen:
                    max_seen = tid
                if tid not in id_to_char:
                    unknown_count += 1

    if max_seen >= tileset_size:
        print(
            f"\nERROR: Tileset mismatch detected!\n"
            f"  Tileset '{os.path.basename(tileset_path)}' has {tileset_size} tiles (IDs 0-{tileset_size-1}).\n"
            f"  Dataset contains tile ID {max_seen}, which is out of range.\n"
            f"  You are probably using the wrong --tileset for this dataset.\n"
            f"  If the dataset was built with extended_tiles.json, pass --tileset extended_tiles.json.\n"
            f"  If the dataset was built with mm2_tileset_full.json, pass --tileset mm2_tileset_full.json.\n"
        )
        sys.exit(1)

    if unknown_count > 0:
        print(
            f"WARNING: {unknown_count} tile IDs in the first {sample_count} scenes "
            f"have no mapping in the tileset. The ASCII grid may contain '?' characters."
        )


def generate_captions(dataset_path, tileset_path, output_path, model, url, timeout, retries,
                       grid_format="ascii", tileset_we_path=None, ascii_output_dir=None,
                       backend="ollama", api_key=None, max_tokens=900):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_char = build_id_to_char(tileset_path)
    char_names = get_char_names(tileset_path)

    _validate_tileset_match(dataset, id_to_char, tileset_path)

    if grid_format == "tokens":
        we_id_to_char = build_id_to_char(tileset_we_path)
        char_to_token = build_char_to_token(we_id_to_char)
        we_char_names = get_char_names(tileset_we_path)
        dict_string = build_token_dict_string(tileset_we_path, char_to_token, we_char_names)
        grid_label = "Token Grid (each cell is a tile-ID token, space-separated)"
    else:
        dict_string = build_dict_string(tileset_path, char_names)
        grid_label = "ASCII Level"

    if ascii_output_dir:
        os.makedirs(ascii_output_dir, exist_ok=True)

    existing = load_existing(output_path)
    if existing:
        print(f"Resuming: {len(existing)} captions already present in {output_path}")
    else:
        print("Starting fresh.")

    results = []
    total = len(dataset)
    generated = 0
    skipped = 0
    errors = 0

    for i, item in enumerate(dataset):
        scene = item["scene"] if isinstance(item, dict) else item
        name = item.get("name", str(i)) if isinstance(item, dict) else str(i)

        if grid_format == "tokens":
            ascii_grid = scene_to_tokens(scene, id_to_char, char_to_token)
        else:
            ascii_grid = scene_to_ascii(scene, id_to_char)

        if ascii_output_dir:
            safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
            with open(os.path.join(ascii_output_dir, f"{safe_name}.txt"), "w", encoding="utf-8") as f:
                f.write(ascii_grid)

        if name in existing:
            results.append(existing[name])
            skipped += 1
            continue

        metadata = compute_metadata(scene, id_to_char, char_names, tileset_path)
        prompt = PROMPT_TEMPLATE.format(
            dict_string=dict_string,
            grid_label=grid_label,
            ascii_grid=ascii_grid,
            metadata=metadata,
        )

        print(f"[{i + 1}/{total}] {name} ...", end=" ", flush=True)
        captions = []
        try:
            if backend == "claude":
                raw = call_claude(prompt, model, api_key, max_tokens, timeout, retries)
            else:
                raw = call_ollama(prompt, model, url, timeout, retries)
            captions = parse_captions(raw)
            if captions:
                print(f"OK ({len(captions)} captions)")
            else:
                print("ERROR: could not parse any captions from response")
                errors += 1
        except RuntimeError as e:
            print(f"ERROR: {e}")
            errors += 1

        deterministic = item.get("prompt") if isinstance(item, dict) else None
        if deterministic:
            captions.append(deterministic)

        entry = {"name": name, "scene": scene, "captions": captions}
        if deterministic:
            entry["prompt"] = deterministic
        results.append(entry)
        generated += 1

        if generated % 10 == 0:
            _write(output_path, results)

    _write(output_path, results)
    print(
        f"\nDone. Generated {generated} new captions, "
        f"{skipped} resumed, {errors} errors -> {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered captions for MM2 ASCII datasets via Ollama."
    )
    parser.add_argument("--dataset", required=True, help="Input dataset JSON.")
    parser.add_argument(
        "--tileset",
        default="mm2_tileset_we.json",
        help=(
            "Tileset JSON the dataset was built with (mm2_tileset_we.json, "
            "extended_tiles.json, or mm2_tileset_full.json). Default: mm2_tileset_we.json"
        ),
    )
    parser.add_argument("--output", required=True, help="Output captioned JSON.")
    parser.add_argument(
        "--backend",
        choices=["ollama", "claude"],
        default="ollama",
        help="LLM backend to use. Default: ollama",
    )
    parser.add_argument(
        "--api-key-file",
        default=None,
        help=(
            "Path to a .txtg file whose first line is the full Claude API key. "
            "Required when --backend claude."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Max output tokens for the Claude backend (5 captions need more room than 1). Default: 900",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. For --backend ollama, default: qwen2.5:14b "
            "(pull with: ollama pull qwen2.5:14b; smaller fallback: qwen2.5:7b or llama3.1:8b). "
            "For --backend claude, default: claude-sonnet-4-6."
        ),
    )
    parser.add_argument(
        "--url",
        default="http://localhost:11434/api/generate",
        help="Ollama API endpoint. Default: http://localhost:11434/api/generate",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="Retry attempts on network failure. Default: 10",
    )
    parser.add_argument(
        "--grid-format",
        choices=["ascii", "tokens"],
        default="tokens",
        help=(
            "How the level grid is rendered in the prompt. 'tokens' (default) renders "
            "each cell as a 'T<NN>' token numbered per --tileset-we, which is simpler "
            "for LLMs to read and count. 'ascii' uses the raw tile characters from "
            "--tileset, same as the original behavior."
        ),
    )
    parser.add_argument(
        "--tileset-we",
        default="mm2_tileset_we.json",
        help=(
            "Tileset JSON defining the T0x token numbering, used only when "
            "--grid-format tokens. Default: mm2_tileset_we.json"
        ),
    )
    parser.add_argument(
        "--ascii-output-dir",
        default=None,
        help=(
            "Optional folder to dump the exact grid text (ascii or token format, "
            "matching --grid-format) sent to the LLM for each scene, one .txt file per scene."
        ),
    )
    args = parser.parse_args()

    for path, label in [(args.dataset, "dataset"), (args.tileset, "tileset")]:
        if not os.path.isfile(path):
            print(f"Error: {label} not found: {path}")
            sys.exit(1)

    if args.grid_format == "tokens" and not os.path.isfile(args.tileset_we):
        print(f"Error: tileset-we not found: {args.tileset_we}")
        sys.exit(1)

    api_key = None
    if args.backend == "claude":
        if not args.api_key_file or not os.path.isfile(args.api_key_file):
            print("Error: --api-key-file (a .txtg file with the API key on its first line) is required for --backend claude")
            sys.exit(1)
        api_key = load_api_key(args.api_key_file)

    model = args.model or ("claude-sonnet-4-6" if args.backend == "claude" else "qwen2.5:14b")

    generate_captions(
        args.dataset,
        args.tileset,
        args.output,
        model,
        args.url,
        args.timeout,
        args.retries,
        grid_format=args.grid_format,
        tileset_we_path=args.tileset_we,
        ascii_output_dir=args.ascii_output_dir,
        backend=args.backend,
        api_key=api_key,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
