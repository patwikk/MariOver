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
    "\xa2": "Coin",
    "g": "Enemy",
    "K": "Koopa",
    "P": "Piranha Plant",
    "t": "Thwomp",
    "^": "Spike",
    "N": "Block",
    "\xb3": "Mushroom Platform",
    "\xb7": "Bridge",
    "\xb4": "Semisolid Platform",
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
    "\xb5": "Goomba's Shoe",
    "\xa2": "Coin",
    "$": "Red Coin",
    "\xa3": "Big Coin",
    "U": "1-Up Mushroom",
    "i": "Fire Flower",
    "\xa4": "Super Star",
    "M": "Super Mushroom",
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

# ── Prompt template ───────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are an expert Mario Maker 2 level captioner.

You will receive three inputs:
1. A symbol dictionary mapping ASCII characters to Mario Maker 2 objects.
2. Pre-computed level metadata (object tile counts, terrain column heights, floor/ceiling analysis).
3. An ASCII level grid (read top-to-bottom, left-to-right).

Trust the metadata for object counts and terrain heights. Do not re-count tiles from the ASCII.

Your output trains a diffusion model. Each phrase must correspond to one concrete, visible structure or feature.

OUTPUT FORMAT

* Output only the caption, nothing else.
* Use lowercase only.
* Write all phrases on a single line separated by periods.
* The entire output must be one unbroken line with no newlines.
* Each phrase describes exactly one structure, object group, region, or feature.
* Phrase length: two to six words each.
* Never explain your reasoning.

PRIORITY ORDER

1. Terrain topology and floor shape.
2. Region divisions (gaps, walls, chambers).
3. Platforms and major mid-air structures.
4. Pipes, doors, and traversal elements.
5. Enemies and hazards (always with placement).
6. Collectibles and power-ups (with placement when prominent).

TERRAIN CLASSES AND DEFINITIONS

Use the most specific matching class. When uncertain, choose the more conservative (smaller, more specific) term.

floor     - continuous ground layer at the bottom
ceiling   - continuous ground layer at the top
wall      - vertical column or barrier of solid tiles attached to floor or ceiling
pillar    - narrow freestanding vertical structure (taller than wide)
platform  - floating horizontal surface not connected to floor or ceiling
bridge    - horizontal surface spanning a gap
staircase - terrain rising or falling in clearly distinct steps; always add ascending or descending
slope     - terrain rising or falling as a smooth incline without steps; always add ascending or descending
hill      - small rounded bump in the ground, 2-5 tiles tall at its peak
plateau   - flat elevated terrain section with steep or near-vertical sides
tower     - tall freestanding structure (taller than wide, at least 5 tiles tall)
mountain  - ONLY for terrain that meets ALL THREE conditions: (1) clearly peaked triangular or rounded summit, (2) peak is at least 6 tiles above the surrounding base terrain, AND (3) base is at least 8 columns wide. Verify against the metadata column heights before using this term.
chamber   - fully or mostly enclosed space formed by terrain
room      - large enclosed region

CRITICAL - MOUNTAIN VS OTHER TERRAIN

Most rising terrain in Mario Maker 2 is NOT a mountain.

Before using mountain, check the metadata column heights. The heights must show a clear rise-and-fall pattern with a peak at least 6 units above the base and a base spanning 8+ columns.

If rising terrain is stepped, use: ascending staircase
If rising terrain is smooth, use: ascending slope
If it is a small bump, use: hill
If it is flat and elevated, use: plateau
When in doubt, do NOT use mountain.

DIRECTIONALITY

Always add direction to staircases and slopes:
  ascending staircase / descending staircase
  ascending slope / descending slope

SHAPE HIERARCHY

Describe terrain as: shape, then direction, then material, then size, then position.
  ascending staircase. hard block wall. ice block plateau. descending slope.

MATERIALS

Include material when the structure is primarily one recognizable tile type:
  ground floor. hard block wall. brick staircase. note block platform. ice block plateau.

OBJECT NAMING

Use the most specific name from the dictionary.
  Use: one goomba / one koopa troopa / one piranha plant
  Avoid: one enemy / one hazard / one collectible

MULTI-TILE OBJECTS

Count game objects, not ASCII tiles. A pipe three tiles tall is one pipe.

QUANTITIES

one / two / three / a few (3-4) / several (5-9) / many (10+)

Never write "one group of N" or "a cluster of N" — just write the count directly: "four enemies" not "one group of four enemies."

POSITION

The metadata includes explicit left/center/right column boundaries for this level. Use those boundaries when assigning position.

Only include a position qualifier when it would genuinely help a reader tell this feature apart from another. Specifically:

USE position when:
- There are two or more instances of the same structure type that need to be distinguished (e.g., "semisolid platform left" and "semisolid platform right")
- A feature is clearly confined to one third of the level width (e.g., a gap that occupies only the center third)

DO NOT use position when:
- There is only one instance of the feature in the level
- The feature spans most of the level width
- You are guessing — omit rather than risk a wrong position

Allowed horizontal positions: left / center / right
Allowed combined positions: upper left / upper center / upper right / lower left / lower center / lower right

Only use "upper" or "lower" when the object is genuinely near the top or bottom third of the level height — not just "above the floor."

At most one position per phrase.

ENEMY PLACEMENT

Always say where enemies are. Priority: (1) support structure, (2) airborne, (3) coarse region.
  one goomba on ground
  two koopas on platform
  one piranha plant on pipe
  several enemies in air

Coarse region (left/center/right) is only for enemy placement when there are enemies in different regions. Do not add a region if all enemies are in the same area as the floor or platform already described.

COLLECTIBLE PLACEMENT

When collectibles are prominent, describe placement:
  line of coins over gap / several coins in air / one mushroom on platform

AIRBORNE OBJECTS

Use "in air" only when the object has no visible support below it.

ADDITIONAL RULES

* Describe only features that are present. Never mention absent features.
* Name structures, not tile arrangements (e.g. "ascending staircase" not "blocks going up to the right").
* Do not use size adjectives (large, small, big, tall, wide, long, short). Use structure class names and materials instead.
* Prefer concise captions over exhaustive tile inventories.
* If a region has no notable features beyond the floor, do not add filler phrases.

---

WORKED EXAMPLES

Example 1:

Symbol dictionary (excerpt): ' '=Air, '#'=Ground, 'g'=Goomba, '?'=Question Block, '|'=Pipe, 'up-arrow'=Pipe (Up)

Metadata:
Object tile counts:
  Goomba: 2
  Question Block: 2
  Pipe (Up): 2
  Pipe: 4
Terrain column heights (left to right, 0=no terrain):
  cols 01-10: 2 2 2 2 2 2 2 2 2 2
  cols 11-20: 2 2 2 2 2 2 2 2 2 2
Floor: present
Ceiling: absent

ASCII Level:

  ?     ?

  |     |
  ^     ^
  g         g
####################
####################

Caption: flat ground floor. two question blocks left. two upward pipes on ground. two goombas on ground

---

Example 2:

Symbol dictionary (excerpt): ' '=Air, '#'=Ground, 'g'=Goomba, 'B'=Brick Block

Metadata:
Object tile counts:
  Goomba: 1
  Brick Block: 6
Terrain column heights (left to right, 0=no terrain):
  cols 01-10: 2 2 2 2 2 2 2 2 2 0
  cols 11-20: 0 0 2 3 4 5 6 7 8 0
Floor: present, gaps at: cols 10-12
Ceiling: absent

ASCII Level:


               #
              ##
             ###
            ####
  g  BBBBBB #####
##########   #####
##########   #####

Caption: flat ground floor left. gap center. ascending staircase right. brick block platform center. one goomba on ground left

---

Symbol dictionary:

{dict_string}


Metadata:
{metadata}


ASCII Level:
{ascii_grid}

Write the caption."""


# ── Core helpers ──────────────────────────────────────────────────────────────

def build_id_to_char(tileset_path):
    with open(tileset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tile_chars = sorted(data["tiles"].keys())
    if "_" not in tile_chars:
        tile_chars.append("_")
    return {idx: char for idx, char in enumerate(tile_chars)}


def get_char_names(tileset_path):
    basename = os.path.basename(tileset_path)
    if "extended_tiles" in basename:
        return EXTENDED_CHAR_NAMES
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
    terrain = TERRAIN_CHARS_EXT if "extended" in basename else TERRAIN_CHARS_MM2

    parts = []

    # Object tile counts (skip Air and Ground to keep it readable)
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
        parts.append("Object tile counts:")
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


def call_ollama(prompt, model, url, timeout, retries):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
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
                print(f"  [RETRY {attempt + 1}/{retries - 1}] {e}")
                time.sleep(3)
            else:
                raise RuntimeError(
                    f"Ollama request failed after {retries} attempts: {e}"
                ) from e


def load_existing(output_path):
    if not os.path.isfile(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["name"]: item for item in data if "caption" in item}


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


def generate_captions(dataset_path, tileset_path, output_path, model, url, timeout, retries):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_char = build_id_to_char(tileset_path)
    char_names = get_char_names(tileset_path)
    dict_string = build_dict_string(tileset_path, char_names)

    _validate_tileset_match(dataset, id_to_char, tileset_path)

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

        if name in existing:
            results.append(existing[name])
            skipped += 1
            continue

        ascii_grid = scene_to_ascii(scene, id_to_char)
        metadata = compute_metadata(scene, id_to_char, char_names, tileset_path)
        prompt = PROMPT_TEMPLATE.format(
            dict_string=dict_string,
            ascii_grid=ascii_grid,
            metadata=metadata,
        )

        print(f"[{i + 1}/{total}] {name} ...", end=" ", flush=True)
        try:
            caption = call_ollama(prompt, model, url, timeout, retries).replace("\n", ". ")
            print("OK")
        except RuntimeError as e:
            print(f"ERROR: {e}")
            caption = ""
            errors += 1

        entry = {"name": name, "scene": scene, "caption": caption}
        if isinstance(item, dict) and "prompt" in item:
            entry["prompt"] = item["prompt"]
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
        required=True,
        help="Tileset JSON (extended_tiles.json or mm2_tileset_full.json).",
    )
    parser.add_argument("--output", required=True, help="Output captioned JSON.")
    parser.add_argument(
        "--model",
        default="qwen2.5:14b",
        help=(
            "Ollama model name. Default: qwen2.5:14b. "
            "Pull with: ollama pull qwen2.5:14b. "
            "Smaller fallback: qwen2.5:7b or llama3.1:8b."
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
        default=3,
        help="Retry attempts on network failure. Default: 3",
    )
    args = parser.parse_args()

    for path, label in [(args.dataset, "dataset"), (args.tileset, "tileset")]:
        if not os.path.isfile(path):
            print(f"Error: {label} not found: {path}")
            sys.exit(1)

    generate_captions(
        args.dataset,
        args.tileset,
        args.output,
        args.model,
        args.url,
        args.timeout,
        args.retries,
    )


if __name__ == "__main__":
    main()
