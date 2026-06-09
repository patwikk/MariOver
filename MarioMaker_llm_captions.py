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
    "¢": "Coin",
    "g": "Enemy",
    "K": "Koopa",
    "P": "Piranha Plant",
    "t": "Thwomp",
    "^": "Spike",
    "N": "Block",
    "³": "Mushroom Platform",
    "·": "Bridge",
    "´": "Semisolid Platform",
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

# ── Prompt template ───────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are an expert Mario Maker 2 level captioner.

You will be given:

1. A dictionary mapping ASCII symbols to Mario Maker 2 objects.
2. Optional pre-computed level metadata (object counts, terrain analysis, gap analysis, region analysis, structure locations). When present, trust the metadata over your own reading of the ASCII level, especially for counts and terrain descriptions.
3. An ASCII level.

Write a caption describing the level.

These captions train a diffusion model that learns to associate short phrases with visible level structures. Each phrase should correspond to one concrete thing that can be generated in a level.

OUTPUT FORMAT

* Output only the caption.
* Use lowercase only.
* Write all phrases on a single line separated by periods.
* Do not output any \n characters. The entire caption must be one single unbroken line of text.
* Do not write prose.
* Do not write complete sentences.
* Each phrase should describe exactly one structure, object group, region, or feature.
* Most phrases should be between two and six words.
* Never explain your reasoning.

GENERAL PRINCIPLES

Each phrase should answer one or more of:

* what is it?
* what shape is it?
* what is it made of?
* where is it?

Prefer concrete, reusable descriptions.

Avoid subjective language.

Do not discuss:

* gameplay
* difficulty
* quality
* fun
* creativity
* designer intent

Do not address the player.

PHRASE RULES

* One phrase = one concept.
* Do not combine multiple structures into one phrase.
* If two things are important, write two phrases.
* Avoid repetition.
* Do not describe the same feature twice.

PRIORITY ORDER

1. Terrain topology.
2. Region divisions and room structure.
3. Platforms and major structures.
4. Pipes, doors, warp elements, and traversal structures.
5. Enemies and hazards.
6. Collectibles and power-ups.

TERRAIN AND STRUCTURE CLASSES

Prefer these structure names when appropriate:

* floor
* ceiling
* hill
* mountain
* plateau
* pillar
* wall
* staircase
* slope
* platform
* bridge
* tower
* chamber
* room
* gap

You may create new structure names when necessary if they are:

* concrete
* geometric
* reusable
* three words or fewer

Examples:

* stepped pyramid
* zigzag staircase
* floating bridge
* split plateau
* block tower
* central arch

Do not invent vague names.

Bad examples:

* interesting formation
* unusual terrain
* decorative structure
* complex layout

DIRECTIONALITY

When a structure has a clear orientation, include it.

Examples:

* ascending staircase
* descending staircase
* ascending slope
* descending slope
* ascending mountain
* descending mountain
* vertical pillar
* horizontal platform

Direction is usually more informative than size.

TERRAIN INTERPRETATION

Treat connected terrain as a single landform whenever possible.

Describe the overall shape rather than individual rows, columns, or tile edges.

Never use vague terms such as:

* raised terrain
* elevated ground
* terrain formation
* ceiling ledge
* ground structure
* tall ground blocks

Always choose the closest concrete landform.

SHAPE HIERARCHY

When describing terrain, prioritize:

1. shape
2. direction
3. material
4. size
5. position

Examples:

* ascending staircase
* descending mountain
* hard block wall
* ice block plateau

STRUCTURE MATERIALS

Whenever a structure is primarily composed of a recognizable material, include the material.

Examples:

* ground mountain
* hard block wall
* hard block chamber
* breakable block tower
* note block platform
* ice block staircase
* spike block pillar

Prefer:

* hard block chamber

instead of:

* enclosed chamber

Prefer:

* breakable block tower

instead of:

* tower

Shape is more important than material, but material should usually be included when recognizable.

REGION STRUCTURE

When terrain divides the level into separate areas, describe the division.

Examples:

* two separate sections
* three separate sections
* central dividing wall
* hard block chamber
* sealed room left

Describe major regions and dividers.

Do not describe how regions connect.

OBJECT NAMING

Use the most specific object name available from the dictionary.

Prefer:

* one goomba
* one koopa troopa
* one piranha plant
* one mushroom
* one upward pipe

instead of:

* one enemy
* one hazard
* one collectible
* one power-up

Only use generic terms when no more specific name exists in the dictionary.

MULTI-TILE OBJECT RECOGNITION

Many Mario Maker objects occupy multiple ASCII cells.

A single object may span multiple rows or columns.

Examples:

* a pipe that is three tiles tall is one pipe
* a pipe that is four tiles wide is one pipe
* a goal pole spanning many rows is one goal
* a clear pipe spanning many cells is one clear pipe

Never count ASCII cells.

Count game objects.

When several connected tiles form one object, count one object.

Object counts are based on connected instances, not symbol frequency.

STRUCTURE RECOGNITION

Recognize structures before counting tiles.

Examples:

* line of coins
* tower of blocks
* ascending staircase
* descending staircase
* semisolid platform
* mushroom platform
* bridge
* wall
* pillar

Use a structure name whenever a clear structure exists.

Only fall back to object counts when no meaningful structure is present.

QUANTITIES

Use:

* one
* two
* three
* a few (3-4)
* several (5-9)
* many (10+)

POSITION

Only include position when it helps distinguish major features.

Allowed positions:

* left
* center
* right
* upper left
* upper center
* upper right
* lower left
* lower center
* lower right

Use at most one position per phrase.

ENEMY AND HAZARD PLACEMENT

Enemy placement is important.

Whenever enemies or hazards are mentioned, also describe where they are located whenever that information is visible.

Placement priority:

1. support structure
2. airborne status
3. coarse region

Examples:

* one goomba on ground
* two koopas on platform
* one piranha plant on pipe
* one enemy on staircase
* several enemies on semisolid platform
* one flying enemy in air

If an enemy is visibly supported by terrain or a structure, describe the support structure.

If an enemy is not visibly supported, describe it as:

* in air

Group enemies that share the same placement.

Examples:

* several goombas on ground
* two koopas on platform
* many enemies in air

Do not describe exact coordinates.

Do not use:

* above a pipe
* beside a block
* next to an enemy

Use support surfaces, airborne status, or coarse regions only.

COLLECTIBLE AND POWER-UP PLACEMENT

When collectibles or power-ups are important visible features, describe their placement.

Examples:

* line of coins over gap
* several coins in air
* one mushroom on platform
* one key on ground
* one fire flower in air

AIRBORNE RECOGNITION

Objects should only be considered airborne when they are not visibly supported by terrain, platforms, pipes, blocks, or other solid structures.

Examples:

* one enemy in air
* several coins in air
* one mushroom in air

Do not use "in air" when the object rests on a visible structure.

ADDITIONAL RULES

* Describe only features that are present.
* Never mention missing features.
* Trust provided metadata over your own counting.
* Count objects, not symbols.
* Name structures rather than tile arrangements whenever possible.
* Prefer concise captions over exhaustive inventories.



Dictionary:

{dict_string}


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
            "num_predict": 256,
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

def generate_captions(dataset_path, tileset_path, output_path, model, url, timeout, retries):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_char = build_id_to_char(tileset_path)
    char_names = get_char_names(tileset_path)
    dict_string = build_dict_string(tileset_path, char_names)

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
        prompt = PROMPT_TEMPLATE.format(dict_string=dict_string, ascii_grid=ascii_grid)

        print(f"[{i + 1}/{total}] {name} ...", end=" ", flush=True)
        try:
            caption = call_ollama(prompt, model, url, timeout, retries)
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
        "--model", default="llama3.1:8b", help="Ollama model name. Default: llama3.1:8b"
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
