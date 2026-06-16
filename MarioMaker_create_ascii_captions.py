import json
import os
import sys
import argparse

# Character → human-readable name for the extended tileset (extended_tiles.json)
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

# Character → human-readable name for the full MM2 tileset (mm2_tileset_full.json)
MM2_CHAR_NAMES = {
    # terrain
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
    # pipes / doors / warps
    "|": "Pipe",
    "↑": "Pipe",
    "↓": "Pipe",
    "←": "Pipe",
    "→": "Pipe",
    "D": "Door",
    "W": "Warp Box",
    "k": "Key",
    "f": "Checkpoint Flag",
    "G": "Goal",
    "c": "Clear Pipe",
    # enemies
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
    # items / power-ups / collectables
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
    # platforms / moving objects
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
    # hazards
    "\xc4": "Fire Bar",
    "\xc5": "Saw",
    "\xc6": "Burner",
    "\xc7": "Spike Trap",
    "\xc8": "Spike Ball",
    "\xc9": "Skewer",
    "\xca": "Twister",
    "\xcb": "Icicle",
    "\xd8": "Cannon",
    # decorations / vehicles / misc
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
    elif "mm2_tileset_full" in basename:
        return MM2_CHAR_NAMES
    else:
        print(f"Warning: unrecognized tileset '{basename}', defaulting to full MM2 mapping.")
        return MM2_CHAR_NAMES


def assign_caption(scene, id_to_char, char_names):
    seen = set()
    names = []
    for row in scene:
        for tile_id in row:
            char = id_to_char.get(tile_id)
            if char is None:
                continue
            name = char_names.get(char)
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return " ".join(f"{n}." for n in names)


def generate_captions(dataset_path, tileset_path, output_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    id_to_char = build_id_to_char(tileset_path)
    char_names = get_char_names(tileset_path)

    captioned = []
    for item in dataset:
        scene = item["scene"] if isinstance(item, dict) else item
        caption = assign_caption(scene, id_to_char, char_names)
        entry = {"scene": scene, "caption": caption}
        if isinstance(item, dict):
            if "name" in item:
                entry["name"] = item["name"]
            if "prompt" in item:
                entry["prompt"] = item["prompt"]
        captioned.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(captioned, f, indent=2, ensure_ascii=False)

    print(f"Captioned {len(captioned)} scenes -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple presence-based captions for MM2 ASCII datasets.")
    parser.add_argument("--dataset", required=True, help="Input dataset JSON.")
    parser.add_argument("--tileset", required=True, help="Tileset JSON (extended_tiles.json or mm2_tileset_full.json).")
    parser.add_argument("--output", required=True, help="Output captioned JSON.")
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        print(f"Error: dataset not found: {args.dataset}")
        sys.exit(1)
    if not os.path.isfile(args.tileset):
        print(f"Error: tileset not found: {args.tileset}")
        sys.exit(1)

    generate_captions(args.dataset, args.tileset, args.output)
