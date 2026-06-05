import argparse
import os
import torch
import random
import numpy as np
from collections import defaultdict, Counter
import json

from level_dataset import convert_to_level_format
import util.common_settings as common_settings
from captions.util import extract_tileset
from models.pipeline_loader import get_pipeline
from models.fdm_pipeline import FDMPipeline
from models.latent_diffusion_pipeline import UnconditionalDDPMPipeline
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate tile type distribution across model checkpoints"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of levels to generate per checkpoint")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size for generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_tiles", type=int, default=common_settings.MARIO_TILE_COUNT, help="Number of tile types")
    parser.add_argument("--tileset", type=str, default=common_settings.MARIO_TILESET, help="Path to tileset JSON")
    parser.add_argument("--width", type=int, default=common_settings.MARIO_WIDTH, help="Width of generated levels")
    parser.add_argument("--height", type=int, default=common_settings.MARIO_HEIGHT, help="Height of generated levels")
    parser.add_argument("--inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=common_settings.GUIDANCE_SCALE, help="Classifier-free guidance scale")
    parser.add_argument("--caption", type=str, default="", help="Caption for conditional models (empty = unconditioned)")
    return parser.parse_args()


# Authoritative MM2 char→name mapping sourced from OBJ_META in mm2_viewer_json.py.
# Can't import that file directly (it pulls in tkinter), so the names are mirrored here.
_MM2_OBJ_NAMES = {
    # terrain
    "#": "Ground",          "B": "Block",               "H": "Hard Block",
    "?": "? Block",         "h": "Hidden Block",         "N": "Note Block",
    "d": "Donut Block",     "I": "Ice Block",            "p": "P Block",
    "O": "ON/OFF Block",    ".": "Dotted-Line Block",    "*": "Blinking Block",
    "^": "Spike Block",     "C": "Crate",                "S": "Stone",
    "{": "Starting Brick",  "=": "Castle Bridge",        "T": "Tree",
    "/": "Slight Slope",    "\\": "Steep Slope",
    # doors / warps
    "|": "Pipe",            "D": "Door",                 "W": "Warp Box",
    "k": "Key",             "f": "Checkpoint Flag",      "G": "Goal",
    "c": "Clear Pipe",
    # enemies
    "g": "Goomba",          "K": "Koopa",                "P": "Piranha Plant",
    "m": "Hammer Bro",      "t": "Thwomp",               "o": "Bob-omb",
    "s": "Spiny",           "b": "Buzzy Beetle",         "L": "Lakitu",
    "l": "Lakitu's Cloud",  "Z": "Banzai Bill",          "V": "Bullet Bill Blaster",
    "y": "Magikoopa",       "<": "Spike Top",            "u": "Boo",
    "X": "Bowser",          "x": "Bowser Jr.",           "@": "Chain Chomp",
    "~": "Cheep Cheep",     "q": "Blooper",              "w": "Wiggler",
    "Y": "Pokey",           "e": "Piranha Creeper",      "F": "Porcupuffer",
    "%": "Fish Bone",       "&": "Lava Bubble",          "r": "Rocky Wrench",
    ",": "Muncher",         "a": "Ant Trooper",          "n": "Monty Mole",
    "R": "Mechakoopa",      "!": "Boom Boom",            "9": "Dry Bones",
    "j": "Skipsqueak",      ";": "Stingby",              "A": "Angry Sun",
    "v": "Charvaargh",      "[": "Bully",
    "1": "Lemmy",           "2": "Morton",               "3": "Larry",
    "4": "Wendy",           "5": "Iggy",                 "6": "Roy",
    "7": "Ludwig",
    # items
    "¢": "Coin",            "$": "Red Coin",             "£": "Large Coin",
    "U": "1-Up Mushroom",   "i": "Fire Flower",          "¤": "Super Star",
    "M": "Super Mushroom",  "¶": "Big Mushroom",         "§": "SMB2 Mushroom",
    "¬": "Super Hammer",    "¦": "P Switch",             "¯": "POW Block",
    "±": "Spring",          "µ": "Goomba's Shoe",        "]": "Cannon Box",
    "}": "Propeller Box",   ")": "Goomba Mask",          "°": "Bullet Bill Mask",
    "²": "Red POW Box",
    # platforms
    "-": "Lift",            "³": "Mushroom Platform",    "´": "Semisolid Platform",
    "·": "Bridge",          "¸": "Lava Lift",            "¹": "Snake Block",
    "º": "Track Block",     "»": "Conveyor Belt",        "¼": "Fast Conveyor Belt",
    "½": "Sprint Platform", "¾": "Seesaw",               "¿": "Swinging Claw",
    "À": "ON/OFF Trampoline", "Á": "Mushroom Trampoline", "J": "Jumping Machine",
    "Â": "Half-Collision Platform", "Ã": "Donut",
    # hazards
    "Ä": "Fire Bar",        "Å": "Saw",                  "Æ": "Burner",
    "Ç": "Spikes",          "È": "Spike Ball",           "É": "Skewer",
    "Ê": "Twister",         "Ë": "Icicle",
    # decoration / other
    "Ì": "Cloud",           "Í": "Vine",                 "Î": "Water Marker",
    "Ï": "Arrow",           "Ð": "One-Way Wall",         "Ñ": "Reel Camera",
    "Ò": "Sound Effect",    "Ó": "Player",               "Ô": "Clown Car",
    "Õ": "Koopa Clown Car", "Ö": "Track",                "×": "Starting Arrow",
    "Ø": "Cannon",          "Ù": "! Block",
    # pipe directions (in tileset but not in OBJ_META)
    "↑": "Pipe (Up)",       "↓": "Pipe (Down)",
    "←": "Pipe (Left)",     "→": "Pipe (Right)",
}

_SKIP_DESCRIPTORS = {"solid", "passable", "moving", "damaging", "hazard", "enemy",
                     "collectable", "platform", "interactive", "decoration", "vehicle",
                     "flying", "projectile", "boss", "slope", "falling"}

def build_char_to_name(tileset_path):
    """
    Return a dict mapping each tile char to a unique human-readable name.
    Uses _MM2_OBJ_NAMES when available (authoritative); falls back to
    descriptor-based heuristics for tilesets not covered by that dict.
    Disambiguates any remaining duplicates by appending the char.
    """
    with open(tileset_path, 'r', encoding='utf-8') as f:
        tileset = json.load(f)

    def pick_name_from_descriptors(descriptors):
        filtered = [d for d in descriptors if d not in ("solid", "passable")]
        if not filtered:
            return None
        multi_word = [d for d in filtered if ' ' in d]
        if multi_word:
            return max(multi_word, key=len)
        specific = [d for d in filtered if d not in _SKIP_DESCRIPTORS]
        if specific:
            return specific[-1]
        return filtered[-1]

    raw_names = {}
    for char, descriptors in tileset['tiles'].items():
        if char in _MM2_OBJ_NAMES:
            raw_names[char] = _MM2_OBJ_NAMES[char]
        else:
            raw_names[char] = pick_name_from_descriptors(descriptors) or char

    # Disambiguate: when two chars share a raw name, append (char) to both
    name_count = Counter(raw_names.values())
    char_to_name = {}
    for char, name in raw_names.items():
        if name_count[name] > 1:
            char_to_name[char] = f"{name} ({char})"
        else:
            char_to_name[char] = name

    return char_to_name


def generate_samples(pipe, num_samples, batch_size, inference_steps, guidance_scale, seed, height, width, caption=""):
    """Generate num_samples levels, returning a list of 2D tile-index scenes."""
    is_unconditional = isinstance(pipe, UnconditionalDDPMPipeline)
    is_fdm = isinstance(pipe, FDMPipeline)

    if hasattr(pipe, 'unet') and pipe.unet is not None:
        device = next(pipe.unet.parameters()).device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_scenes = []
    remaining = num_samples

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        generator = torch.Generator(device).manual_seed(seed)

        with torch.no_grad():
            if is_fdm:
                param_values = {"caption": [caption] * current_batch, "batch_size": current_batch}
            elif is_unconditional:
                param_values = {
                    "num_inference_steps": inference_steps,
                    "height": height, "width": width,
                    "output_type": "tensor", "batch_size": current_batch,
                }
            else:
                param_values = {
                    "caption": [caption] * current_batch,
                    "num_inference_steps": inference_steps,
                    "height": height, "width": width,
                    "guidance_scale": guidance_scale,
                    "output_type": "tensor", "batch_size": current_batch,
                }

            samples = pipe(generator=generator, **param_values).images

        for i in range(len(samples)):
            sample = samples[i].unsqueeze(0)
            scene = convert_to_level_format(sample)[0].tolist()
            all_scenes.append(scene)

        remaining -= current_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_scenes


def count_tiles(scenes, id_to_char, char_to_name):
    """Count tile occurrences across all scenes, keyed by human-readable name."""
    counts = defaultdict(int)
    for scene in scenes:
        for row in scene:
            for tile_id in row:
                char = id_to_char.get(tile_id, '_')
                name = char_to_name.get(char, char)
                counts[name] += 1
    return dict(counts)


def write_distribution(path, header, counts, total, tile_names):
    """Write a readable distribution file sorted by frequency."""
    present = [(n, counts.get(n, 0)) for n in tile_names if counts.get(n, 0) > 0]
    present.sort(key=lambda x: -x[1])
    col_width = max((len(n) for n, _ in present), default=10) + 2
    with open(path, 'w') as f:
        f.write(header + "\n")
        f.write("=" * (col_width + 30) + "\n")
        for name, count in present:
            pct = 100.0 * count / total if total > 0 else 0.0
            f.write(f"{name:<{col_width}}: {count:>8}  ({pct:5.1f}%)\n")
        f.write("=" * (col_width + 30) + "\n")
        f.write(f"{'Total':<{col_width}}: {total:>8}\n")


def collect_checkpoint_dirs(model_path):
    checkpoint_dirs = [
        (int(d.split("-")[-1]), os.path.join(model_path, d))
        for d in os.listdir(model_path)
        if os.path.isdir(os.path.join(model_path, d)) and d.startswith("checkpoint-")
    ]
    checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: x[0])

    has_unet = os.path.isdir(os.path.join(model_path, "unet"))
    has_fdm = os.path.isdir(os.path.join(model_path, "final-model")) or (
        not has_unet and any(
            f in os.listdir(model_path) for f in ["config.json", "pytorch_model.bin", "model.safetensors"]
        )
    )
    if checkpoint_dirs and (has_unet or has_fdm):
        checkpoint_dirs.append((checkpoint_dirs[-1][0] + 1, model_path))

    return checkpoint_dirs


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tile_chars, id_to_char, _, _ = extract_tileset(args.tileset)
    char_to_name = build_char_to_name(args.tileset)
    # Unique ordered tile names — no duplicates, no padding tile
    seen = set()
    tile_names = []
    for c in tile_chars:
        if c == '_':
            continue
        name = char_to_name.get(c, c)
        if name not in seen:
            seen.add(name)
            tile_names.append(name)

    checkpoint_dirs = collect_checkpoint_dirs(args.model_path)
    if not checkpoint_dirs:
        print(f"No checkpoints found in {args.model_path}")
        return

    out_root = os.path.join(args.model_path, "tile_distributions")
    total_path = os.path.join(out_root, "total_distribution.txt")

    if os.path.exists(out_root):
        print(f"Error: Output directory '{out_root}' already exists. Delete it to re-run.")
        return

    created_dirs = []
    created_files = []

    try:
        os.makedirs(out_root)
        created_dirs.append(out_root)

        total_counts = defaultdict(int)
        total_tiles = 0

        for epoch, checkpoint_dir in tqdm(checkpoint_dirs, desc="Evaluating checkpoints"):
            print(f"\nCheckpoint {epoch}: {checkpoint_dir}")
            pipe = get_pipeline(checkpoint_dir).to(device)
            model_type = "unconditional" if isinstance(pipe, UnconditionalDDPMPipeline) else "conditional"
            print(f"  Model type: {model_type}")

            scenes = generate_samples(
                pipe,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                height=args.height,
                width=args.width,
                caption=args.caption,
            )

            counts = count_tiles(scenes, id_to_char, char_to_name)
            total = sum(counts.values())

            # Per-checkpoint folder and file
            ckpt_dir = os.path.join(out_root, f"checkpoint-{epoch}")
            os.makedirs(ckpt_dir)
            created_dirs.append(ckpt_dir)

            dist_path = os.path.join(ckpt_dir, "distribution.txt")
            header = f"Checkpoint {epoch} — Tile Distribution ({len(scenes)} samples, {total} tiles)"
            write_distribution(dist_path, header, counts, total, tile_names)
            created_files.append(dist_path)

            # Print quick summary to console
            summary = "  ".join(
                f"{n}={counts.get(n,0)} ({100*counts.get(n,0)/total:.1f}%)"
                for n in tile_names if counts.get(n, 0) > 0
            )
            print(f"  {summary}")

            for name, count in counts.items():
                total_counts[name] += count
            total_tiles += total

            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        header = f"Total Distribution Across All Checkpoints ({total_tiles} tiles)"
        write_distribution(total_path, header, total_counts, total_tiles, tile_names)
        created_files.append(total_path)

        print(f"\nDone. Results saved to: {out_root}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Cleaning up output files created this run...")
        for path in created_files:
            if os.path.exists(path):
                os.remove(path)
                print(f"  Deleted: {path}")
        for d in reversed(created_dirs):
            try:
                os.rmdir(d)
                print(f"  Removed dir: {d}")
            except OSError:
                pass
        raise


if __name__ == "__main__":
    main()
