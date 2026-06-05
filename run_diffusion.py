#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import random
from level_dataset import visualize_samples, samples_to_scenes
from create_ascii_captions import save_level_data
from create_level_json_data import load_tileset, MM2_EXTRA_TILE
import util.common_settings as common_settings
from models.pipeline_loader import get_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MM2 levels using a trained diffusion model (unconditional)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of levels to generate")
    parser.add_argument("--output_dir", type=str, default="generated_levels", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--inference_steps", type=int, default=common_settings.NUM_INFERENCE_STEPS, help="Number of denoising steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation")
    parser.add_argument("--save_as_json", action="store_true", help="Save generated levels as JSON")
    parser.add_argument("--level_width", type=int, default=None, help="Override model width in tiles")
    parser.add_argument(
        "--output_format",
        type=str,
        default="ascii",
        choices=["ascii", "image", "both"],
        help="Output format: ascii text files, tile images, or both",
    )
    parser.add_argument("--tileset", default="smb.json", help="Path to tileset JSON")
    parser.add_argument("--describe_absence", action="store_true", default=False, help="Caption mentions when tiles are entirely absent")
    return parser.parse_args()


def samples_to_ascii(samples, id_to_char):
    """Convert [batch, channels, height, width] tensors to lists of ASCII row strings."""
    indices = torch.argmax(samples, dim=1).cpu().numpy()
    results = []
    for grid in indices:
        rows = ["".join(id_to_char.get(int(idx), "?") for idx in row) for row in grid]
        results.append(rows)
    return results


def save_ascii_levels(ascii_levels, output_dir, start_index=0):
    for i, rows in enumerate(ascii_levels):
        path = os.path.join(output_dir, f"level_{start_index + i:04d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows))


def generate_levels(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    tile_to_id = load_tileset(args.tileset, extra_tile=MM2_EXTRA_TILE)
    id_to_char = {v: k for k, v in tile_to_id.items()}
    print(f"Tileset: {len(tile_to_id)} tile types from {args.tileset}")

    print(f"Loading model from {args.model_path}...")
    pipeline = get_pipeline(args.model_path)
    pipeline.to(device)

    model_channels = pipeline.unet.config.in_channels
    if model_channels != len(tile_to_id):
        print(f"Warning: model has {model_channels} channels but tileset has {len(tile_to_id)} tiles")

    ss = pipeline.unet.config.sample_size
    if isinstance(ss, (tuple, list)):
        scene_height, scene_width = ss
    else:
        scene_height = scene_width = ss

    if args.level_width is not None:
        scene_width = args.level_width
        print(f"Overriding model width to {scene_width} tiles")

    print(f"Scene size: {scene_height}x{scene_width}")

    total_samples = args.num_samples
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size
    all_samples = []

    for batch_idx in range(num_batches):
        current_batch_size = min(args.batch_size, total_samples - batch_idx * args.batch_size)
        print(f"Generating batch {batch_idx+1}/{num_batches} ({current_batch_size} samples)...")

        with torch.no_grad():
            samples = pipeline(
                batch_size=current_batch_size,
                generator=torch.Generator(device).manual_seed(args.seed + batch_idx),
                num_inference_steps=args.inference_steps,
                output_type="tensor",
                height=scene_height,
                width=scene_width,
            ).images

        all_samples.append(samples)
        start_index = batch_idx * args.batch_size

        if args.output_format in ("ascii", "both"):
            ascii_levels = samples_to_ascii(samples, id_to_char)
            save_ascii_levels(ascii_levels, args.output_dir, start_index)
            print(f"  Saved {len(ascii_levels)} ASCII levels to {args.output_dir}")

        if args.output_format in ("image", "both"):
            visualize_samples(samples, args.output_dir, True, start_index)
            print(f"  Saved {current_batch_size} level images to {args.output_dir}")

    all_samples = torch.cat(all_samples, dim=0)[:total_samples]
    print(f"Done. Generated {total_samples} levels.")

    if args.save_as_json:
        scenes = samples_to_scenes(all_samples)
        out_path = os.path.join(args.output_dir, "all_levels.json")
        save_level_data(scenes, args.tileset, out_path, False, args.describe_absence, exclude_broken=False)
        print(f"Saved {len(scenes)} captioned scenes to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    generate_levels(args)
