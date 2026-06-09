import argparse
import json
import random


def get_tile_names(captioned_json):
    names = set()
    with open(captioned_json, encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        for part in item.get("caption", "").split("."):
            name = part.strip()
            if name:
                names.add(name)
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description="Generate random test captions for the MM simple presence-based caption scheme.")
    parser.add_argument("--json", required=True, help="Captioned dataset JSON (used to extract tile vocabulary).")
    parser.add_argument("--output", required=True, help="Output JSON file.")
    parser.add_argument("--num_captions", type=int, default=100)
    parser.add_argument("--min_tiles", type=int, default=1)
    parser.add_argument("--max_tiles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=512)
    args = parser.parse_args()

    tile_names = get_tile_names(args.json)
    print(f"Tile vocabulary ({len(tile_names)} names): {tile_names}")

    random.seed(args.seed)
    captions = []
    seen = set()
    attempts = 0
    cap = min(args.max_tiles, len(tile_names))

    while len(captions) < args.num_captions and attempts < args.num_captions * 200:
        attempts += 1
        k = random.randint(args.min_tiles, cap)
        subset = tuple(sorted(random.sample(tile_names, k)))
        caption = " ".join(f"{name}." for name in subset)
        if caption not in seen:
            seen.add(caption)
            captions.append({"caption": caption})

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved {len(captions)} random captions -> {args.output}")


if __name__ == "__main__":
    main()
