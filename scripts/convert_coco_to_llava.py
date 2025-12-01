"""Utility to convert COCO detection annotations into LLaVA-style JSON.

Example usage:
    python scripts/convert_coco_to_llava.py \
        --coco-root /path/to/coco \
        --split train2017 \
        --output coco_train_llava.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

BBox = Sequence[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco-root",
        type=Path,
        required=True,
        help="Path to the COCO dataset directory that contains `annotations/`.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train2017",
        choices=["train2017", "val2017"],
        help="Which split to convert. Uses `annotations/instances_<split>.json`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output JSON file in LLaVA format.",
    )
    parser.add_argument(
        "--bbox-per-image",
        type=int,
        default=None,
        help=(
            "Maximum number of bounding boxes to include per image. "
            "The largest boxes by area are kept when this limit is set."
        ),
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Ignore bounding boxes with an area (in pixels^2) smaller than this threshold.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of image samples written to the output JSON.",
    )
    parser.add_argument(
        "--always-include-image-token",
        action="store_true",
        help="If set, every question will be prefixed with the <image> token."
        " Otherwise only the first question per image includes it.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent level for the output file.",
    )
    return parser.parse_args()


def load_instances(instances_path: Path) -> tuple[Dict[int, dict], Dict[int, str], Dict[int, List[dict]]]:
    with instances_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    images = {img["id"]: img for img in data["images"]}
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}

    annotations: Dict[int, List[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)

    return images, categories, annotations


def coco_bbox_to_xyxy(box: BBox) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def format_coordinate(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def build_question(idx: int, bbox: BBox, include_image_token: bool) -> str:
    x1, y1, x2, y2 = coco_bbox_to_xyxy(bbox)
    coords = ", ".join(
        [
            format_coordinate(coord)
            for coord in (x1, y1, x2, y2)
        ]
    )
    prompt = (
        f"Bounding box #{idx + 1} is located at (x1, y1, x2, y2) = ({coords}). "
        "What object is inside this region?"
    )
    return ("<image>\n" if include_image_token else "") + prompt


def build_answer(category_name: str, bbox: BBox) -> str:
    x1, y1, x2, y2 = coco_bbox_to_xyxy(bbox)
    coords = ", ".join(
        format_coordinate(coord) for coord in (x1, y1, x2, y2)
    )
    return (
        f"The region corresponds to a '{category_name}' and covers coordinates ({coords})."
    )


def select_annotations(anns: Iterable[dict], bbox_per_image: int | None, min_area: float) -> List[dict]:
    filtered = [ann for ann in anns if ann.get("area", 0.0) >= min_area]
    if bbox_per_image is None or len(filtered) <= bbox_per_image:
        return filtered
    return sorted(filtered, key=lambda ann: ann.get("area", 0.0), reverse=True)[:bbox_per_image]


def convert_split(
    images: Dict[int, dict],
    categories: Dict[int, str],
    annotations: Dict[int, List[dict]],
    image_root: Path,
    bbox_per_image: int | None,
    min_area: float,
    max_samples: int | None,
    always_include_image_token: bool,
) -> List[dict]:
    dataset: List[dict] = []
    # sort by file name for deterministic output
    sorted_image_ids = sorted(images.keys(), key=lambda img_id: images[img_id]["file_name"])

    for image_id in sorted_image_ids:
        anns = annotations.get(image_id)
        if not anns:
            continue

        selected = select_annotations(anns, bbox_per_image, min_area)
        if not selected:
            continue

        conversations = []
        for idx, ann in enumerate(selected):
            category_name = categories.get(ann.get("category_id", -1), "unknown object")
            include_token = always_include_image_token or idx == 0
            question = build_question(idx, ann["bbox"], include_token)
            answer = build_answer(category_name, ann["bbox"])
            conversations.extend(
                [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer},
                ]
            )

        dataset.append(
            {
                "id": str(images[image_id]["id"]),
                "image": str((image_root / images[image_id]["file_name"]).resolve()),
                "conversations": conversations,
            }
        )

        if max_samples is not None and len(dataset) >= max_samples:
            break

    return dataset


def main() -> None:
    args = parse_args()
    instances_path = args.coco_root / "annotations" / f"instances_{args.split}.json"
    if not instances_path.is_file():
        raise FileNotFoundError(f"Could not find {instances_path}")

    images, categories, annotations = load_instances(instances_path)
    image_root = args.coco_root / args.split
    dataset = convert_split(
        images=images,
        categories=categories,
        annotations=annotations,
        image_root=image_root,
        bbox_per_image=args.bbox_per_image,
        min_area=args.min_area,
        max_samples=args.max_samples,
        always_include_image_token=args.always_include_image_token,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(dataset, fp, ensure_ascii=False, indent=args.indent)

    print(
        f"Wrote {len(dataset)} samples with LLaVA conversations to {args.output}"
    )


if __name__ == "__main__":
    main()