"""Create a small offline demo catalog for ClothLens."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IMAGE_SIZE = (384, 512)


@dataclass(frozen=True, slots=True)
class DemoItem:
    item_id: str
    category: str
    name: str
    color_name: str
    color: tuple[int, int, int]


CATALOG_ITEMS = [
    DemoItem("demo-top-blue-shirt", "top", "Blue shirt", "blue", (42, 104, 211)),
    DemoItem("demo-top-red-shirt", "top", "Red shirt", "red", (214, 68, 68)),
    DemoItem("demo-bottom-blue-jeans", "bottom", "Blue jeans", "blue", (50, 92, 168)),
    DemoItem("demo-bottom-black-pants", "bottom", "Black pants", "black", (40, 42, 48)),
    DemoItem("demo-dress-green", "dress", "Green dress", "green", (62, 156, 99)),
    DemoItem("demo-dress-purple", "dress", "Purple dress", "purple", (126, 82, 178)),
    DemoItem("demo-shoes-white", "shoes", "White sneakers", "white", (235, 235, 228)),
    DemoItem("demo-shoes-black", "shoes", "Black shoes", "black", (35, 35, 38)),
    DemoItem("demo-bag-brown", "bag", "Brown bag", "brown", (145, 91, 48)),
    DemoItem("demo-bag-red", "bag", "Red bag", "red", (198, 56, 67)),
    DemoItem(
        "demo-accessories-hat",
        "accessories",
        "Yellow hat",
        "yellow",
        (238, 196, 61),
    ),
    DemoItem(
        "demo-accessories-scarf",
        "accessories",
        "Orange scarf",
        "orange",
        (229, 126, 53),
    ),
]

EVALUATION_QUERIES = [
    {
        "query_id": "query-top-blue-shirt",
        "image_path": "queries/query-top-blue-shirt.jpg",
        "category": "top",
        "relevant_item_ids": ["demo-top-blue-shirt", "demo-top-red-shirt"],
        "draw_as": "demo-top-blue-shirt",
    },
    {
        "query_id": "query-bottom-blue-jeans",
        "image_path": "queries/query-bottom-blue-jeans.jpg",
        "category": "bottom",
        "relevant_item_ids": ["demo-bottom-blue-jeans", "demo-bottom-black-pants"],
        "draw_as": "demo-bottom-blue-jeans",
    },
    {
        "query_id": "query-dress-green",
        "image_path": "queries/query-dress-green.jpg",
        "category": "dress",
        "relevant_item_ids": ["demo-dress-green", "demo-dress-purple"],
        "draw_as": "demo-dress-green",
    },
    {
        "query_id": "query-shoes-white",
        "image_path": "queries/query-shoes-white.jpg",
        "category": "shoes",
        "relevant_item_ids": ["demo-shoes-white", "demo-shoes-black"],
        "draw_as": "demo-shoes-white",
    },
    {
        "query_id": "query-bag-brown",
        "image_path": "queries/query-bag-brown.jpg",
        "category": "bag",
        "relevant_item_ids": ["demo-bag-brown", "demo-bag-red"],
        "draw_as": "demo-bag-brown",
    },
    {
        "query_id": "query-accessories-hat",
        "image_path": "queries/query-accessories-hat.jpg",
        "category": "accessories",
        "relevant_item_ids": ["demo-accessories-hat", "demo-accessories-scarf"],
        "draw_as": "demo-accessories-hat",
    },
]


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    return parser.parse_args(argv)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _base_canvas() -> Image.Image:
    image = Image.new("RGB", IMAGE_SIZE, color=(246, 248, 252))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 28, 356, 484), radius=28, fill=(255, 255, 255))
    return image


def _draw_label(draw: ImageDraw.ImageDraw, item: DemoItem, *, query: bool) -> None:
    title = f"{item.category.upper()} / {item.color_name}"
    subtitle = "query image" if query else item.item_id
    draw.text((42, 42), title, fill=(25, 35, 55), font=_font(22))
    draw.text((42, 454), subtitle, fill=(92, 104, 124), font=_font(18))


def _draw_top(draw: ImageDraw.ImageDraw, color: tuple[int, int, int]) -> None:
    points = [
        (128, 120),
        (256, 120),
        (302, 180),
        (262, 216),
        (244, 188),
        (244, 340),
        (140, 340),
        (140, 188),
        (122, 216),
        (82, 180),
    ]
    draw.polygon(points, fill=color)


def _draw_bottom(draw: ImageDraw.ImageDraw, color: tuple[int, int, int]) -> None:
    draw.rounded_rectangle((142, 122, 242, 190), radius=18, fill=color)
    draw.polygon([(148, 186), (190, 186), (176, 366), (122, 366)], fill=color)
    draw.polygon([(198, 186), (238, 186), (270, 366), (216, 366)], fill=color)


def _draw_dress(draw: ImageDraw.ImageDraw, color: tuple[int, int, int]) -> None:
    draw.polygon([(156, 116), (228, 116), (274, 366), (110, 366)], fill=color)
    draw.polygon([(156, 116), (122, 172), (154, 194), (178, 132)], fill=color)
    draw.polygon([(228, 116), (262, 172), (230, 194), (206, 132)], fill=color)


def _draw_shoes(draw: ImageDraw.ImageDraw, color: tuple[int, int, int]) -> None:
    outline = (36, 45, 60)
    draw.rounded_rectangle(
        (88, 250, 188, 304),
        radius=22,
        fill=color,
        outline=outline,
        width=4,
    )
    draw.rounded_rectangle(
        (202, 250, 302, 304),
        radius=22,
        fill=color,
        outline=outline,
        width=4,
    )
    draw.line((104, 284, 176, 284), fill=outline, width=3)
    draw.line((218, 284, 290, 284), fill=outline, width=3)


def _draw_bag(draw: ImageDraw.ImageDraw, color: tuple[int, int, int]) -> None:
    draw.arc((132, 112, 252, 236), 180, 360, fill=(50, 55, 70), width=8)
    draw.rounded_rectangle((106, 190, 278, 350), radius=26, fill=color)
    draw.rectangle((128, 214, 256, 238), fill=(255, 255, 255))


def _draw_accessories(draw: ImageDraw.ImageDraw, item: DemoItem) -> None:
    if "hat" in item.item_id:
        draw.ellipse((126, 160, 258, 252), fill=item.color)
        draw.rounded_rectangle((82, 238, 302, 270), radius=16, fill=item.color)
    else:
        draw.rounded_rectangle((118, 142, 266, 196), radius=22, fill=item.color)
        draw.polygon([(150, 192), (198, 192), (172, 360), (120, 360)], fill=item.color)
        draw.polygon([(204, 192), (246, 192), (292, 350), (242, 360)], fill=item.color)


def draw_item(item: DemoItem, *, query: bool = False) -> Image.Image:
    image = _base_canvas()
    draw = ImageDraw.Draw(image)
    if item.category == "top":
        _draw_top(draw, item.color)
    elif item.category == "bottom":
        _draw_bottom(draw, item.color)
    elif item.category == "dress":
        _draw_dress(draw, item.color)
    elif item.category == "shoes":
        _draw_shoes(draw, item.color)
    elif item.category == "bag":
        _draw_bag(draw, item.color)
    elif item.category == "accessories":
        _draw_accessories(draw, item)
    _draw_label(draw, item, query=query)
    return image


def create_demo_data(root: Path) -> None:
    catalog_dir = root / "catalog"
    images_dir = catalog_dir / "images"
    evaluation_dir = root / "evaluation"
    queries_dir = evaluation_dir / "queries"
    images_dir.mkdir(parents=True, exist_ok=True)
    queries_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    for item in CATALOG_ITEMS:
        draw_item(item).save(images_dir / f"{item.item_id}.jpg", quality=92)
        metadata.append(
            {
                "item_id": item.item_id,
                "category": item.category,
                "brand": "ClothLens Demo",
                "color": item.color_name,
                "name": item.name,
                "image_url": "",
            }
        )
    (catalog_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    items_by_id = {item.item_id: item for item in CATALOG_ITEMS}
    manifest = []
    for query in EVALUATION_QUERIES:
        item = items_by_id[str(query["draw_as"])]
        query_path = evaluation_dir / str(query["image_path"])
        draw_item(item, query=True).save(query_path, quality=92)
        manifest.append(
            {
                "query_id": query["query_id"],
                "image_path": query["image_path"],
                "category": query["category"],
                "relevant_item_ids": query["relevant_item_ids"],
            }
        )
    (evaluation_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    arguments = parse_arguments(argv)
    create_demo_data(Path(arguments.root))
    print(f"Demo data written to {Path(arguments.root).resolve()}")


if __name__ == "__main__":
    main()
