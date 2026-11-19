"""Official AP-10K full-image classification splits; annotations stay eval-only."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class AP10KDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        *,
        split_id=1,
        transforms=None,
        annotation_dir="annotations",
        images_dir="data",
        num_classes=None,
    ):
        if split not in {"train", "val", "test"} or split_id not in {1, 2, 3}:
            raise ValueError("Use official train/val/test and split_id 1, 2, or 3")
        self.root, self.transform = Path(root), transforms
        documents = {}
        for name in ("train", "val", "test"):
            path = self.root / annotation_dir / f"ap10k-{name}-split{split_id}.json"
            documents[name] = json.loads(path.read_text())
        categories = documents["train"]["categories"]
        if any(doc["categories"] != categories for doc in documents.values()):
            raise ValueError("AP-10K category metadata differs across official splits")
        active = {
            a["category_id"] for doc in documents.values() for a in doc["annotations"]
        }
        self.categories = [c for c in categories if c["id"] in active]
        self.classes = [c["name"] for c in self.categories]
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        lookup = {c["id"]: i for i, c in enumerate(self.categories)}
        if num_classes is not None and num_classes != len(self.classes):
            raise ValueError(
                f"dataset.num_classes={num_classes}, but AP-10K has {len(self.classes)} represented species"
            )
        seen = set()
        for doc in documents.values():
            ids = [i["id"] for i in doc["images"]]
            names = [i["file_name"] for i in doc["images"]]
            keys = {("id", i) for i in ids} | {("name", n) for n in names}
            if (
                len(set(ids)) != len(ids)
                or len(set(names)) != len(names)
                or seen & keys
            ):
                raise ValueError("Duplicate images or leakage between AP-10K splits")
            seen |= keys
        self.annotations = documents[split]
        by_image = defaultdict(list)
        for annotation in self.annotations["annotations"]:
            by_image[annotation["image_id"]].append(annotation)
        self.samples, self.image_ids = [], []
        for image in sorted(self.annotations["images"], key=lambda row: row["id"]):
            species = {a["category_id"] for a in by_image[image["id"]]}
            if len(species) != 1:
                raise ValueError(
                    f"Image {image['id']} has missing or ambiguous species labels"
                )
            path = self.root / images_dir / image["file_name"]
            if not path.is_file():
                raise FileNotFoundError(path)
            self.samples.append((str(path), lookup[next(iter(species))]))
            self.image_ids.append(image["id"])
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # No keypoints, boxes, or masks are returned to the learner.
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, label, index
