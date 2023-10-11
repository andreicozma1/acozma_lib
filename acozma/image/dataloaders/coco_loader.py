from pathlib import Path
from pprint import pprint
from typing import Callable, Optional

import numpy as np
from pycocotools.coco import COCO
from typeguard import typechecked

from .data_loader import ImageDatasetLoader


class COCOLoader(ImageDatasetLoader):
    def __init__(
        self,
        base_dir: Path,
        index: Optional[int] = None,
        ignore_orphans: bool = False,
        *args,
        **kwargs,
    ):
        ImageDatasetLoader.__init__(self, base_dir, *args, **kwargs)

        if len(self.json_paths) > 1:
            msg_options = (
                "Available Options:\n"
                + "\n".join(
                    f"  {i}: {path}" for i, path in enumerate(self.json_paths)
                ).rstrip()
            )

            if index is None:
                raise ValueError(
                    f"Multiple JSON files found. Please pass the `index` argument to the constructor to select one of the following indices:\n{msg_options}"
                )

            if index >= len(self.json_paths):
                raise ValueError(
                    f"Invalid index {index}. Please select one of the following indices:\n{msg_options}"
                )
            print(msg_options)
            print(f"Selected JSON file at index: {index}")
            self.coco_json_path = self.json_paths[index]
        elif len(self.json_paths) == 1:
            self.coco_json_path = self.json_paths[0]

        print("COCO JSON path:", self.coco_json_path)
        print("=" * 80)
        self.coco = COCO(self.coco_json_path)

        orphans = self.find_orphans()
        if orphans and not ignore_orphans:
            print(
                "WARN: Orphaned references found. You may pass the `ignore_orphans` argument to the constructor to ignore them."
            )
            pprint(orphans)

        if ignore_orphans:
            print("=" * 80)
            print("INFO: Cleaning orphaned image references...")
            self.coco.dataset["images"] = [
                img for img in self.coco.dataset["images"] if img["id"] not in orphans
            ]
            print("INFO: Cleaning orphaned annotation references...")
            self.coco.dataset["annotations"] = [
                ann
                for ann in self.coco.dataset["annotations"]
                if ann["image_id"] not in orphans
            ]
            self.coco.createIndex()

        self.catIdToName = {
            cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }

    def find_orphans(self):
        orphans = []
        for imgId, img in self.coco.imgs.items():
            img_path = self.get_img_path(img["file_name"], strict=False, silent=True)
            if img_path is None:
                orphans.append(imgId)

        return orphans

    def __len__(self) -> int:
        return len(self.coco.imgs)

    def __getitem__(self, index_or_slice: int | slice) -> dict | list[dict]:
        imgIds = list(self.coco.imgs.keys())[index_or_slice]

        if isinstance(index_or_slice, int):
            return self.coco.imgs[imgIds]

        if isinstance(index_or_slice, slice):
            return [self.coco.imgs[imgId] for imgId in imgIds]

        raise TypeError(f"Invalid index type: {type(index_or_slice)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def convertToLabelMe(
        self,
        imgIds: Optional[list[int] | int] = None,
        catIds: Optional[list[int] | int] = None,
    ):
        if not catIds:
            catIds = list(self.coco.cats.keys())
        if not isinstance(catIds, list):
            catIds = [catIds]
        print(f"catIds: {len(catIds)}")

        if not imgIds:
            imgIds = list(self.coco.imgs.keys())
        if not isinstance(imgIds, list):
            imgIds = [imgIds]
        print(f"Initial imgIds: {len(imgIds)}")

        # filtering
        filteredImgIds = set()
        for catId in catIds:
            filteredImgIds.update(self.coco.catToImgs[catId])
        imgIds = filteredImgIds.intersection(imgIds)

        print(f"Filtered imgIds: {len(imgIds)}")
        # load image data
        img_data_orig = self.coco.loadImgs(imgIds)

        remove_keys = ["coco_url", "data_captured", "flickr_url", "license"]
        for d in img_data_orig:
            for k in remove_keys:
                d.pop(k, None)

        img_data_new = []
        for d in img_data_orig:
            imagePath = self.get_img_path(d["file_name"], strict=False, silent=False)
            if imagePath is None:
                print("Skipping...")
                continue

            annIds = self.coco.getAnnIds(imgIds=d["id"], catIds=catIds, iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            shapes = []
            for ann in anns:
                contour = np.array(ann["segmentation"][0], dtype=np.int32)
                contour = contour.reshape((-1, 2))
                x0, y0 = np.min(contour, axis=0)
                x1, y1 = np.max(contour, axis=0)
                shapes.append(
                    {
                        "label": self.catIdToName[ann["category_id"]],
                        "points": [[int(x0), int(y0)], [int(x1), int(y1)]],
                        "group_id": None,
                        "description": "",
                        "shape_type": "rectangle",
                        "flags": {},
                    }
                )
            nd = {
                "version": "5.3.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": d["file_name"],
                "imageData": None,
                "imageHeight": d["height"],
                "imageWidth": d["width"],
            }
            img_data_new.append(nd)

        return img_data_new

    @typechecked
    def plot_anns(
        self,
        img: dict,
        filter: Optional[Callable[[dict], bool]] = None,
    ):
        import PIL
        from matplotlib import pyplot as plt

        print("Image:")
        pprint(img)

        file_name = img["file_name"]
        file_path = self.get_img_path(file_name, strict=False, silent=False)
        if file_path is None:
            print("Skipping...")
            return

        annIds = self.coco.getAnnIds(imgIds=img["id"])
        anns = self.coco.loadAnns(annIds)
        if filter:
            anns = [ann for ann in anns if filter(ann)]
        print("Annotations:")
        pprint(anns)

        I = PIL.Image.open(file_path).convert("RGB")

        _, ax = plt.subplots(1, 2, figsize=(15, 15))

        ax[0].imshow(I)
        ax[0].set_title("Original image")

        ax[1].imshow(I)
        ax[1].set_title("Annotated image")
        self.coco.showAnns(anns)

        plt.tight_layout()
        plt.show()

    def print_info(self):
        super().print_info()
        allCatIds = self.coco.getCatIds()
        allCats = self.coco.loadCats(allCatIds)
        # catIdToName = {cat["id"]: cat["name"] for cat in allCats}
        print("-" * 80)
        print("COCO Info:")
        self.coco.info()

        print("-" * 80)
        print("COCO Categories:")

        for cat in allCats:
            catId = cat["id"]
            imgIds = self.coco.getImgIds(catIds=catId)
            imgs = self.coco.loadImgs(imgIds)
            imgs_found = []
            for img in imgs:
                img_path = self.get_img_path(
                    img["file_name"], strict=False, silent=True
                )
                if img_path is not None:
                    imgs_found.append(img)
            print(
                {
                    **cat,
                    "num_imgs": len(imgIds),
                    "num_imgs_found": len(imgs_found),
                }
            )
