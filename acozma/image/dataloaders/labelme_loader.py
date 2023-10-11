import json
from pathlib import Path
from pprint import pprint
from typing import Callable, Optional

from .. import bbox
from .data_loader import ImageDatasetLoader
from .data_stats import ImageDatasetStats


class LabelMeLoader(ImageDatasetLoader):
    def __init__(self, base_dir: Path, ignore_orphans: bool = False, *args, **kwargs):
        ImageDatasetLoader.__init__(self, base_dir, *args, **kwargs)

        orphans = self.find_orphans()
        if orphans and not ignore_orphans:
            print(
                "WARN: Orphaned references found. You may pass the `ignore_orphans` argument to the constructor to ignore them."
            )
            pprint(orphans)

        if ignore_orphans:
            print("=" * 80)
            json_orphans = [p for p in self.json_names if p in orphans]
            img_orphans = [p for p in self.img_names if p in orphans]
            if json_orphans:
                print("INFO: Cleaning orphaned JSON references...")
                pprint(json_orphans)
                for p in json_orphans:
                    self.json_map.pop(p)
            if img_orphans:
                print("INFO: Cleaning orphaned image references...")
                pprint(img_orphans)
                for p in img_orphans:
                    self.img_map.pop(p)

    def find_orphans(self):
        json_fnames = set(self.json_names)
        img_fnames = set(self.img_names)

        return json_fnames.symmetric_difference(img_fnames)

    def __load_json(self, json_path: Path) -> dict:
        # TODO: Cache loaded JSON files
        with open(json_path, "r") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.json_paths)

    def __getitem__(self, index_or_slice: int | slice) -> dict | list[dict]:
        if isinstance(index_or_slice, int):
            return self.__load_json(self.json_paths[index_or_slice])

        if isinstance(index_or_slice, slice):
            return [self.__load_json(p) for p in self.json_paths[index_or_slice]]

        raise TypeError(f"Invalid index type: {type(index_or_slice)}")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_gt_bboxs(
        self,
        img: dict,
        filter: Optional[Callable[[dict], bool]] = None,
    ) -> list[bbox.BBox]:
        bboxs = []
        for shape in img["shapes"]:
            if filter and not filter(shape):
                continue
            label = shape["label"]
            points = shape["points"]

            x1, y1 = points[0]
            x2, y2 = points[1]
            bboxs.append(
                # bbox.BBox((x1, y1, x2, y2), mode=bbox.BBoxMode.XYXY, label=label)
                bbox.BBox.from_xyxy((x1, y1, x2, y2), label=label)
            )
        return bboxs

    def plot_anns(
        self,
        img: dict,
        bboxs_gt: list[bbox.BBox],
        bboxs_pred: Optional[list[bbox.BBox]] = None,
    ):
        import PIL
        from matplotlib import pyplot as plt

        print("Image:")
        pprint(img)

        file_name = img["imagePath"]
        file_path = self.get_img_path(file_name, strict=False, silent=False)
        if file_path is None:
            print("Skipping...")
            return

        I = PIL.Image.open(file_path).convert("RGB")

        _, ax = plt.subplots(1, 2, figsize=(15, 15))

        ax[0].imshow(I)
        ax[0].set_title("Original image")

        ax[1].imshow(I)
        ax[1].set_title("Annotated image")
        for shape in bboxs_gt:
            x, y, w, h = shape.xywh
            rect = plt.Rectangle(
                (x, y), w, h, fill=False, edgecolor="green", linewidth=1
            )
            ax[1].add_patch(rect)

        if bboxs_pred:
            for shape in bboxs_pred:
                x, y, w, h = shape.xywh
                rect = plt.Rectangle(
                    (x, y), w, h, fill=False, edgecolor="blue", linewidth=1
                )
                ax[1].add_patch(rect)

        plt.tight_layout()
        plt.show()

    def print_info(self):
        super().print_info()
        print("=" * 80)
        print("LabelMeLoader: Info")

        annot_labels = set()
        annot_label_types = set()
        annot_descriptions = set()
        annot_group_ids = set()
        sample_resolutions = set()

        annot_counts = []
        samples_with_no_annots = []

        for d in self:
            resolution = f"{d['imageWidth']}x{d['imageHeight']}"
            sample_resolutions.add(resolution)

            shapes = d["shapes"]
            num_shapes = len(shapes)
            annot_counts.append(num_shapes)
            if num_shapes == 0:
                samples_with_no_annots.append(d)
                continue

            for shape in shapes:
                annot_label_types.add(shape["shape_type"])
                annot_group_ids.add(shape["group_id"])
                annot_descriptions.add(shape["description"])
                annot_labels.add(shape["label"])

        resolution_mapping = {}

        for sr in sample_resolutions:
            w, h = sr.split("x")
            wh = int(w) * int(h)
            resolution_mapping[wh] = sr

        print("Total number of samples:", len(self))
        print("  - With annotations:", len(self) - len(samples_with_no_annots))
        print("  - With no annotations:", len(samples_with_no_annots))
        print("Total # of annotations:", sum(annot_counts))

        print("Number of annotations/sample")
        print("  - Min:", min(annot_counts))
        print("  - Max:", max(annot_counts))
        print("  - Avg:", sum(annot_counts) / len(annot_counts))

        print("-" * 80)
        print("Unique # of sample resolutions:", len(sample_resolutions))
        print("  - Lowest:", resolution_mapping[max(resolution_mapping.keys())])
        print(
            "  - Highest:",
            resolution_mapping[min(resolution_mapping.keys())],
        )

        print("-" * 80)
        print("# Labels & Annotations")
        print("Unique annotation labels:", annot_labels)
        print("Unique annotation shape types:", annot_label_types)

        print()
        print("-" * 80)
        print("Samples with no annotations:", samples_with_no_annots)
        print("-" * 80)
        print("Unique sample resolutions:", sample_resolutions)
        print("-" * 80)

    def get_stats(self):
        dataset_stats = ImageDatasetStats()

        for d in self:
            dataset_stats.img_areas.append(d["imageWidth"] * d["imageHeight"])
            dataset_stats.img_widths.append(d["imageWidth"])
            dataset_stats.img_heights.append(d["imageHeight"])

            shapes = d["shapes"]
            num_shapes = len(shapes)
            dataset_stats.bbox_counts.append(num_shapes)
            if num_shapes == 0:
                continue
            for shape in shapes:
                match shape["shape_type"]:
                    case "rectangle":
                        points = shape["points"]
                        pw, ph = abs(points[1][0] - points[0][0]), abs(
                            points[1][1] - points[0][1]
                        )
                        dataset_stats.bbox_areas.append(pw * ph)
                        dataset_stats.bbox_widths.append(pw)
                        dataset_stats.bbox_heights.append(ph)
                    case _:
                        raise NotImplementedError(
                            f"Unsupported shape type: {shape['shape_type']}"
                        )

        return dataset_stats
