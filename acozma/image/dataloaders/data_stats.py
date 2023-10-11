import numpy as np
from matplotlib import pyplot as plt


class ImageDatasetStats:
    def __init__(self):
        self.img_areas = []
        self.img_widths = []
        self.img_heights = []

        self.bbox_areas = []
        self.bbox_widths = []
        self.bbox_heights = []

        self.bbox_counts = []

        self.style_kwargs = {
            "color": "skyblue",
            "alpha": 0.5,
            "edgecolor": "black",
        }

    def plot_annot_counts(
        self,
        ax=None,
    ):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(15, 5))

        unique_counts, counts = np.unique(self.bbox_counts, return_counts=True)

        ax.bar(
            unique_counts,
            counts,
            **self.style_kwargs,
        )

        ax.set_title("Annotation Counts")
        ax.set_xlabel("Number of Annotations")
        ax.set_ylabel("Frequency")

        ax.set_xticks(range(0, max(unique_counts) + 1))

        if ax is None:
            plt.tight_layout()
            plt.show()

    def plot_image_dims(
        self,
        num_bins: int = 50,
        ax=None,
    ):
        if ax is None:
            _, ax = plt.subplots(1, 3, figsize=(25, 5))

        ax[0].hist(
            self.img_areas,
            bins=num_bins,
            **self.style_kwargs,
        )

        ax[0].set_title("Image Areas")
        ax[0].set_xlabel("Area (pixels)")
        ax[0].set_ylabel("Frequency")

        ax[1].hist(
            self.img_widths,
            bins=num_bins,
            **self.style_kwargs,
        )

        ax[1].set_title("Image Widths")
        ax[1].set_xlabel("Width (pixels)")
        ax[1].set_ylabel("Frequency")

        ax[2].hist(
            self.img_heights,
            bins=num_bins,
            **self.style_kwargs,
        )

        ax[2].set_title("Image Heights")
        ax[2].set_xlabel("Height (pixels)")
        ax[2].set_ylabel("Frequency")

        if ax is None:
            plt.tight_layout()
            plt.show()

    def plot_bbox_dims(
        self,
        num_bins: int = 50,
        ax=None,
    ):
        if ax is None:
            _, ax = plt.subplots(1, 3, figsize=(25, 5))

        ax[0].hist(
            self.bbox_areas,
            bins=num_bins,
            **self.style_kwargs,
        )

        ax[0].set_title("BBox Areas")
        ax[0].set_xlabel("Area (pixels)")
        ax[0].set_ylabel("Frequency")

        ax[1].hist(
            self.bbox_widths,
            bins=num_bins,
            **self.style_kwargs,
        )

        ax[1].set_title("BBox Widths")
        ax[1].set_xlabel("Width (pixels)")
        ax[1].set_ylabel("Frequency")

        ax[2].hist(
            self.bbox_heights,
            bins=num_bins,
            **self.style_kwargs,
        )

        ax[2].set_title("BBox Heights")
        ax[2].set_xlabel("Height (pixels)")
        ax[2].set_ylabel("Frequency")

        if ax is None:
            plt.tight_layout()
            plt.show()
