import os
from pathlib import Path


class ImageDatasetLoader:
    def __init__(
        self,
        base_path: Path,
        valid_img_exts: list[str] = ["jpg", "jpeg", "png"],
    ):
        self.base_img_dir: Path = base_path
        self.valid_img_exts: list[str] = valid_img_exts
        self.valid_json_exts: list[str] = ["json"]
        print("=" * 80)
        print("DatasetLoader: Initializing")
        print(" - base_img_dir:", self.base_img_dir)

        assert os.path.isdir(
            self.base_img_dir
        ), f"Path does not exist or is not a directory: {self.base_img_dir}"

        self.json_map: dict[str, Path] = self.__create_unique_mapping(
            self.__recursive_find(self.base_img_dir, self.valid_json_exts)
        )
        self.__print_info_json()

        self.img_map: dict[str, Path] = self.__create_unique_mapping(
            self.__recursive_find(self.base_img_dir, self.valid_img_exts)
        )
        self.__print_info_img()

        print("-" * 80)

    @property
    def json_names(self) -> list[str]:
        return list(self.json_map.keys())

    @property
    def img_names(self) -> list[str]:
        return list(self.img_map.keys())

    @property
    def json_paths(self) -> list[Path]:
        return list(self.json_map.values())

    @property
    def img_paths(self) -> list[Path]:
        return list(self.img_map.values())

    def get_img_path(self, pattern: str, strict=True, silent=False) -> Path | None:
        # if pattern is a path, we only want the filename
        name = os.path.splitext(os.path.basename(pattern.replace("\\", "/")))[0]

        if name in self.img_map:
            return self.img_map[name]

        msg = f"Could not find image path for: {pattern}"
        if strict:
            raise ValueError(msg)

        if not silent:
            print("WARN:", msg)
        return None

    def __recursive_find(self, path: Path, exts: list[str]) -> list[Path]:
        print("-" * 80)
        print("Searching:", exts)
        paths: list[Path] = []
        for ext in exts:
            paths.extend(list(list(Path(path).rglob(f"*.{ext}"))))

        paths.sort(key=lambda p: os.path.basename(p))
        assert paths, f"No files found with exts: {exts}"
        return paths

    def __create_unique_mapping(self, paths: list[Path]) -> dict[str, Path]:
        mapping = {}
        names = [os.path.splitext(os.path.basename(f))[0] for f in paths]
        for name, path in zip(names, paths):
            if name in mapping:
                msg = (
                    f"Found duplicate names while creating mapping for {name}:\n"
                    f" - Existing: {mapping[name]}\n"
                    f" - New: {path}"
                )
                if os.path.getsize(path) == os.path.getsize(mapping[name]):
                    print(f"INFO: {msg}\n - File sizes are equal, ignoring new path")
                    continue
                raise ValueError(msg)
            mapping[name] = path
        return mapping

    def __print_info_json(self):
        print("JSONs:")
        print(" - total paths:", len(self.json_names))
        print(" - unique names:", len(set(self.json_names)))

    def __print_info_img(self):
        print("IMGs:")
        print(" - total paths:", len(self.img_names))
        print(" - unique names:", len(set(self.img_names)))

    def print_info(self):
        print("=" * 80)
        print("DatasetLoader: Info")
        self.__print_info_json()
        self.__print_info_img()

    def __repr__(self):
        return f"DatasetLoader({self.base_img_dir})"

    def __str__(self):
        return f"DatasetLoader({self.base_img_dir})"
