import os
import re
import string
from collections import defaultdict

from .decorators import listify
from .string import remove_longest_repeating_substring


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


@listify
def get_normalized_filename(fname: str, has_extension: bool) -> str:
    """
    Replaces dots with dashes and spaces with underscores in a filename.
    Throws an exception if the value of `filename` is a path.
    """
    if os.path.sep in fname:
        raise ValueError(f"The value of `filename` cannot be a path: `{fname}`")

    # print(f"Before: `{fname}`")

    # map of characters to replace and their replacements
    map_substitutions = {
        " ": "_",
        ".": "-",
    }
    # add every character in `string.punctuation` to the map if it is not already in the map
    for char in string.punctuation:
        if char not in map_substitutions and char not in list(
            map_substitutions.values()
        ):
            map_substitutions[char] = ""

    if has_extension:
        fname_noext, extension = os.path.splitext(fname)
    else:
        fname_noext, extension = fname, ""
    # replace characters in `map_substitutions` with their replacements
    for char, replacement in map_substitutions.items():
        fname_noext = fname_noext.replace(char, replacement)
    # replace multiple underscores with a single underscore
    for char in list(map_substitutions.values()):
        # fname_noext = re.sub(r"_{2,}", "_", fname_noext)
        if len(char) == 0:
            continue
        fname_noext = re.sub(rf"{char}{{2,}}", char, fname_noext)

    # finally, strip away any leading or trailing characters that are values in `map_substitutions`
    substitution_string = "".join(map_substitutions.values())
    # also strip away any whitespace or punctuation
    substitution_string = substitution_string + string.whitespace + string.punctuation
    # strip away any leading or trailing characters that are values in `map_substitutions`
    fname_noext = fname_noext.strip(substitution_string)

    # Re-add the extension
    fname_noext = fname_noext + extension
    # print(f"After: `{fname_noext}`")
    return fname_noext


def get_map_ext_to_fpath_from_list(
    files: list, remove_extensions: bool = False
) -> dict:
    map_ext_to_fpath = defaultdict(list)

    for file in files:
        # Get the file extension and create the full path
        file_noext, extension = os.path.splitext(file)

        map_ext_to_fpath[extension].append(file_noext if remove_extensions else file)

    return map_ext_to_fpath


def get_map_ext_to_fpath_from_path(dpath: str, recursive=True, **kwargs) -> dict:
    files = []
    # Walk the directory tree and add files to the list
    if recursive:
        for root, _, files_ in os.walk(dpath):
            for file in files_:
                fullpath = os.path.join(root, file)
                files.append(fullpath)
    else:
        for file in os.listdir(dpath):
            fullpath = os.path.join(dpath, file)
            # Only files (not directories) are added to the list
            if os.path.isfile(fullpath):
                files.append(fullpath)

    return get_map_ext_to_fpath_from_list(files, **kwargs)


def normalize_filenames(dpath: str, dry_run: bool = True):
    """
    Recursively normalize file names in a given folder.
    For each folder, a dictionary is created to hold the files by extension.
    For each set of files with the same extension, the file names are normalized to replace whitespace and certain symbols to underscores and dashes.
    The longest repeating substring occuring in the set of files is also removed from the file names. This is useful when certain files have a common prefix or suffix.
    By default, this function is run in dry run mode to prevent accidental renaming of files. To rename the files after verifying the output, set `dry_run=False`.
    """
    min_len = 5
    dry_run_msg = (
        "DRY RUN: Please verify the output and set `dry_run=False` to rename the files"
    )
    if dry_run:
        print(dry_run_msg)

    for root, _, files in os.walk(dpath):
        print("=" * 80)
        print(f"Root: `{root}`")
        # Create a dictionary to hold the files by extension
        map_ext_to_fpath = get_map_ext_to_fpath_from_list(files, remove_extensions=True)

        # Rename the files
        for extension, files_source in map_ext_to_fpath.items():
            print("-" * 80)
            print(f"Extension: `{extension}`")
            files_target = files_source.copy()
            files_target = get_normalized_filename(files_target, has_extension=False)
            files_target = remove_longest_repeating_substring(
                files_target, min_len=min_len
            )
            if files_source == files_target:
                print("\tSkip: No changes to be made")
                continue

            for fname_source, fname_target in zip(files_source, files_target):
                fpath_source = os.path.join(root, f"{fname_source}{extension}")
                fpath_target = os.path.join(root, f"{fname_target}{extension}")
                if os.path.exists(fpath_target):
                    print(f"\tSkip: `{fpath_source}` (target already exists)")
                    continue

                if fpath_source == fpath_target:
                    print(f"\tSkip: `{fpath_source}` (source and target are the same)")
                    continue
                print(f"\tRename: `{fpath_source}` -> `{fpath_target}`")
                if not dry_run:
                    os.rename(fpath_source, fpath_target)
    print("=" * 80)
    if dry_run:
        print(dry_run_msg)
    else:
        print("Done!")
