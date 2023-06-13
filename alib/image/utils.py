from typing import List, Tuple

import numpy as np
from skimage import exposure
from skimage.metrics import (
    mean_squared_error,
    normalized_mutual_information,
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity,
)


def slice_image(image: np.ndarray, num_slices: int = 5) -> List[np.ndarray]:
    """
    Slices an image into `num_slices` equally-sized slices along the first axis.
    Note: If the image cannot be divided into `num_slices` slices of equal size,
    the last slice will be the remainder of the image wrapped around to the
    beginning of the image to make up for the difference.

    Args:
        image: A 2D numpy array.
        num_slices: The number of slices to divide the image into.

    Returns:
        A list of 2D numpy arrays.
    """
    all_slices = []
    # The length of each slice
    slice_len = image.shape[0] // num_slices
    for slice_start in range(0, image.shape[0], slice_len):
        # Calculate the end index of the slice
        slice_end = slice_start + slice_len
        if slice_end > image.shape[0]:
            # Wrap around to the beginning of the image if necessary
            slice_end -= image.shape[0]
            slice_i = np.concatenate(
                (image[slice_start:, :], image[:slice_end, :]), axis=0
            )
        else:
            # Slice the image normally
            slice_i = image[slice_start:slice_end, :]

        all_slices.append(slice_i)

    return all_slices


def topk_matches(
    target: np.ndarray, variations: List[np.ndarray], k: int = 5
) -> List[Tuple[np.ndarray, float]]:
    """
    Finds the top `k` most similar images to `target` from `variations`.
    The similarity metric used is structural similarity.

    Args:
        target: A 2D numpy array to use as the target image.
        variations: A list of 2D numpy arrays from which to find the top matches.
        k: The number of top matches to return from `variations`.

    Returns:
        A list of tuples containing the top `k` matches and their similarity scores.
    """
    image_scores = []

    for variation in variations:
        # Skip the target image
        if np.array_equal(target, variation):
            continue

        data_range = target.max() - target.min()
        # Calculate the structural similarity between the target and variation
        score = structural_similarity(
            target, variation, data_range=data_range, gaussian_weights=True, sigma=1.5
        )
        image_scores.append((variation, score))

    # Sort the images by their similarity score
    image_scores.sort(key=lambda x: x[1], reverse=True)
    # Return the top k matches
    return image_scores[:k]


def topk_average(
    target: np.ndarray,
    variations: List[np.ndarray],
    k: int = 5,
    return_metrics: bool = False,
) -> np.ndarray:
    """
    Finds the top `k` most similar images to `target` from `variations` and returns their average.
    Optionally, returns various scores to assess the similarity of the average image to the target.

    Args:
        target: A 2D numpy array.
        variations: A list of 2D numpy arrays.
        k: The number of top matches to return.
        return_metrics: Whether to also return the similarity metrics of the average image.

    Returns:
        The average of the top `k` most similar images to `target`.
        If `return_metrics` is True, also returns the similarity metrics of the average image.
    """
    matches = topk_matches(target, variations, k=k)
    matches = [x[0] for x in matches]

    average = np.mean(matches, axis=0)
    average = exposure.match_histograms(average, target)

    if return_metrics:
        metrics = get_similarity_scores(target, average)
        return (average, metrics)
    else:
        return average


def remove_background(
    image: np.ndarray,
    scales: list = [8, 10, 12],
    top_k: int = 4,
    power: float = 1.0,
    rescale: bool = True,
) -> np.ndarray:
    """
    Removes the background of an image with a repeating texture pattern using multiple passes at different scales.

    This function uses a multi-pass algorithm to remove the periodic texture background from an image. It iterates
    through the provided scales, dividing the image into vertical slices and identifying the top `k` most similar
    slices for each slice. The average of these top `k` slices is then subtracted from the original slice to remove
    the background. The background-removed image is raised to the power of `power` to enhance the separation between
    background and foreground. The process is repeated for each scale to further refine the results.

    Args:
        image: np.ndarray
            A 2D numpy array representing the input image.
        scales: list, optional, default: [8, 10, 12]
            A list of integers specifying the number of slices to divide the image into at each pass.
            The length of this list determines the number of passes to make over the image.
        top_k: int, optional, default: 4
            The number of top matches to the current slice to consider when removing the background.
            A higher value will result in a more accurate background removal, but will also be more computationally expensive.
        power: float, optional, default: 1.0
            The power to raise the background-removed image to before subtracting it from the current slice.
            This enhances the separation of remaining features (defects) from any remaining texture parts.
        rescale: bool, optional, default: True
            Whether to rescale the output image to [0, 1]. Typically desirable for further processing.

    Returns:
        np.ndarray: The background-removed image.

    """
    image_out = np.zeros_like(image)

    # Perform several passes over the image at different scales
    for sr in scales:
        # Divide the image into the number of slices specified by `sr`
        slices = slice_image(image, num_slices=sr)

        # Perform a vertical scan over the image (slice by slice)
        slice_start = 0
        for curr_slice in slices:
            slice_len = curr_slice.shape[0]
            # Find the top `k` most similar slices from the other slices in the current pass
            slice_texture_avg = topk_average(curr_slice, slices, k=top_k)
            # Subtract the average of the top `k` similar slices from the current slice to remove the background
            slice_bg_removed = np.clip(curr_slice - slice_texture_avg, 0.0, 1.0)
            # Enhance separation between the background and the foreground
            slice_bg_removed = np.power(slice_bg_removed, power)

            slice_end = slice_start + slice_len

            # Place the background-removed image at the correct position in the output image
            if slice_end > image_out.shape[0]:
                # For cases where the number of slices is not a multiple of the image height
                slice_end -= image_out.shape[0]
                # The corresponding slice from the full background-removed image
                # Wrap the current slice around from bottom back to top by concatenating
                slice_out_curr = np.concatenate(
                    (image_out[slice_start:, :], image_out[:slice_end, :]), axis=0
                )
                # Combine the current estimate with the previous estimate to get a better estimate
                slice_out_new = np.mean([slice_out_curr, slice_bg_removed], axis=0)
                # Place the new estimate back into the output image
                # The part of the slice that doesn't wrap around
                image_out[slice_start:, :] = slice_out_new[
                    : image_out.shape[0] - slice_start
                ]
                # The part of the slice that wraps around to the top
                image_out[:slice_end, :] = slice_out_new[
                    image_out.shape[0] - slice_start :
                ]
            else:
                # The corresponding slice from the full background-removed image
                slice_out_curr = image_out[slice_start:slice_end, :]
                # Combine the current estimate with the previous estimate to get a better estimate
                slice_out_new = np.mean([slice_out_curr, slice_bg_removed], axis=0)
                # Place the new estimate back into the output image
                image_out[slice_start:slice_end, :] = slice_out_new

            # Move to the next slice
            slice_start += slice_len

    # Rescale the output image to [0, 1]
    # This helps maintain consistency for further processing
    if rescale:
        image_out = min_max_scale(image_out)

    # The output image will be a reconstruction of the original image with the periodic texture removed
    # and the defects enhanced
    return image_out


def min_max_scale(image):
    min_val, max_val = np.min(image), np.max(image)
    return (image - min_val) / (max_val - min_val)


def nextpow2(N):
    n = 1
    while n < N:
        n *= 2
    return n


def get_similarity_scores(image_true, image_test):
    data_range = image_true.max() - image_true.min()
    score_ssim = structural_similarity(
        image_true,
        image_test,
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
    )
    score_mutual_info = normalized_mutual_information(image_true, image_test)
    score_psnr = peak_signal_noise_ratio(image_true, image_test, data_range=data_range)
    score_mse = mean_squared_error(image_true, image_test)
    score_nrmse = normalized_root_mse(image_true, image_test)

    return {
        "ssim": score_ssim,
        "norm_mutual_info": score_mutual_info,
        "psnr": score_psnr,
        "mse": score_mse,
        "norm_rmse": score_nrmse,
    }


def print_metrics(metrics):
    divider = " | "
    string = "".join(
        f"{key.upper()}: {value:.3f}{divider}" for key, value in metrics.items()
    )
    print(f"Similarity Scores: {string[: -len(divider)]}")


def print_info(image, start=None):
    info = f"min: {np.min(image):.3f} | max: {np.max(image):.3f} | shape: {np.shape(image)}"
    print(f"{start or '':>10}: {info}")
    return info
