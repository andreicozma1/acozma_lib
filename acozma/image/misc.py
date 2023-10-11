
import numpy as np
from skimage import filters


def get_gradients(image: np.ndarray):
    # get vertical and horizontal gradients
    grad_h = filters.sobel_h(image)
    grad_v = filters.sobel_v(image)

    # get gradient magnitude and angle
    grad_mag = np.sqrt(np.square(grad_h) + np.square(grad_v))
    # grad_mag = grad_mag / np.max(grad_mag)

    grad_angle = np.arctan2(grad_v, grad_h)

    return grad_h, grad_v, grad_mag, grad_angle


def get_lines(grad_mag, grad_angle, threshold: float = 0.0):
    vertical_lines = np.zeros_like(grad_mag)
    horizontal_lines = np.zeros_like(grad_mag)
    diagonal_lines = np.zeros_like(grad_mag)
    anti_diagonal_lines = np.zeros_like(grad_mag)

    for i in range(grad_mag.shape[0]):
        for j in range(grad_mag.shape[1]):
            if grad_mag[i, j] < threshold:
                continue

            angle = grad_angle[i, j]
            if angle < 0:
                angle = angle + 2 * np.pi

            # vertical
            if np.abs(angle - np.pi / 2) < np.pi / 8:
                vertical_lines[i, j] = grad_mag[i, j]

            # horizontal
            elif np.abs(angle - np.pi) < np.pi / 8 or np.abs(angle - 0) < np.pi / 8:
                horizontal_lines[i, j] = grad_mag[i, j]

            # diagonal
            elif np.abs(angle - np.pi / 4) < np.pi / 8:
                diagonal_lines[i, j] = grad_mag[i, j]

            # anti-diagonal
            elif np.abs(angle - 3 * np.pi / 4) < np.pi / 8:
                anti_diagonal_lines[i, j] = grad_mag[i, j]

    return vertical_lines, horizontal_lines, diagonal_lines, anti_diagonal_lines
