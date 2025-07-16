import numpy as np
import matplotlib.pyplot as plt
import pyfftw  # fancy and fast ffts
import numba


# Try to import CuPy (if available)
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# User choice (default is False, user can enable GPU manually)
USE_GPU = False  # Set to True if the user wants GPU acceleration


def _stack_fft2_gpu(input_stack, direction="forward"):
    """
    Perform the 2D FFT on a stack of images using CuPy (CUDA).
    """
    input_stack = cp.asarray(input_stack, dtype=cp.complex64)  # Move data to GPU

    if direction == "forward":
        out_stack = cp.fft.fft2(input_stack, axes=(-2, -1))  # GPU FFT
    else:
        out_stack = cp.fft.ifft2(input_stack, axes=(-2, -1))  # GPU Inverse FFT

    return out_stack  # CuPy array (still on GPU)


def _stack_fft2_cpu(input_stack, direction="FFTW_FORWARD"):
    """
    Perform the 2D FFT on a stack of images using complex64 for faster computation.
    """
    input_stack = input_stack.astype(np.complex64)  # Convert input to complex64

    # Allocate memory-aligned output array
    out_stack = pyfftw.empty_aligned(input_stack.shape, dtype=np.complex64)

    fft = pyfftw.FFTW(
        input_array=input_stack,
        output_array=out_stack,
        axes=(-2, -1),
        direction=direction,
        flags=("FFTW_MEASURE",),
        threads=8,
    )

    fft()
    return out_stack


def _stack_fft2(input_stack, direction, use_gpu=False):
    if use_gpu:
        if not CUPY_AVAILABLE:
            raise ValueError(
                "Cupy is not installed, please install it before using GPU."
            )
        return _stack_fft2_gpu(input_stack, direction="FFTW_FORWARD")
    else:
        return _stack_fft2_cpu(input_stack, direction="FFTW_FORWARD")


def stack_fft2(input_stack, use_gpu=False):
    return _stack_fft2(input_stack, direction="FFTW_FORWARD", use_gpu=use_gpu)


def stack_ifft2(input_stack, use_gpu=False):
    return _stack_fft2(input_stack, direction="FFTW_BACKWARD", use_gpu=use_gpu)


# def stack_fft2(input_stack):
#     return _stack_fft2(input_stack.astype(np.complex64), direction="FFTW_FORWARD")


# def stack_ifft2(input_stack):
#     return _stack_fft2(input_stack.astype(np.complex64), direction="FFTW_BACKWARD")


@numba.njit(parallel=True)
def _stack_roll_cpu(img_stack, row_shift=0, col_shift=0):
    """
    Roll a stack of images manually since Numba doesn't support np.roll(axis=(-2,-1)).
    """
    new_stack = np.empty_like(img_stack)

    for i in numba.prange(img_stack.shape[0]):  # Parallel loop
        # Roll along rows
        new_stack[i] = np.vstack((img_stack[i, -row_shift:], img_stack[i, :-row_shift]))
        # Roll along columns
        new_stack[i] = np.hstack(
            (new_stack[i, :, -col_shift:], new_stack[i, :, :-col_shift])
        )

    return new_stack


def stack_roll(img_stack, row_shift=0, col_shift=0, use_gpu=False):
    if use_gpu:
        return _stack_roll_gpu(img_stack, row_shift, col_shift)
    else:
        return _stack_roll_cpu(img_stack, row_shift, col_shift)


def stack_crop(img_stack, crop, use_gpu=False):
    if use_gpu:
        return _stack_crop_gpu(img_stack, crop)
    else:
        return _stack_crop_cpu(img_stack, crop)


@numba.njit
def _stack_crop_cpu(img_stack, crop):
    """
    Crop the stack efficiently.
    """
    return img_stack[..., crop:-crop, crop:-crop]


def _stack_crop_gpu(img_stack, crop):
    """
    Crop the stack efficiently.
    """
    return img_stack[..., crop:-crop, crop:-crop]


def stack_pad(img_stack, pad):
    """
    Efficient padding with preallocated arrays.
    """
    pad_values = [(pad, pad), (pad, pad)]
    if img_stack.ndim == 3:
        pad_values.insert(0, (0, 0))
    return np.pad(img_stack, pad_values)


def _stack_roll_gpu(img_stack, row_shift=0, col_shift=0):
    """
    Roll a stack of images along both axes using CuPy (CUDA).

    Parameters:
    -----------
    img_stack : cupy.ndarray
        Input stack of images on GPU.
    row_shift : int
        The amount to roll rows.
    col_shift : int
        The amount to roll columns.

    Returns:
    --------
    cupy.ndarray
        The rolled stack of images.
    """
    img_stack = cp.asarray(img_stack)  # Ensure data is on GPU

    # Roll along rows and columns
    rolled_stack = cp.roll(img_stack, shift=(row_shift, col_shift), axis=(-2, -1))

    return rolled_stack


def stack_recenter(img_stack, center, use_gpu=False):
    """
    Recenters the image stack using stack_roll.
    """
    row_center, col_center = center
    row_shift = img_stack.shape[-2] // 2 - row_center
    col_shift = img_stack.shape[-1] // 2 - col_center
    return stack_roll(img_stack, row_shift, col_shift, use_gpu=use_gpu)
