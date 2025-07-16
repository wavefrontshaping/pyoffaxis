# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:42:40 2021

@author: Manip
"""
from hashlib import new
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_yen
import scipy.ndimage.filters as filters
import scipy
from scipy import ndimage as ndi
from scipy.ndimage.measurements import center_of_mass, label
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

from zoomfft2d import ZoomFFT2D
import skimage.morphology as skm
from scipy.ndimage import distance_transform_edt
import os
import sys

from .stack_functions import (
    stack_fft2,
    stack_ifft2,
    stack_recenter,
    stack_pad,
    stack_crop,
)

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def get_axis_mask(res, center, width):
    X, Y = np.meshgrid(np.arange(res[0]), np.arange(res[1]))
    X, Y = X - center[0], Y - center[1]
    mask = (np.abs(X) > width / 2) * (np.abs(Y) > width / 2)
    return mask


def trim_image(image, threshold):
    image_abs = np.abs(image)
    image_abs = image_abs / np.max(image_abs)
    # Create a mask for rows and columns where any value is above the threshold
    row_mask = np.any(image_abs > threshold, axis=1)
    col_mask = np.any(image_abs > threshold, axis=0)

    # Use the masks to select the rows and columns
    trimmed_image = image[row_mask][:, col_mask]

    return trimmed_image


def get_disk_mask(res, radius, center=None):

    if center is None:
        center = [s / 2 - 0.5 for s in res]

    X, Y = np.meshgrid(np.arange(res[1]), np.arange(res[0]))
    return (X - center[1]) ** 2 + (Y - center[0]) ** 2 < radius**2


class Holography:
    """
    Used to handle a sequence of images obtained through off-axis holography.

    I would like to change it so that this class handles any type of holography
    and then build separate classes defining the specific types e.g. off axis
    quadriwave, etc. This would use the composition over inheritance principle.
    """

    def __init__(
        self,
        # holo_stack,
        dim,
        padding=0,
        reference=None,
        sigma_noise=None,
        display=False,
        use_gpu=False,
    ):
        # Just grab the dimension of the images you want to process
        self.dim = dim
        self.padding = padding
        self.fourier_mask = {}
        self.sigma_noise = sigma_noise
        self.display = display
        self.use_gpu = use_gpu

        assert np.min(reference) >= 0
        self.reference = reference
        self.is_zoom_fft_initialized = False
        self.zft = None
        self.izft = None
        self.zft_data = {}
        self.bias_ref = None

    def removeReferenceAmplitude(self, img_stack, do_filter_ref):

        # Thikhonov regularization
        sigma = self.sigma_noise if self.sigma_noise else 0
        # sigma = 0 means no regularization, prone to errors!

        if do_filter_ref:
            ref = self.ref_filt
        else:
            ref = self.reference

        # inverse_ref = np.sqrt(ref) / (sigma + ref)
        inverse_ref = 1.0 / (sigma + np.sqrt(ref))

        if self.use_gpu:
            inverse_ref = cp.asarray(inverse_ref)

        img_stack *= inverse_ref

        if self.use_gpu:
            img_stack = img_stack.get()

        # if no regularization, there can be wrong values, replaced by 0
        img_stack = np.nan_to_num(img_stack, nan=0.0, posinf=0.0, neginf=0.0)

        return img_stack

    def setMaskParameters(self, mask_parameters):
        if type(mask_parameters) is dict:
            self.fourier_mask = mask_parameters
        else:
            raise TypeError("mask_paramters must be a dict.")

    def calibrate(
        self,
        img_stack,
        relative_distance=1.2,
        threshold_coeff=0.5,
        peak_min_distance=20,
        sigma_filter=3,
        radius_mask_coeff=0.1,
        neighborhood_size_filter=10,
        mask_ratio=1.0,
        axis_mask_width=None,
    ):
        """
        Used to obtain the indices of the -1 order.
        Use it before computing using getFieldImage
        inputs: holo_stack -> the stack of off-axis images obtained experimentally

        outputs: x_first, y_first -> the indices of the center of mass of the order -1.
        """

        if self.padding:
            img_stack = stack_pad(img_stack, self.padding)

        fft_stack = stack_fft2(img_stack.astype(complex))

        plt.figure()
        plt.imshow(np.log(np.abs(fft_stack[0])))

        new_res = fft_stack.shape[-2:]  # [s + 2*self.padding for s in self.dim]
        center = (self.dim[0] / 2 + self.padding, self.dim[1] / 2 + self.padding)

        # mask to remove the spatial frequencies close to DC
        radius_mask = np.min(new_res) * radius_mask_coeff
        mask_ft = 1 - get_disk_mask(new_res, radius=radius_mask, center=center)

        # mask to remove spatial frequencies close to the axis
        if axis_mask_width:
            axis_mask = get_axis_mask(new_res, center, axis_mask_width).astype(
                mask_ft.dtype
            )
            mask_ft *= axis_mask

        # Compute the mean Fourier transform and block the zeroth order
        mean_FT = np.fft.fftshift(np.sum(np.abs(fft_stack), axis=0))
        mean_FT = mask_ft * mean_FT

        mean_FT *= mean_FT > threshold_coeff * np.max(mean_FT)

        blurred_FT = scipy.ndimage.gaussian_filter(
            mean_FT, sigma=neighborhood_size_filter
        )

        # Begin the image processing pipline
        mask_thresh = np.abs(blurred_FT) > threshold_coeff * threshold_yen(
            np.abs(blurred_FT)
        )
        distance = ndi.distance_transform_edt(mask_thresh)

        if self.display:
            plt.figure()
            plt.subplot(221)
            plt.imshow(mask_ft)
            plt.subplot(222)
            plt.imshow(blurred_FT)
            plt.colorbar()
            plt.subplot(223)
            plt.imshow(mask_thresh)
            plt.subplot(224)
            plt.imshow(distance)

        local_maxi = peak_local_max(
            distance, indices=False, min_distance=peak_min_distance
        )

        labels = label(local_maxi)[0]
        merged_peaks = center_of_mass(local_maxi, labels, range(1, np.max(labels) + 1))
        merged_peaks = np.array(merged_peaks, dtype=int)

        # Order the peaks according to their y coordinate
        row_peaks = merged_peaks[np.argsort(merged_peaks[:, 1]), 0]
        col_peaks = merged_peaks[np.argsort(merged_peaks[:, 1]), 1]

        # Get the position of the peak
        pos_first = np.where(col_peaks > center[0] * relative_distance)[0][0]

        if len(col_peaks[pos_first:]) > 1:
            coord_tuples = list(zip(row_peaks[pos_first:], col_peaks[pos_first:]))
            inter_pos_first = np.argmax([distance[co] for co in coord_tuples])
            pos_first = pos_first + inter_pos_first

        self.fourier_mask["center"] = [row_peaks[pos_first], col_peaks[pos_first]]
        self.fourier_mask["size"] = distance[
            self.fourier_mask["center"][0], self.fourier_mask["center"][1]
        ]

        self.fourier_mask["size"] *= mask_ratio

        ## Generate mask to select the desired order in the FFT
        self.fourier_mask["mask"] = np.fft.ifftshift(
            get_disk_mask(
                new_res,
                radius=self.fourier_mask["size"],
                center=self.fourier_mask["center"],
            )
        ).astype(float)
        # a bit of blurring for apodization
        self.fourier_mask["mask"] = scipy.ndimage.gaussian_filter(
            self.fourier_mask["mask"], sigma=sigma_filter
        )

        self.fourier_mask["mask_ref"] = get_disk_mask(
            new_res, radius=self.fourier_mask["size"], center=center
        ).astype(float)
        self.fourier_mask["mask_ref"] = scipy.ndimage.gaussian_filter(
            self.fourier_mask["mask_ref"], sigma=sigma_filter
        )

        trimmed_mask = trim_image(self.fourier_mask["mask"], 1e-3)
        self.fourier_mask["trimmed_mask"] = trimmed_mask

        freq_center_x = (
            (self.fourier_mask["center"][0] - new_res[0] / 2) * 2 / new_res[0]
        )
        freq_center_y = (
            (self.fourier_mask["center"][1] - new_res[1] / 2) * 2 / new_res[1]
        )
        self.fourier_mask["freq_center"] = [freq_center_x, freq_center_y]

        delta_freq_x = 2 * trimmed_mask.shape[0] * 2 / new_res[0]
        delta_freq_y = 2 * trimmed_mask.shape[1] * 2 / new_res[1]
        self.fourier_mask["delta_freqs"] = [delta_freq_x, delta_freq_y]

        # Plot the results so that the user can verify the right peak was
        # selected
        if self.display:
            plt.figure(),
            plt.subplot(121)
            plt.imshow(np.abs(mean_FT))
            plt.plot(col_peaks, row_peaks, "ro")
            plt.plot(center[1], center[0], "bo")
            circle = plt.Circle(
                (self.fourier_mask["center"][1], self.fourier_mask["center"][0]),
                self.fourier_mask["size"],
                color="r",
                fill=False,
            )
            ax = plt.gca()
            ax.add_artist(circle)
            plt.subplot(122)
            plt.imshow(self.fourier_mask["mask"])

    def saveCalibration(self, filename):
        np.savez(filename, calib_data=self.fourier_mask)

    def loadCalibration(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.fourier_mask = data["calib_data"][()]

    def filter_ref(self):
        ref = self.reference
        if self.padding:
            ref = np.pad(ref, self.padding)
        fft_ref = np.fft.fftshift(np.fft.fft2(ref.astype(complex)))
        fft_ref *= self.fourier_mask["mask_ref"].astype(complex)
        self.ref_filt = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_ref)))
        if self.padding:
            self.ref_filt = stack_crop(self.ref_filt[None, ...], self.padding)

    def getFieldStack(self, img_stack, do_filter_ref=True):
        """
        Compute the complex field stack.
        """

        if self.padding:
            img_stack = stack_pad(img_stack, self.padding)

        fft_stack = stack_fft2(img_stack.astype(np.complex128), use_gpu=self.use_gpu)

        mask = (
            cp.asarray(self.fourier_mask["mask"])
            if self.use_gpu
            else self.fourier_mask["mask"]
        )
        # Apply mask
        fft_stack *= mask
        # Recentering
        fft_stack = stack_recenter(
            fft_stack, self.fourier_mask["center"], use_gpu=self.use_gpu
        )

        # Inverse FFT
        complex_stack = stack_ifft2(fft_stack, use_gpu=self.use_gpu)

        if self.padding:
            complex_stack = stack_crop(
                complex_stack, self.padding, use_gpu=self.use_gpu
            )

        if self.reference is not None:
            if do_filter_ref:
                self.filter_ref()
            complex_stack = self.removeReferenceAmplitude(complex_stack, do_filter_ref)

        return complex_stack

    def initializeZoomFFT(self, N_out):
        """
        Initialize the zoom FFT.
        """
        delta_f = self.fourier_mask["delta_freqs"][0]  # mask_diam/N0*coeff

        f_center_x, f_center_y = self.fourier_mask["freq_center"]
        fnx = [f_center_x - delta_f / 2, f_center_x + delta_f / 2]
        fny = [f_center_y - delta_f / 2, f_center_y + delta_f / 2]
        f_center = [f_center_x / 2, f_center_y / 2]
        f_range = [delta_f / 2, delta_f / 2]
        N_fft = self.fourier_mask["trimmed_mask"].shape[0]

        self.zft_data.update(
            {
                "fnx": fnx,
                "fny": fny,
                "f_center": f_center,
                "f_range": f_range,
                "N_fft": N_fft,
                "N_out": N_out,
            }
        )

        N = self.dim[0]
        self.zft = ZoomFFT2D([N] * 2, [N_fft] * 2, f_center, f_range)

        if_range = [N / N_fft * f for f in f_range]

        self.izft = ZoomFFT2D(
            [N_fft] * 2, [N_out] * 2, [0, 0], if_range, direction="backward"
        )
        self.filter_ref_zoom()
        self.is_zoom_fft_initialized = True

    def getFieldStackZoom(
        self,
        img_stack,
        N_out=None,
        compensate_for_ref=True,
        compensate_for_bias=False,
        do_filter_ref=True,
    ):
        """
        Compute the complex field stack using zoom FFT.
        """
        if N_out is None:
            if self.reference is not None:
                N_out = self.reference.shape[0]
            else:
                raise ValueError("N_out must be specified if no reference is provided")

        if not self.is_zoom_fft_initialized:
            self.initializeZoomFFT(N_out)

        fft_stack = self.zft(img_stack)
        complex_stack = self.izft(fft_stack * self.fourier_mask["trimmed_mask"])

        if compensate_for_ref or compensate_for_bias:
            complex_stack = self.correct_image_stack(
                complex_stack,
                compensate_for_ref,
                do_filter_ref,
                compensate_for_bias,
            )

        return complex_stack

    def filter_ref_zoom(self):
        ref = self.reference

        N = self.dim[0]
        N_fft = self.zft_data["N_fft"]
        f_range = self.zft_data["f_range"]

        zft = ZoomFFT2D([N] * 2, [N_fft] * 2, 0.0, f_range)

        fft_ref = zft(ref)
        filtered_ref = self.izft(fft_ref * self.fourier_mask["trimmed_mask"])

        self.ref_filt_zoom = np.abs(filtered_ref)

    def compensate_for_bias(self, img_stack):
        if self.bias_ref is None:
            self.bias_ref = self.getFieldStackZoom(
                self.reference, compensate_for_ref=False, compensate_for_bias=False
            )
        return img_stack - self.bias_ref

    def compensate_for_ref(self, img_stack, do_filter_ref=True):
        if do_filter_ref:
            ref = self.ref_filt_zoom
        else:
            ref = self.reference

        # Thikhonov regularization
        sigma = self.sigma_noise if self.sigma_noise else 0
        inverse_ref = 1.0 / (sigma + np.sqrt(ref))

        if self.use_gpu:
            inverse_ref = cp.asarray(inverse_ref)

        img_stack *= inverse_ref

        if self.use_gpu:
            img_stack = img_stack.get()

        # if no regularization, there can be wrong values, replaced by 0
        img_stack = np.nan_to_num(img_stack, nan=0.0, posinf=0.0, neginf=0.0)

        return img_stack

    def correct_image_stack(
        self,
        img_stack,
        compensate_for_ref=True,
        do_filter_ref=True,
        compensate_for_bias=False,
    ):
        img_stack = img_stack.copy()

        if compensate_for_bias:
            img_stack = self.compensate_for_bias(img_stack)

        if compensate_for_ref:
            img_stack = self.compensate_for_ref(img_stack, do_filter_ref)

        return img_stack
