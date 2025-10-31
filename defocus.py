"""
Amit Kohli
10-24-2025
This file contains functions for computing defocus PSFs
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import pathlib

# dirname = str(pathlib.Path(__file__).parent.parent.absolute())
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 500


class Microscope:
    def __init__(
        self,
        dim,
        NA,
        mag,
        wavelength,
        pixel_size,
        device=torch.device("cpu"),
    ):
        self.dim = dim
        self.NA = NA
        self.mag = mag
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.sample_pitch = pixel_size / mag
        self.dim = dim
        self.k = (2 * np.pi) / wavelength  # wavenumber
        self.device = device

    def get_pupil_coords(self):

        fourier_coords_x = torch.linspace(
            -1 / (2 * self.sample_pitch),
            1 / (2 * self.sample_pitch),
            self.dim,
            device=self.device,
        )
        fourier_coords_y = torch.linspace(
            1 / (2 * self.sample_pitch),
            -1 / (2 * self.sample_pitch),
            self.dim,
            device=self.device,
        )
        pupil_coords_x = (fourier_coords_x * self.wavelength) / self.NA
        pupil_coords_y = (fourier_coords_y * self.wavelength) / self.NA

        [Px, Py] = torch.meshgrid(pupil_coords_x, pupil_coords_y)

        return Px, Py

    def get_pupil_aperture(self):
        """
        Returns the pupil aperture as a binary mask
        """
        Px, Py = self.get_pupil_coords()
        aperture = torch.sqrt(torch.square(Px) + torch.square(Py)) <= 1

        return aperture

    def get_diff_limit(self):
        Px, Py = self.get_pupil_coords()
        diff_limit = torch.sqrt(torch.square(Px) + torch.square(Py)) < 2

        return diff_limit

    def get_pupil(
        self,
        defocus_length=0.0,
    ):
        # returns pupil coordinates and pupil aperture in
        Px, Py = self.get_pupil_coords()

        # make the pupil function
        aperture = self.get_pupil_aperture()

        # phase due to defocus
        W_defocus = 0.5 * defocus_length * (self.NA**2) * (Px**2 + Py**2) * aperture

        P = aperture * torch.exp(1j * self.k * W_defocus)

        return P

    def get_psf(self, defocus_length):
        """
        Returns the PSF of the system at a given defocus length and seidel coefficients
        """
        # compute the PSF using the angular spectrum method
        pupil = self.get_pupil(defocus_length=defocus_length)

        psf = (
            torch.abs(torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pupil))))
            ** 2
        )

        # normalize the PSF
        psf /= torch.sum(psf)

        return psf
