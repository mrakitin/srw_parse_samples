# -*- coding: utf-8 -*-
#############################################################################
# SRW Samples library
# Authors/Contributors: Maksim Rakitin
# October 26, 2016
#############################################################################

import math
import os

import numpy as np
import srwlib
from PIL import Image


def SRWLOptSample(image_data, limit_value, nx, ny, resolution, thickness, delta, atten_len, xc=0, yc=0, e_start=0,
                  e_fin=0):
    """Setup Sample element.

    :param image_data: data from the provided TIFF file.
    :param limit_value: maximum possible value (from bits per point).
    :param nx: number of horizontal points.
    :param ny: number of vertical points.
    :param resolution: resolution of the image [m/pixel].
    :param thickness: thickness of the sample [m].
    :param delta: refractive index decrement.
    :param atten_len: attenuation length [m].
    :param xc: horizontal coordinate of center [m].
    :param yc: vertical coordinate of center [m].
    :param e_start: initial photon energy [eV].
    :param e_fin: final photon energy [eV].
    :return: transmission (SRWLOptT) type optical element which simulates the Sample.
    """

    input_parms = {
        "type": "sample",
        "horizontalPoints": nx,
        "verticalPoints": ny,
        "resolution": resolution,
        "thickness": thickness,
        "refractiveIndex": delta,
        "attenuationLength": atten_len,
        "horizontalCenterCoordinate": xc,
        "verticalCenterCoordinate": yc,
        "initialPhotonEnergy": e_start,
        "finalPhotonPnergy": e_fin,
    }

    ne = 1
    fx = 1e+23
    fy = 1e+23
    rx = nx * resolution
    ry = ny * resolution

    opT = srwlib.SRWLOptT(nx, ny, rx, ry, None, 1, fx, fy, xc, yc, ne, e_start, e_fin)

    # Same data alignment as for wavefront: outmost loop vs y, inmost loop vs e
    hx = rx / (nx - 1)
    hy = ry / (ny - 1)
    ofst = 0
    y = -0.5 * ry  # the image is always centered on the grid, however grid can be shifted
    for iy in range(ny):
        x = -0.5 * rx
        for ix in range(nx):
            for ie in range(ne):
                pathInBody = thickness * image_data[ix, iy] / limit_value
                opT.arTr[ofst] = math.exp(-0.5 * pathInBody / atten_len)  # amplitude transmission
                opT.arTr[ofst + 1] = -delta * pathInBody  # optical path difference
                ofst += 2
            x += hx
        y += hy

    opT.input_parms = input_parms

    return opT


def read_image(tiff_path, bottom_limit=None, show_images=False):
    """Read TIFF image.

    :param tiff_path: full path to the image.
    :param bottom_limit: the bottom limit separating the image and the legend (black block).
    :return: dictionary with the read data and the maximum possible value.
    """
    # Read the image:
    orig_image = Image.open(tiff_path)

    # Convert it to NumPy array:
    imarray = np.array(orig_image, )

    # Get bits per point:
    mode_to_bpp = {'1': 1, 'L': 8, 'P': 8, 'I;16': 16, 'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24, 'I': 32, 'F': 32}
    bpp = mode_to_bpp[orig_image.mode]
    limit_value = float(2 ** bpp - 1)

    # Get the bottom limit if it's not provided:
    if not bottom_limit:
        bottom_limit = np.where(imarray[:, 0] == 0)[0][0]

    # Remove the bottom black area:
    truncated_imarray = np.copy(imarray[:bottom_limit, :])
    data = np.transpose(truncated_imarray)

    # Remove background noise:
    idxs_less = np.where(data < limit_value * 0.5)
    data[idxs_less] = np.uint16(0)

    # Generate new image object to track the changes:
    new_image = Image.fromarray(np.transpose(data))

    if show_images:
        orig_image.show()
        new_image.show()

    return {
        'data': data,
        'limit_value': limit_value,
        'bottom_limit': bottom_limit,
        'orig_image': orig_image,
        'new_image': new_image,
    }


if __name__ == '__main__':
    # Parameters:
    images_dir = 'samples'
    resolutions = {  # [nm / pixel]
        'H5.tif': 1.417411,
        'H4.tif': 1.417411,
        'H3.tif': 1.417411,
        'R5.tif': 2.480469,
        'R7.tif': 2.834821,
        'R10.tif': 3.96875,
        'R15.tif': 4.960938,
        'R20.tif': 6.614583,
        'H5R5.tif': 3.96875,
        '5rings.npy': 10,
        '7rings.npy': 10,
        '10rings.npy': 10,
    }
    tiff_name = 'R5.tif'
    # tiff_name = 'R7.tif'
    # tiff_name = 'R10.tif'
    # tiff_name = 'H5R5.tif'
    tiff_path = os.path.join(images_dir, tiff_name)
    # tiff_path = 'C:/bin/mrakitin/tiff_reader/data/xf21id1_cam01_H5_V5_029.tif'
    show_images = True

    d = read_image(tiff_path, show_images=show_images)

    # SRW part:
    nx = d['data'].shape[0]
    ny = d['data'].shape[1]
    resolution = resolutions[tiff_name] * 1e-9  # [m/pixel]
    # thickness = 100e-9
    # thickness = 1e-6
    thickness = 10e-6
    # material = 'Au'
    delta = 3.23075074E-05  # for Au at 9646 eV
    atten_len = 4.06544e-6  # [m] for Au at 9646 eV

    opT = SRWLOptSample(d['data'], d['limit_value'], nx, ny, resolution, thickness, delta, atten_len)

    print('')
