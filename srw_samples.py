import os

import numpy as np
from PIL import Image

# Parameters:
images_dir = 'samples'
tiff_name = 'R5.tif'
tiff_path = os.path.join(images_dir, tiff_name)
# tiff_path = 'C:/bin/mrakitin/tiff_reader/data/xf21id1_cam01_H5_V5_029.tif'
bottom_limit = 836

# Read the image:
im = Image.open(tiff_path)
# im.show()

# Get bits per point:
mode_to_bpp = {'1': 1, 'L': 8, 'P': 8, 'I;16': 16, 'RGB': 24, 'RGBA': 32, 'CMYK': 32, 'YCbCr': 24, 'I': 32, 'F': 32}
bpp = mode_to_bpp[im.mode]
limit_value = 2 ** bpp - 1

# Convert it to NumPy array:
imarray = np.array(im, )
max_value = imarray.max()
truncated_imarray = np.copy(imarray[:bottom_limit, :])

# Remove the bottom black area:
a = Image.fromarray(truncated_imarray)  # first black row in bottom
a.show()

print('')
