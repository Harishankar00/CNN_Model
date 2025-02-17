import numpy as np
from PIL import Image
mask = np.array(Image.open("/path/to/sample_mask.png"), dtype=np.int64)
print("Unique values in mask:", np.unique(mask))
