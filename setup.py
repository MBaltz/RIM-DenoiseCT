from RIM import RIM

from dival.datasets.standard import get_standard_dataset
from dival import get_reference_reconstructor

import numpy as np
import cv2

model_rim = RIM(10, (362, 362), (362362))

impl = "astra_cpu"
dataset = get_standard_dataset('lodopab', impl=impl)
print(f"Checando LoDoPaB-CT: {dataset.check_for_lodopab()}")
keras_generator = dataset.create_keras_generator("train", 1, shuffle=True)

rec = get_reference_reconstructor("fbp", "lodopab", impl=impl)

# x, y = keras_generator.__getitem__(0)
# print(x.shape, y.shape) # (1, 1000, 513) (1, 362, 362)

for i in range(2):

    x, y = keras_generator.__getitem__(i)
    x, y = x[0], y[0]
    rec_x = rec.reconstruct(x)

    rec_x -= np.min(rec_x)
    rec_x /= np.max(rec_x)
    
    x -= np.min(x)
    x /= np.max(x)

    print(f"Min_rec: {format(np.min(rec_x), '.10f')}")
    print(f"Min_rec: {format(np.max(rec_x), '.10f')}")

    print(f"Min_x: {format(np.min(x), '.10f')}")
    print(f"Max_x: {format(np.max(x), '.10f')}")

    print(f"Min_y: {format(np.min(y), '.10f')}")
    print(f"Max_y: {format(np.max(y), '.10f')}")

    print()

    i_x = cv2.normalize(np.float64(x), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    cv2.imwrite('aux/x_'+str(i)+'.png', i_x)
    
    i_gt = cv2.normalize(np.float64(y), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    # cv2.imwrite('gt.png', i_gt)
    i_rec = cv2.normalize(np.float64(rec_x), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    vis = np.concatenate((i_rec, i_gt), axis=1)
    cv2.imwrite('aux/concat_'+str(i)+'.png', vis)
