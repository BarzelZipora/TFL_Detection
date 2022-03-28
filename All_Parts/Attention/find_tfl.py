import math

# from ModelFrame import Attention
# from ModelFrame.Attention import test_find_tfl_lights
# from attention import build_kernel
from All_Parts.Attention.attention import build_kernel, test_find_tfl_lights

try:
    import test
    import os
    import json
    import glob
    import argparse
    import cv2

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from scipy import misc
    from PIL import Image
    import matplotlib.pyplot as plt
    from scipy.spatial import distance

except ImportError:
    print("Need to fix the installation")
    raise


class Find_TFL:
    def __init__(self):
        self.kernel = build_kernel()

    def __find_tfl_test(self, image_path, kernel, fig_num=None):
        red_x, red_y, green_x, green_y, image = test_find_tfl_lights(image_path, kernel, )
        red_tuple = [(red_x[i], red_y[i]) for i in range(len(red_x))]
        green_tuple = [(green_x[i], green_y[i]) for i in range(len(green_x))]
        for red_point in red_tuple:
            green_tuple = list(filter(lambda x: self.get_distance(red_point, x) > 15, green_tuple))
        return red_tuple, green_tuple

    def get_distance(self, red_point, green_point):
        return math.dist([red_point[0], red_point[1]], [green_point[0], green_point[1]])

    def run(self, path_image):
        kernel = self.kernel
        red_tuple, green_tuple = self.__find_tfl_test(path_image, kernel)
        return red_tuple, green_tuple
