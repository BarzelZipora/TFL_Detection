import numpy as np
from matplotlib import pyplot as plt

# from ModelFrame import SFM
# from ModelFrame.Model_frame import Frame_Model
from All_Parts.Distance import SFM
from All_Parts.Distance.Model_frame import Frame_Model


class Distance_TFL:

    def run(self, path, prev_frame, tfls,tfl_points, currentEm,focal,pp):

        prev_container = Frame_Model(prev_frame)
        curr_container = Frame_Model(path)
        prev_container.traffic_light = np.array(tfl_points)
        curr_container.traffic_light = np.array(tfls)
        curr_container.EM = currentEm
        curr_container.corresponding_ind = SFM.calc_TFL_dist(prev_container, curr_container, focal, pp)
        return curr_container

