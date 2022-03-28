from matplotlib import pyplot as plt


class Frame_Model:
    def __init__(self,frame_path):
        self.img = plt.imread(frame_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []