import ast
import pickle

import numpy as np

# from Tfl_Manager import SFM
# from Tfl_Manager.TFL_manager import TFl_manager
# from View.view import View
from Tfl_Manager.TFL_manager import TFl_manager
from View.view import View


class Controller:

    def __init__(self, pls_path):
        self.pls_path = pls_path
        self.pkl, self.index, self.frame_list = self.get_paths()
        self.data = self.load_data()
        self.focal = self.data['flx']
        self.pp = self.data['principle_point']
        self.tfl_man = TFl_manager(self.focal, self.pp)
        self.run()

    def load_data(self):
        with open(self.pkl, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        return data

    def get_paths(self):
        pkl = ''
        frame_list = []
        with open(self.pls_path, "r") as pls_file:
            paths_list = pls_file.readlines()
            for path in paths_list:
                path = path.strip('\n')
                if path.endswith('pkl'):
                    pkl = 'Controller/' + path
                elif path.endswith('png'):
                    frame_list.append('Controller/' + path)
                else:
                    index = int(path)
        return pkl, index, frame_list

    def calculate_EM(self, prev_frame_id, curr_frame_id):
        EM = np.eye(4)
        if prev_frame_id < 0:
            return EM
        for i in range(prev_frame_id, curr_frame_id):
            EM = np.dot(self.data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
        return EM

    def run(self):
        self.tfl_man.run_all(self.frame_list[0], np.eye(4))

        for frame in self.frame_list[1:]:
            prev_frame_id = int(frame.strip('_leftImg8bit.png')[-2:])
            current_em = self.calculate_EM(prev_frame_id - 1, prev_frame_id)

            path, red_candidates, green_candidates, red_TFLs, green_TFLs, current_frame, rot_pts, foe = self.tfl_man.run_all(
                frame, current_em)

            view = View()
            view.view_plot(path, red_candidates, green_candidates, red_TFLs, green_TFLs, current_frame, rot_pts, foe)
