import numpy as np
from matplotlib import pyplot as plt

# from ModelFrame import SFM
# from ModelFrame.SFM import prepare_3D_data


class View:

    def view_plot(self, path, red_candidates, green_candidates, red_TFLs, green_TFLs, current_frame, rot_pts, foe):
        fig, (distance_sec, tfl_sec, suspicious_sec) = plt.subplots(1, 3, figsize=(12, 6))
        fig.canvas.set_window_title('Traffic Lights Detection')
        plt.suptitle(f"Frame {path}")

        ''''attention'''
        suspicious_sec.set_title('Suspicious candidates')
        suspicious_sec.imshow(plt.imread(path))
        if len(red_candidates) > 0:
            suspicious_sec.plot(red_candidates[:, 0], red_candidates[:, 1], 'rx', color='r', markersize=4)
        if len(green_candidates) > 0:
            suspicious_sec.plot(green_candidates[:, 0], green_candidates[:, 1], 'rx', color='g', markersize=4)

        '''prediction'''
        tfl_sec.set_title('Traffic light candidates')
        tfl_sec.imshow(plt.imread(path))
        if len(red_TFLs) > 0:
            tfl_sec.plot(red_TFLs[:, 0], red_TFLs[:, 1], 'rx', color='r', markersize=4)
        if len(green_TFLs) > 0:
            tfl_sec.plot(green_TFLs[:, 0], green_TFLs[:, 1], 'rx', color='g', markersize=4)

        '''calc dist'''
        distance_sec.set_title('tfl distances')
        if current_frame:
            distance_sec.imshow(current_frame.img)
            curr_p = current_frame.traffic_light
            distance_sec.plot(curr_p[:, 0], curr_p[:, 1], 'b+')
            if foe is not None:
                for i in range(len(curr_p)):
                    distance_sec.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
                    if current_frame.valid[i]:
                        distance_sec.text(curr_p[i, 0], curr_p[i, 1],
                                          r'{0:.1f}'.format(current_frame.traffic_lights_3d_location[i, 2]), color='r')
                distance_sec.plot(foe[0], foe[1], 'r+')
                distance_sec.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')

        plt.show()
