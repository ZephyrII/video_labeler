# import dlib
import cv2


class Tracker():
    def __init__(self):
        # self.t=dlib.correlation_tracker()
        self.cvtracker = cv2.TrackerKCF_create()

    def start(self, im, p0, p1):
        # p0 is leftupper position of the obj, p1 is rightbottom of the obj
        # self.t.start_track(im, dlib.rectangle(p0[0],p0[1], p1[0], p1[1]))
        self.cvtracker.init(im, (p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]))

    def track(self, im):
        # self.t.update(im)
        # print self.t.get_position()
        # position=self.t.get_position()
        ok, position = self.cvtracker.update(im)
        return (int(position[0]), int(position[1])), (int(position[0]+position[2]), int(position[1]+position[3]))

