import numpy as np


class Detection(object):

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature.cpu(), dtype=np.float32)


    def to_xyah(self):

        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
