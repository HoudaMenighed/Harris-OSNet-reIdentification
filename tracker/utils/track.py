import cv2
import numpy as np
from tracker.utils.kalman_filter import KalmanFilter


class TrackState:

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:

    def __init__(self, detection, track_id, class_id, conf, n_init, max_age, ema_alpha,
                 feature=None):
        self.track_id = track_id
        self.class_id = int(class_id)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            feature /= np.linalg.norm(feature)
            self.features.append(feature)

        self.conf = conf
        self._n_init = n_init
        self._max_age = max_age

        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(detection)

    def to_tlwh(self):

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2

        #print('ret : ', ret)
        return ret

    def to_tlbr(self):

        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret


    def ECC(self, src, dst, warp_mode = cv2.MOTION_EUCLIDEAN, eps = 1e-5,
        max_iter = 100, scale = 0.1, align = False):

        # skip if current and previous frame are not initialized (1st inference)
        if (src.any() or dst.any() is None):
            return None, None
        # skip if current and previous fames are not the same size
        elif (src.shape != dst.shape):
            return None, None

        # BGR2GRAY
        if src.ndim == 3:
            # Convert images to grayscale
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        # make the imgs smaller to speed up
        if scale is not None:
            if isinstance(scale, float) or isinstance(scale, int):
                if scale != 1:
                    src_r = cv2.resize(src, (0, 0), fx = scale, fy = scale,interpolation =  cv2.INTER_LINEAR)
                    dst_r = cv2.resize(dst, (0, 0), fx = scale, fy = scale,interpolation =  cv2.INTER_LINEAR)
                    scale = [scale, scale]
                else:
                    src_r, dst_r = src, dst
                    scale = None
            else:
                if scale[0] != src.shape[1] and scale[1] != src.shape[0]:
                    src_r = cv2.resize(src, (scale[0], scale[1]), interpolation = cv2.INTER_LINEAR)
                    dst_r = cv2.resize(dst, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                    scale = [scale[0] / src.shape[1], scale[1] / src.shape[0]]
                else:
                    src_r, dst_r = src, dst
                    scale = None
        else:
            src_r, dst_r = src, dst

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else :
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        try:
            (cc, warp_matrix) = cv2.findTransformECC (src_r, dst_r, warp_matrix, warp_mode, criteria, None, 1)
        except cv2.error as e:
            return None, None
        

        if scale is not None:
            warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
            warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

        if align:
            sz = src.shape
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                src_aligned = cv2.warpPerspective(src, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR)
            else :
                # Use warpAffine for Translation, Euclidean and Affine
                src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR)
            return warp_matrix, src_aligned
        else:
            return warp_matrix, None


    def get_matrix(self, matrix):
        eye = np.eye(3)
        dist = np.linalg.norm(eye - matrix)
        if dist < 100:
            return matrix
        else:
            return eye

    def camera_update(self, previous_frame, next_frame):
        warp_matrix, src_aligned = self.ECC(previous_frame, next_frame)
        if warp_matrix is None and src_aligned is None:
            return
        [a,b] = warp_matrix
        warp_matrix=np.array([a,b,[0,0,1]])
        warp_matrix = warp_matrix.tolist()
        matrix = self.get_matrix(warp_matrix)

        x1, y1, x2, y2 = self.to_tlbr()
        x1_, y1_, _ = matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = matrix @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.mean[:4] = [cx, cy, w / h, h]


    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf):

        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection, class_id, conf):

        self.conf = conf
        self.class_id = class_id.int()
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, detection.to_xyah(), detection.confidence)

        feature = detection.feature / np.linalg.norm(detection.feature)

        smooth_feat = self.ema_alpha * self.features[-1] + (1 - self.ema_alpha) * feature
        smooth_feat /= np.linalg.norm(smooth_feat)
        self.features = [smooth_feat]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):

        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):

        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted
