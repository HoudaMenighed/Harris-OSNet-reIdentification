import numpy as np
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # c'est la matrice H des observations
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.

        # Ils contrôlent la quantité d'incertitude dans la position et la vitesse respectivement.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160


    # créer une nouvelle piste à partir d'une détection non associée
    # retourner la moyenne et la covariance des nouvelles pistes
    def initiate(self, measurement):

        mean_pos = measurement # les positions de la bounding box
        mean_vel = np.zeros_like(mean_pos) # la vitesse ( velocity)
        # le mean dans ce cas est l'etat du systeme xk
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[0],   # the center point x
            2 * self._std_weight_position * measurement[1],   # the center point y
            1 * measurement[2],                               # the ratio of width/height
            2 * self._std_weight_position * measurement[3],   # the height
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * measurement[2],
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance


    #utilise la matrice de transition _motion_mat pour prédire l'état futur à partir de l'état actuel
    #Elle met à jour la covariance en tenant compte de l'incertitude du modèle de mouvement
    def predict(self, mean, covariance):

        std_pos = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
            1 * mean[2],
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[0],
            self._std_weight_velocity * mean[1],
            0.1 * mean[2],
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov #la predection de la cobariance P_k^- = F_k P_{k-1} F_k^T + Q_k

        return mean, covariance

    def project(self, mean, covariance, confidence=.0):

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]


        std = [(1 - confidence) * x for x in std]

        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        #pour le calcule de gain de kalman
        # cette covariance est pour calculer seulement (H_k P_k^- H_k^T + R_k)^{-1} de K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^{-1}
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, confidence=.0):

        projected_mean, projected_cov = self.project(mean, covariance, confidence)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)

        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T,check_finite=False).T

        # measurement dans ce cas est le zk
        innovation = measurement - projected_mean

        # l mean dans ce cas est le xk
        # state update
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        #mise a jour de la covariance de l'erreur d'estimation P_k^+ = (I - K_k H_k) P_k^-
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):

        mean, covariance = self.project(mean, covariance)

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]


        # mahalanobis distance
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
