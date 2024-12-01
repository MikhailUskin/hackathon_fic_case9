import numpy as np
from typing import Optional


class KalmanState:
    def __init__(self, id, timestamp=0, x0=None, y0=None, eol_time=30) -> None:
        self.id: float = id
        self.vec_size: float = 8
        self.state_vector: np.ndarray = np.ones((self.vec_size, 1)) / 10
        self.covar_matrix: np.ndarray = np.ones((self.vec_size, self.vec_size)) / 10
        self.timestamp = timestamp
        self.eol_time = eol_time
        self.eol_timer = 0

    def is_state_outdated(self) -> bool:
        return self.eol_timer > self.eol_time


class SimpleKalman:
    def __init__(self, id, x0=None, y0=None, frame_res=(1920, 1080)) -> None:
        self.kalman_states: list[KalmanState] = []
        self.current_timestamp = 0
        self.add_person(id, self.current_timestamp)
        self.current_state = self.get_state_by_id(id)
        self.state_vector = self.current_state.state_vector
        if x0:
            self.state_vector[0] = x0
        if y0:
            self.state_vector[2] = y0
        self.state_vector[1][0] = 0.0
        self.state_vector[3][0] = 0.0
        self.covar_matrix = self.current_state.covar_matrix
        self.transit_matrix = np.eye(self.current_state.vec_size)
        self.observ_matrix = np.zeros((4, self.current_state.vec_size))
        self.form_covar = np.eye(4)
        self.form_matrix = np.zeros((self.current_state.vec_size, 4))
        self.observ_covar_vec = np.array([9 / 3, 9 / 3])  # Vary this [bbox_pos_noise_std, bbox_size_noise_std]
        self.observ_covar = np.eye(4) * self.observ_covar_vec[0] ** 2
        self.form_covar_vec = np.array([9 / 12, 9 / 24])  # And this [bbox_pos_form_noise_std, bbox_size_form_noise_std]
        self.form_covar[0][0] = self.form_covar_vec[0] ** 2
        self.form_covar[1][1] = self.form_covar_vec[0] ** 2
        self.form_covar[2][2] = self.form_covar_vec[1] ** 2
        self.form_covar[3][3] = self.form_covar_vec[1] ** 2
        self.observ_types = {'bbox': self.extrapolate_observ_bbox}
        self.frame_res = frame_res

    def extrapolate_by_id(self, id, timestamp) -> None:
        self.current_state = self.get_state_by_id(id)
        if not self.current_state:
            self.current_state = self.get_state_by_id(id)
        dT = timestamp - self.current_state.timestamp
        self.current_timestamp = timestamp
        self.current_state.timestamp = self.current_timestamp
        self.transit_matrix[0][1] = dT
        self.transit_matrix[2][3] = dT
        self.transit_matrix[4][5] = dT
        self.transit_matrix[6][7] = dT
        self.state_vector = self.current_state.state_vector
        self.state_vector = np.matmul(self.transit_matrix, self.state_vector)
        self.form_matrix[1][0] = dT
        self.form_matrix[3][1] = dT
        self.form_matrix[5][2] = dT
        self.form_matrix[7][3] = dT
        self.covar_matrix = self.current_state.covar_matrix
        self.covar_matrix = np.matmul(np.matmul(self.transit_matrix, self.covar_matrix),
                                      np.transpose(self.transit_matrix)) + np.matmul(
            np.matmul(self.form_matrix, self.form_covar), np.transpose(self.form_matrix))
        self.current_state.eol_timer += dT
        self.set_current_state(id)

    def extrapolate_all_except_id(self, id=None) -> None:
        for kalman_state in self.kalman_states:
            if kalman_state.id != id:
                self.extrapolate_by_id(kalman_state.id, self.current_timestamp)
                if kalman_state.is_state_outdated():
                    self.remove_person_state(id)

    def ext_and_est(self, id, timestamp, observ_vect, observ_type) -> None:
        self.extrapolate_by_id(id, timestamp)
        self.estimate_by_id(id, observ_vect, observ_type)
        self.extrapolate_all_except_id(id)

    def estimate_by_id(self, id, observ_vect, observ_type) -> None:
        self.observ_types[observ_type]()
        self.gain = np.matmul(np.matmul(self.covar_matrix, np.transpose(self.observ_matrix)),
                              np.linalg.inv(np.matmul(np.matmul(self.observ_matrix, self.covar_matrix),
                                                      np.transpose(self.observ_matrix)) + self.observ_covar))
        self.covar_matrix = np.matmul((np.eye(len(self.state_vector)) - np.matmul(self.gain, self.observ_matrix)),
                                      self.covar_matrix)
        self.state_vector = self.state_vector + np.matmul(self.gain, observ_vect - self.ext_observ)
        self.current_state.state_vector = self.state_vector
        self.current_state.covar_matrix = self.covar_matrix
        self.current_state.eol_timer = 0
        self.correct_bbox_size()
        self.set_current_state(id)

    def extrapolate_observ_bbox(self) -> None:
        self.ext_observ = np.zeros((4, 1))
        self.ext_observ[0] = self.state_vector[0]
        self.ext_observ[1] = self.state_vector[2]
        self.observ_matrix = np.zeros((4, self.current_state.vec_size))
        self.observ_matrix[0][0] = 1
        self.observ_matrix[1][2] = 1
        self.observ_matrix[2][4] = 1
        self.observ_matrix[3][6] = 1

    def add_person(self, id, timestamp, x0=None, y0=None) -> None:
        kalman_state = KalmanState(id, timestamp=timestamp)
        self.kalman_states.append(kalman_state)

    def get_state_by_id(self, id) -> KalmanState:
        for kalman_state in self.kalman_states:
            if kalman_state.id == id:
                return kalman_state
        self.add_person(id, self.current_timestamp)
        self.get_state_by_id(id)

    def remove_person_state(self, id) -> None:
        remove_ind = None
        for state_ind, kalman_state in enumerate(self.kalman_states):
            if kalman_state.id == id:
                remove_ind = state_ind
        #del self.kalman_states[remove_ind]

    def set_current_state(self, id) -> None:
        for state_ind, kalman_state in enumerate(self.kalman_states):
            if kalman_state.id == id:
                self.kalman_states[state_ind] = self.current_state

    def correct_bbox_size(self) -> None:
        if (int(self.current_state.state_vector[0] + self.current_state.state_vector[4]) >= self.frame_res[0]):
            scale_x = self.frame_res[0] / int(self.current_state.state_vector[0] + self.current_state.state_vector[4])
            self.current_state.state_vector[4] = self.frame_res[0] - 1

        if (int(self.current_state.state_vector[2] + self.current_state.state_vector[6]) >= self.frame_res[1]):
            scale_y = self.frame_res[1] / int(self.current_state.state_vector[2] + self.current_state.state_vector[6])
            self.current_state.state_vector[6] = self.frame_res[1] - 1

        if (int(self.current_state.state_vector[0] + self.current_state.state_vector[4]) - self.frame_res[0] <= 30) or (
                int(self.current_state.state_vector[2] + self.current_state.state_vector[6]) - self.frame_res[1] <= 30):
            self.remove_person_state(self.current_state.id)

    def get_bbox_by_id(self, id) -> tuple[int, int, int, int]:
        kalman_state_vector = self.get_state_by_id(id).state_vector
        x_left = int(kalman_state_vector[0])
        x_right = int(kalman_state_vector[0] + kalman_state_vector[4])
        y_bottom = int(kalman_state_vector[2])
        y_top = int(kalman_state_vector[2] + kalman_state_vector[6])
        return (x_left, y_bottom, x_right, y_top)

    def get_all_bboxes(self) -> list[tuple[int, int, int, int]]:
        bboxes = []
        for kalman_state in self.kalman_states:
            bboxes.append(self.get_bbox_by_id(kalman_state.id))
        return bboxes

    def track(self, id, timestamp, bbox) -> None:
        self.ext_and_est(id, timestamp, bbox, 'bbox')

    def update(self, timestamp) -> None:
        self.current_timestamp = timestamp
        self.extrapolate_all_except_id()
