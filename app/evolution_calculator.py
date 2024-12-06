import math
from typing import List

import numpy as np


class EvolutionCalculator:
    def __init__(self, ions_num: int):
        self._ions_num = ions_num

    def create_v_matrix(self, red_couplings: np.ndarray, blue_couplings: np.ndarray) -> np.ndarray:
        if len(red_couplings) != self._ions_num:
            raise f"The size of red couplings array should be {self._ions_num}"
        if len(blue_couplings) != self._ions_num:
            raise f"The size of blue couplings array should be {self._ions_num}"
        rows: List = []
        for i in range(0, 2 ** self._ions_num):
            row: List = []
            for j in range(0, 2 ** self._ions_num):
                row.append(self._get_coupling(red_couplings, i, j) + self._get_coupling(blue_couplings, j, i))
            rows.append(row)

        return np.array(rows)

    def calculate_ms_values_and_vectors(self, v_matrix: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        v_matrix_dag: np.ndarray = np.transpose(np.conj(v_matrix))

        m1: np.ndarray = np.dot(v_matrix, v_matrix_dag)
        m2: np.ndarray = np.dot(v_matrix_dag, v_matrix)

        A_U, A_S, A_Vh = np.linalg.svd(m1)
        B_U, B_S, B_Vh = np.linalg.svd(m2)
        return A_U, A_S, B_U, B_S

    def _get_coupling(self, couplings: np.ndarray, i: int, j: int):
        diff: int = j - i
        # Check if the difference is power of 2
        if diff <= 0 or (diff & (diff - 1) != 0):
            return 0
        coupling_idx: int = diff.bit_length() - 1
        denom: int = 2 ** (coupling_idx + 1)
        if (i % denom) < (denom / 2):
            return couplings[coupling_idx]
        return 0
