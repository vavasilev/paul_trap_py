import math
from typing import List

import numpy as np


class EvolutionCalculator:
    def __init__(self, ions_num: int):
        self._ions_num = ions_num

    def create_v_matrix(self, red_couplings: np.array, blue_couplings: np.array) -> np.array:
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

    def _get_coupling(self, couplings: np.array, i: int, j: int):
        diff: int = j - i
        # Check if the difference is power of 2
        if diff <= 0 or (diff & (diff - 1) != 0):
            return 0
        coupling_idx: int = diff.bit_length() - 1
        denom: int = 2 ** (coupling_idx + 1)
        if (i % denom) < (denom / 2):
            return couplings[coupling_idx]
        return 0
