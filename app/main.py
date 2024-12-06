import numpy as np
import matplotlib.pyplot as plt

from app.evolution_calculator import EvolutionCalculator


def main():
    calc: EvolutionCalculator = EvolutionCalculator(4)
    v_matrix: np.array = calc.create_v_matrix(np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8]))
    display_matrix(v_matrix)

def display_matrix(matrix: np.array):
    rows, columns = matrix.shape

    w = 10
    h = 10
    plt.figure(1, figsize=(w, h))
    tb = plt.table(cellText=matrix, loc=(0, 0), cellLoc='center')

    tc = tb.properties()['celld']
    for cell in tc.values():
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / columns)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

if __name__=="__main__":
    main()