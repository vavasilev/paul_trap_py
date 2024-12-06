import numpy as np
import matplotlib.pyplot as plt

from app.evolution_calculator import EvolutionCalculator



def main():
    desired_array: np.ndarray = [0.5, 0, 0, 0.5, 0, 0.5, 0.5, 0]
    ion_num: int = 3
    is_complex: bool = False
    accuracy: float = 10e-5
    range: int = 10

    min_distance: float = 100.0
    distance: float = min_distance
    min_array: np.ndarray
    A_S: np.ndarray
    A_U: np.ndarray

    while distance >= accuracy:
        array = np.array((np.rint(range*np.random.rand(2*ion_num)) + 1.j*np.rint(range*np.random.rand(2*ion_num)) if is_complex else np.rint(range*np.random.rand(6))))
        distance, A_S, A_U = f(array, desired_array)
        if distance < min_distance:
            min_distance = distance
            min_array = array

    print(f"min_array = {min_array}")
    print(f"min_distance = {min_distance}")

    display_vector(A_S)
    display_matrix(A_U)

def f(red_blue: np.ndarray, desired: np.ndarray):
    red_blue_split = np.split(red_blue, 2)
    red_couplings: np.ndarray = red_blue_split[0]
    blue_couplings: np.ndarray = red_blue_split[1]
    calc: EvolutionCalculator = EvolutionCalculator(int(len(red_blue)/2))
    v_matrix: np.ndarray = calc.create_v_matrix(red_couplings, blue_couplings)
    A_U, A_S, B_U, B_S = calc.calculate_ms_values_and_vectors(v_matrix)

    reflection_vector = A_U[:, 1]
    distnce = 1-np.absolute(np.dot(desired, reflection_vector)/(np.linalg.norm(desired)*np.linalg.norm(reflection_vector)))
    return distnce, A_S, A_U


def display_matrix(matrix: np.ndarray):
    rows, columns = matrix.shape

    w = 10
    h = 10
    plt.figure(1, figsize=(w, h))
    tb = plt.table(cellText=np.round(matrix, 2), loc=(0, 0), cellLoc='center')

    tc = tb.properties()['celld']
    for cell in tc.values():
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / columns)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

def display_vector(vector: np.ndarray):
    columns = len(vector)

    w = 10
    h = 1
    plt.figure(1, figsize=(w, h))
    tb = plt.table(cellText=[np.round(vector, 2)], loc=(0, 0), cellLoc='center')

    tc = tb.properties()['celld']
    for cell in tc.values():
        cell.set_width(1.0 / columns)

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


if __name__ == "__main__":
    main()
