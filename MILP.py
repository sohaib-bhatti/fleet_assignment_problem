import numpy as np
import scipy.optimize as opt


# use this to ensure x is a col. vector
def flatten(U, T):
    E = U.transpose() @ T
    # print("the big ol matrix is of shape", np.shape(E))
    print(E)
    return np.ndarray.flatten(E)


def main():
    U = np.array([[1, 2, 3, 4, 5]])
    T = np.array([[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])

    num_aircraft_types = np.size(U)
    num_destinations = np.size(T)

    x_size = num_aircraft_types * num_destinations

    c = flatten(U, T)
    # print("c is of shape", np.shape(c))
    # first 10 entries of c correspond to the first airplane type,
    # next 10 to the second airplane type, etc.

    demand = np.array([700, 300, 200, 700, 100, 900, 1000, 300, 1300, 1000])
    capacities = np.array([100, 200, 300, 400, 500])

    A = np.zeros([num_destinations, x_size])

    # ensure that flights going to each airport meet passenger demands

    for i in range(num_destinations):
        for j in range(num_aircraft_types):
            A[i, j * 10 + i] = capacities[j]

    constraints = opt.LinearConstraint(A, lb=demand)

    """""""""""""""""""""
    THE STAR OF THE SHOW
    """""""""""""""""""""
    x_star = opt.milp(c, constraints=constraints)

    print(x_star)


if __name__ == "__main__":
    main()
