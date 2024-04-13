import numpy as np
import scipy.optimize as opt
from tabulate import tabulate


# use this to ensure x is a col. vector
def flatten(U, T):
    E = U.transpose() @ T
    # print("the big ol matrix is of shape", np.shape(E))
    return np.ndarray.flatten(E)


def main():
    U = np.array([[1, 2, 3, 4, 5]])  # cost of operating each aircraft in $/hr
    T = np.array([[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])  # route time in hours

    num_aircraft_types = np.size(U)
    num_destinations = np.size(T)

    x_size = num_aircraft_types * num_destinations

    c = flatten(U, T)
    # print("c is of shape", np.shape(c))
    # first 10 entries of c correspond to the first airplane type,
    # next 10 to the second airplane type, etc.

    demand = np.array([700, 300, 200, 700, 100, 900, 1000, 300, 1300, 1000])
    capacities = np.array([100, 200, 300, 400, 500])
    num_aircraft = np.array([1000, 1000, 1000, 1000, 1000])

    A = np.zeros([num_destinations, x_size])
    # ensure that flights going to each airport meet passenger demands

    for i in range(num_destinations):
        for j in range(num_aircraft_types):
            A[i, j * num_destinations + i] = capacities[j]

    # ensure that we aren't exceeding the number of aircraft in the fleet
    for i in range(num_aircraft_types):
        new_row = np.zeros(x_size)
        new_row[num_destinations*i:num_destinations*i + num_destinations] = 1
        A = np.vstack((A, new_row))

    lb = np.pad(demand, (0, np.shape(A)[0] - np.size(demand)))
    ub = np.pad(num_aircraft,
                (np.shape(A)[0] - np.size(num_aircraft), 0),
                'constant', constant_values=1000000)

    # some airports have restrictions on the larger aircraft
    # account for operation time

    # ensure that constraint vectors are correct dimension
    constraints = opt.LinearConstraint(A, lb=lb, ub=ub)

    x_star = opt.milp(c, constraints=constraints, integrality=1).x

    x_star = np.reshape(x_star, (num_aircraft_types, num_destinations))

    print(tabulate(x_star))


if __name__ == "__main__":
    main()
