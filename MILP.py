import numpy as np
import scipy.optimize as opt
import pandas as pd

# use this to ensure x is a col. vector
def flatten(U, T):
    E = U.transpose() @ T
    # print("the big ol matrix is of shape", np.shape(E))
    return np.ndarray.flatten(E)


def main():
    # cost of operating each aircraft in $/hr
    U = np.array([[3300, 4200, 4800, 5100, 9000]])
    # route time in hours
    T = np.array([[1, 1.3, 1.4, 1.5, 1.7, 2.1, 2.3, 3.3, 4.3, 4.9]])

    num_aircraft_types = np.size(U)
    num_destinations = np.size(T)

    x_size = num_aircraft_types * num_destinations

    c = flatten(U, T)
    # print("c is of shape", np.shape(c))
    # first 10 entries of c correspond to the first airplane type,
    # next 10 to the second airplane type, etc.

    demand = np.array([2000, 1300, 500, 1300, 800, 300, 1700, 400, 1700, 900])
    min_flights = np.array([1, 2, 3, 4, 5, 5, 5, 5, 5, 5])
    capacities = np.array([180, 230, 250, 330, 470])
    num_aircraft = np.array([15, 5, 7, 90, 30])
    max_flight_hours = 20*num_aircraft

    A = np.zeros([num_destinations, x_size])
    # ensure that flights going to each airport meet passenger demands
    for i in range(num_destinations):
        for j in range(num_aircraft_types):
            A[i, j * num_destinations + i] = -capacities[j]

    # ensure that we aren't exceeding the number of aircraft in the fleet
    for i in range(num_aircraft_types):
        new_row = np.zeros(x_size)
        new_row[num_destinations*i:num_destinations*i + num_destinations] = 1
        A = np.vstack((A, new_row))

    # ensure min flight requirements
    for i in range(num_destinations):
        new_row = np.zeros(x_size)
        for j in range(num_aircraft_types):
            new_row[i + j * 10] = -1
        A = np.vstack((A, new_row))

    # account for operation time
    for i in range(num_aircraft_types):
        new_row = np.zeros(x_size)
        new_row[num_destinations*i:
                num_destinations*i+num_destinations] = T[0, i]
        A = np.vstack((A, new_row))

    ub = np.concatenate((-demand, num_aircraft,
                         -min_flights, max_flight_hours))

    constraints = opt.LinearConstraint(A, ub=ub)

    x_star = opt.milp(c, constraints=constraints, integrality=1).x


def sensitivity_analysis(c, x_star, A, b, sensitivity=0.1):
    sens = np.array([np.floor(x_star * (1-sensitivity)),
                     np.ceil(x_star * (1+sensitivity))])
    new_values = np.array([c.transpose() @ sens[0], c.transpose() @ sens[1]])

    return sens, new_values


if __name__ == "__main__":
    main()
