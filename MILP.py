import numpy as np
import scipy.optimize as opt
import pandas as pd


# use this to ensure x is a col. vector
def flatten(U, H, C):
    E = U.transpose() @ H
    # print("the big ol matrix is of shape", np.shape(E))
    return np.ndarray.flatten(E) + C.transpose()


def main():
    # number of days we're planning for
    timeline = 10 * 365

    # cost of operating each aircraft in $/hr
    U = np.array([[15000, 15500]])

    # cost of purchasing an aircraft
    P = np.array([[200000000, 350000000]])

    # cost of maintenance
    M = np.array([[500, 1000]]) * timeline

    # cost of take-off
    T = np.array([[5000, 8000]])

    # route time in hours
    H = np.array([[1, 1.3, 3]])

    num_aircraft_types = np.size(U)
    num_destinations = np.size(H)
    x_size = num_aircraft_types * num_destinations

    C = np.transpose(P + M + T)

    print(C)

    C = np.repeat(C, repeats=num_destinations, axis=0)

    print(C)

    c = flatten(U, H, C)

    print(c)

    # print("c is of shape", np.shape(c))
    # first 10 entries of c correspond to the first airplane type,
    # next 10 to the second airplane type, etc.

    demand = np.array([2000, 1300, 1500])
    min_flights = np.array([1, 2, 5])
    capacities = np.array([180, 230])
    num_aircraft = np.array([100, 100])
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
            new_row[i + j * num_aircraft_types] = -1
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

    print(x_star)


def sensitivity_analysis(c, x_star, A, b, sensitivity=0.1):
    sens = np.array([np.floor(x_star * (1-sensitivity)),
                     np.ceil(x_star * (1+sensitivity))])
    new_values = np.array([c.transpose() @ sens[0], c.transpose() @ sens[1]])

    return sens, new_values


if __name__ == "__main__":
    main()
