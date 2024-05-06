import numpy as np
import scipy.optimize as opt
import pandas as pd
from tabulate import tabulate
from random import randint


# use this to ensure x is a col. vector
def flatten(U, H, C):
    E = U.transpose() @ H
    # print("the big ol matrix is of shape", np.shape(E))
    return np.ndarray.flatten(E) + C.transpose()


def main():
    years = 10
    timeline = years * 365  # number of days we're planning for

    # Airplane Models:
    # Airbus A319-100
    # Airbus A320-200
    # Airbus A321-200
    # Boeing 737-800

    total_operating_cost = np.array([[7244, 9600, 9777, 5757]])  # /hr

    U = total_operating_cost * 0.8 * timeline  # cost of operating each aircraft /hr
    P = np.array([[89000000, 101000000, 114000000, 106000000]])  # cost of purchasing an aircraft
    M = np.array([[20000, 20000, 20000, 20000]]) * years  # cost of maintenance /yr
    T = total_operating_cost * 0.2 * timeline  # cost of take-off
    H = np.array([[2, 1.75, 2.25, 2.5, 2.7, 3.3, 1.25, 1.5, 2.75, 2]])  # route time in hours

    num_aircraft_types = np.size(U)
    num_destinations = np.size(H)
    x_size = num_aircraft_types * num_destinations

    C = np.transpose(P + M + T)
    C = np.repeat(C, repeats=num_destinations, axis=0)

    c = np.squeeze(flatten(U, H, C))

    """demand = np.array([2000, 1300, 1500])
    min_flights = np.array([1, 2, 5])
    capacities = np.array([180, 230])
    num_aircraft = np.array([100, 100])
    max_flight_hours = 20*num_aircraft"""

    demand = np.array([220640, 221480, 193760, 158200, 137480, 124040,
                       103320, 96600, 86240, 75600]) / 365
    min_flights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    capacities = np.array([128, 150, 190, 172])
    num_aircraft = np.array([1000, 1000, 1000, 1000])
    max_flight_hours = 20*num_aircraft

    A = np.zeros([num_destinations, x_size])
    # ensure that flights going to each airport meet passenger demands
    for i in range(num_destinations):
        for j in range(num_aircraft_types):
            A[i, j * num_destinations + i] = -capacities[j]

    # ensure that we aren't exceeding the number of aircraft in the fleet
    for i in range(num_aircraft_types):
        new_row = np.zeros(x_size)
        new_row[num_destinations*i:num_destinations*i + num_destinations] = -1
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
                num_destinations*i+num_destinations] = -T[0, i]
        A = np.vstack((A, new_row))

    ub = np.concatenate((-demand, -min_flights,
                         max_flight_hours, num_aircraft))

    constraints = opt.LinearConstraint(A, ub=ub)

    x_star = opt.milp(c, constraints=constraints, integrality=1).x
    nu_star = opt.milp(c, constraints=constraints, integrality=1).fun

    sensitivity_analysis(c, x_star, A, ub, nu_star, num_aircraft_types, num_destinations)

    x_star = np.reshape(x_star, (num_aircraft_types, num_destinations))

    # print((x_star))


def sensitivity_analysis(c, x_star, A, b, nu_star,
                         num_aircraft_types, num_destinations):
    up = x_star.astype(int)
    down = x_star.astype(int)

    up[np.nonzero(up)] = up[np.nonzero(up)] + 1
    down[np.nonzero(down)] = down[np.nonzero(down)] - 1

    new_values = np.array([c.transpose() @ up, c.transpose() @ down])

    stoch_analysis = np.zeros(4)
    for i in range(10000):
        x_temp = x_star
        for j in range(np.size(x_star)): 
            if x_temp[j] != 0:
                x_temp[j] += randint(-1, 1)
        if c.transpose() @ x_temp <= nu_star and np.all(np.less_equal(A @ x_temp, b)):
            stoch_analysis[0] += 1
        elif c.transpose() @ x_temp > nu_star and np.all(np.less_equal(A @ x_temp, b)):
            stoch_analysis[1] += 1
        elif c.transpose() @ x_temp <= nu_star and np.any(np.greater(A @ x_temp, b)):
            stoch_analysis[2] += 1
        elif c.transpose() @ x_temp > nu_star and np.any(np.greater(A @ x_temp, b)):
            stoch_analysis[3] += 1

    # print(stoch_analysis)
    A_temp = A
    b_temp = b
    for i in range(num_destinations):
        b_temp[i] = b_temp[i] * 1.1

    constraints = opt.LinearConstraint(A_temp, ub=b_temp)
    x_star_temp = opt.milp(c, constraints=constraints, integrality=1).x
    nu_star_temp = opt.milp(c, constraints=constraints, integrality=1).fun

    x_star_temp = np.reshape(x_star_temp, (num_aircraft_types,
                                           num_destinations)).astype(int)

    A_temp = A
    b_temp = b
    for i in range(num_destinations):
        b_temp[i] = b_temp[i] * 0.9

    constraints = opt.LinearConstraint(A_temp, ub=b_temp)
    x_star_temp = opt.milp(c, constraints=constraints, integrality=1).x
    nu_star_temp = opt.milp(c, constraints=constraints, integrality=1).fun

    x_star_temp = np.reshape(x_star_temp, (num_aircraft_types,
                                           num_destinations)).astype(int)

    print(x_star_temp)
    print(nu_star_temp)

    # dual 
    constraints = opt.LinearConstraint(-A.T, lb=c)
    lambda_star = opt.milp(b, constraints=constraints, bounds=(0, None))

    print(lambda_star)

    return new_values


if __name__ == "__main__":
    main()
