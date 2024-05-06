import numpy as np
import scipy.optimize as opt
import pandas as pd


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
    # Boeing 747

    total_operating_cost = np.array([[7244, 9600, 9777, 5757, 25000]])  # /hr

    U = total_operating_cost * 0.8 * timeline  # cost of operating each aircraft /hr
    P = np.array([[89000000, 101000000, 114000000, 106100000, 120000000]])  # cost of purchasing an aircraft
    M = np.array([[20000, 20000, 20000, 20000, 20000]]) * timeline  # cost of maintenance /yr
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
                       103320, 96600, 86240, 75600])
    min_flights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    capacities = np.array([128, 150, 190, 172, 660])
    num_aircraft = np.array([1000, 1000, 1000, 1000, 1000])
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

    print("c", c)
    print("A:", A)
    print("ub: ", ub)

    constraints = opt.LinearConstraint(A, ub=ub)

    x_star = opt.milp(c, constraints=constraints, integrality=1)

    print(x_star)


def sensitivity_analysis(c, x_star, A, b, sensitivity=0.1):
    sens = np.array([np.floor(x_star * (1-sensitivity)),
                     np.ceil(x_star * (1+sensitivity))])
    new_values = np.array([c.transpose() @ sens[0], c.transpose() @ sens[1]])

    return sens, new_values


if __name__ == "__main__":
    main()
