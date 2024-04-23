import numpy as np
import scipy.optimize as opt


def main():
    # original form

    A = np.array([[0.2, 0.1, 0.1],
                  [0.2, 0.2, 0.1],
                  [0.3, 0.25, 0.2],
                  [0.2, 0.25, 0.28],
                  [0.1, 0.2, 0.32],
                  ])

    b = np.array([[400000, 600000, 500000, 1000000, 300000]])

    c = np.array([[155, 135, 120]])

    x_star = opt.linprog(c, A_ub=-A, b_ub=-b)

    # standard form

    A_tilde = np.c_[A, -np.identity(5)]

    b = np.array([[400000, 600000, 500000, 1000000, 300000]])

    c_tilde = np.array([[155, 135, 120, 0, 0, 0, 0, 0]])

    x_star_standard = opt.linprog(c_tilde, A_eq=A_tilde, b_eq=b)

    # dual problem

    lambda_star = opt.linprog(-b, A_ub=A_tilde.T,
                              b_ub=c_tilde, bounds=(0, None))

    print(lambda_star)


if __name__ == "__main__":
    main()
