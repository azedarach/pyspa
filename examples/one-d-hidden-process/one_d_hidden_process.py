import argparse

import matplotlib.pyplot as plt
import numpy as np

from pyspa.spa import EuclideanSPAModel

def generate_dataset(length=500, state_prob=0.5, poisson_mean=0.5,
                     lower_bound=0.5, upper_bound=1.5):
    if state_prob > 1:
        raise ValueError("probability of a single state must be in [0, 1]")

    Y = np.random.random(length)
    Y[Y >= state_prob] = 2
    Y[Y < state_prob] = 1

    X = np.zeros(length)
    X[Y < 1.5] = np.random.poisson(poisson_mean, size=(np.sum(Y < 1.5),))
    X[Y > 1.5] = np.random.uniform(lower_bound, upper_bound,
                                   size=(np.sum(Y > 1.5),))

    return (Y, X)

def write_output(hidden_series, visible_series, model, output_file):
    with open(output_file, "w") as ofs:
        affiliations = model.affiliations
        states = model.states
        print(states[0])
        k = model.clusters
        statistics_size = np.size(hidden_series)

        state_names = ["S" + str(i) for i in range(1, k + 1)]
        aff_names = ["Gamma" + str(i) for i in range(1, k + 1)]

        header = ("# Y,X," + ",".join(state_names) + ","
                  + ",".join(aff_names) + "\n")
        ofs.write(header)

        for i in range(statistics_size):
            line = str(hidden_series[i]) + "," + str(visible_series[i])
            line = line + "," + ",".join([str(states[j,0]) for j in range(k)])
            line = line + "," + ",".join([str(g) for g in affiliations[i,:]])
            ofs.write(line + "\n")

def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description=("Fit SPA model to a one dimensional time series"
                     " generated by a two-state hidden process"))
    parser.add_argument("--length", dest="length", type=int, default=500,
                        help="time series length")
    parser.add_argument("--state-probability", dest="state_probability",
                        type=float, default=0.5,
                        help="hidden process state probability")
    parser.add_argument("--poisson-mean", dest="poisson_mean", type=float,
                        default=0.5, help="mean of Poisson distributed state")
    parser.add_argument("--uniform-lower-bound", dest="lower_bound",
                        type=float, default=0.5,
                        help="lower bound of uniform distributed state")
    parser.add_argument("--uniform-upper-bound", dest="upper_bound",
                        type=float, default=1.5,
                        help="upper bound of uniform distributed state")
    parser.add_argument("--n-clusters", dest="n_clusters", type=int,
                        default=2, help="number of clusters")
    parser.add_argument("--output-file", dest="output_file", default="",
                        help="name of output file")
    args = parser.parse_args()
    return args

def main():
    args = parse_cmd_line_args()

    (Y, X) = generate_dataset(args.length, args.state_probability,
                              args.poisson_mean, args.lower_bound,
                              args.upper_bound)

    model = EuclideanSPAModel(X[:,np.newaxis], args.n_clusters)

    gamma0 = np.random.random((args.length, args.n_clusters))
    normalizations = np.sum(gamma0, axis=1)
    gamma0 = np.divide(gamma0, normalizations[:, np.newaxis])

    model.find_optimal_approx(gamma0)

    if args.output_file:
        write_output(Y, X, model, args.output_file)

if __name__ == "__main__":
    main()