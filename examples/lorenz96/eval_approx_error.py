import argparse

import numpy as np

from pyspa.spa import EuclideanSPAModel

def read_data(input_file, delimiter=",", verbose=False):
    data = np.genfromtxt(input_file, names=True, delimiter=delimiter)
    fields = [f for f in data.dtype.fields]

    if "t" not in fields:
        raise RuntimeError("no time data found in dataset")
    t = data["t"]

    if verbose:
        print("Reading " + str(np.size(t)) + " points")

    n = np.sum(np.array([f.find("xt") for f in fields]) == 0)

    if n == 0:
        raise RuntimeError("no data found for xt variables in dataset")
    elif verbose:
        print("Found number of slow variables n = " + str(n))

    m = int(np.sum(np.array([f.find("yt") for f in fields]) == 0) / n)
    if m == 0:
        raise RuntimeError("no data found for yt variables in dataset")
    elif verbose:
        print("Found number of fast variables m = " + str(m))

    x = np.vstack([data["xt" + str(i)] for i in range(1, n + 1)])
    y = np.zeros((n, m, np.size(t)))
    for i in range(n):
        for j in range(m):
            y[i,j,:] = data["yt" + str(i + 1) + str(j+1)]

    return (t, x, y)

def build_dataset(x, y, row_skip=350, verbose=False):
    T = x.shape[1]
    n = x.shape[0]
    m = y.shape[1]

    y_flat = np.reshape(np.moveaxis(y, -1, 0), (T, n * m))
    data = np.hstack([np.transpose(x), y_flat])

    if verbose:
        print("Skipping initial " + str(row_skip) + " points")

    return data[row_skip:,:]

def get_initial_affiliations(T, K):
    gamma = np.random.random((T, K))
    normalizations = np.sum(gamma, axis=1)
    return np.divide(gamma, normalizations[:, np.newaxis])

def evaluate_approximation_error(model, data, measure="euclidean"):
    validation_model = EuclideanSPAModel(
        data, model.clusters,
        eps_s_sq=model.eps_s_sq,
        normalize=model.normalize,
        stopping_tol=model.stopping_tol,
        max_iterations=model.max_iterations,
        gamma_solver=model.gamma_solver,
        verbose=model.verbose,
        use_exact_states=model.use_exact_states,
        use_trial_step=model.use_trial_step)

    validation_model.states = model.states
    validation_model.solve_subproblem_gamma()

    reconstruction = np.matmul(validation_model.affiliations,
                               validation_model.states)
    errors = data - reconstruction

    if measure == "euclidean":
        return np.mean(np.sqrt(np.sum(np.power(errors, 2), axis=1)))
    else:
        raise ValueError("unrecognized error norm")

def fit_single_spa_model(data, n_clusters, regularization,
                         annealing_steps=5, verbose=False):
    statistics_size = data.shape[0]

    model = EuclideanSPAModel(data, n_clusters, eps_s_sq=regularization,
                              verbose=verbose, use_trial_step=True)

    best_model = None
    best_qf = 9.e99
    for i in range(annealing_steps):
        gamma0 = get_initial_affiliations(statistics_size, n_clusters)
        model.find_optimal_approx(gamma0)
        if best_model is None:
            best_model = model
            best_qf = model.eval_quality_function()
        else:
            qf = model.eval_quality_function()
            if qf < best_qf:
                best_model = model
                best_qf = qf

    return (best_model, best_qf)

def fit_spa_model_with_oos(dataset, clusters, regularizations,
                           annealing_steps, fraction, verbose=False):

    if fraction <= 0. or fraction >= 1.:
        raise ValueError("training fraction must be between 0 and 1")

    statistics_size = dataset.shape[0]
    training_size = 1 + int(fraction * statistics_size)
    validation_size = statistics_size - training_size

    training_data = dataset[:training_size,:]
    validation_data = dataset[training_size:,:]

    results = np.size(clusters) * np.size(regularizations) * [None]
    idx = 0
    for k in clusters:
        for eps in regularizations:
            (model, qf) = fit_single_spa_model(training_data, k, eps,
                                               annealing_steps, verbose)
            approx_err = evaluate_approximation_error(model, validation_data)
            results[idx] = (k, eps, model, qf, approx_err)
            idx += 1

    return results

def fit_spa_model_with_kfold(dataset, clusters, regularizations,
                             annealing_steps, k, verbose=False):

    if k < 2:
        raise ValueError("number of folds must be at least 2")

    statistics_size = dataset.shape[0]
    fold_indices = np.random.permutation(statistics_size)
    folds = k * [None]
    fold_size = int(np.floor(statistics_size / k))
    for i in range(k - 1):
        indices = fold_indices[i * fold_size : (i+1) * fold_size]
        folds[i] = dataset[indices,:]
    folds[k - 1] = dataset[fold_indices[(k - 1) * fold_size:],:]

    results = np.size(clusters) * np.size(regularizations) * [None]
    idx = 0
    for k in clusters:
        for eps in regularizations:
            avg_approx_err = 0.0
            best_model = None
            best_qf = 9.e99
            for i in k:
                validation_data = folds[i]
                training_data = np.concatenate([folds[f] for f in k if f != i])
                (model, qf) = fit_single_spa_model(training_data, k, eps,
                                                   annealing_steps, verbose)
                if best_model is None:
                    best_model = model
                    best_qf = qf
                else:
                    if qf < best_qf:
                        best_model = model
                        best_qf = qf

                approx_err = evaluate_approximation_error(model,
                                                          validation_data)
                avg_approx_err = (approx_err + i * avg_approx_err) / (i + 1)
            results[idx] = (k, eps, best_model, best_qf, avg_approx_err)
            idx += 1

    return results

def fit_spa_model(dataset, clusters=[2], regularizations=[1e-3],
                  annealing_steps=5, cv_method="oos", fraction=0.75,
                  verbose=False):
    statistics_size = dataset.shape[0]
    if verbose:
        print("Using dataset of size " + str(statistics_size))

    if cv_method == "oos":
        return fit_spa_model_with_oos(dataset, clusters, regularizations,
                                      annealing_steps, fraction, verbose)
    elif cv_method == "kfold":
        validation_fraction = 1.0 - fraction
        k = int(np.floor(1.0 / validation_fraction))
        if k < 2:
            if verbose:
                print("Warning: k-fold CV has fewer than 2 subdivisions")
            k = 2
        return fit_spa_model_with_kfold(dataset, clusters, regularizations,
                                        annealing_steps, k, verbose)
    else:
        raise ValueError("unrecognized cross-validation method")

def write_approximation_errors(fit_results, output_file):
    lines = []
    for fit in fit_results:
        k = fit[0]
        eps = fit[1]
        qf = fit[3]
        err = fit[-1]
        line = (str(k) + "," + str(eps) + "," + str(qf) + ","
                + str(err) + "\n")
        lines.append(line)

    with open(output_file, "w") as ofs:
        ofs.writelines(lines)

def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description="Evaluate approximation error on test data set")

    parser.add_argument("input_file", help="input data to read")
    parser.add_argument("--min-clusters", dest="min_clusters", type=int,
                        default=2, help="minimum number of clusters")
    parser.add_argument("--max-clusters", dest="max_clusters", type=int,
                        default=40, help="maximum number of clusters")
    parser.add_argument("--min-log10-eps", dest="min_log10_eps", type=float,
                        default=-14,
                        help="Log10 of minimum regularization parameter")
    parser.add_argument("--max-log10-eps", dest="max_log10_eps", type=float,
                        default=-3,
                        help="Log10 of maximum regularization parameter")
    parser.add_argument("--n-regularizations", dest="n_regularizations",
                        type=int, default=5,
                        help="Number of regularizations to try")
    parser.add_argument("--n-annealing", dest="n_annealing", type=int,
                        default=5, help="number of annealing steps")
    parser.add_argument("--cv-method", dest="cv_method",
                        choices=["oos", "kfold"], default="oos",
                        help="Cross-validation method to use")
    parser.add_argument("--training-fraction", dest="train_fraction",
                        type=float, default=0.75,
                        help="fraction of data to use as training")
    parser.add_argument("-v,--verbose", dest="verbose", action="store_true",
                        help="produce verbose output")

    return parser.parse_args()

def main():
    args = parse_cmd_line_args()

    if args.verbose:
        print("Reading data from '" + args.input_file + "'")

    (t, x, y) = read_data(args.input_file, verbose=args.verbose)

    if args.verbose:
        print("Building initial dataset")
    dataset = build_dataset(x, y)

    regularizations = np.logspace(args.min_log10_eps, args.max_log10_eps,
                                  args.n_regularizations)
    if args.verbose:
        print("Trying regularizations: ", regularizations)

    clusters = np.arange(args.min_clusters, args.max_clusters + 1)
    if args.verbose:
        print("Trying numbers of clusters: ", clusters)

    results = fit_spa_model(dataset, clusters=clusters,
                            regularizations=regularizations,
                            annealing_steps=args.n_annealing,
                            cv_method=args.cv_method,
                            fraction=args.train_fraction,
                            verbose=args.verbose)

    write_approximation_errors(results, "approx_errors.csv")

if __name__ == "__main__":
    main()
