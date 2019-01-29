import argparse
import numpy as np
import pickle
import shelve

from pyspa.spa import EuclideanSPAModel

import lorenz96_utils

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
        return np.sum(np.power(errors, 2)) / np.size(errors)
    else:
        raise ValueError("unrecognized error norm")

def fit_single_spa_model(data, n_clusters, regularization,
                         annealing_steps=5, stopping_tolerance=1.e-2,
                         max_iterations=500, use_trial_step=True,
                         verbose=False):
    statistics_size = data.shape[0]

    model = EuclideanSPAModel(data, n_clusters, eps_s_sq=regularization,
                              verbose=verbose, use_trial_step=use_trial_step,
                              stopping_tol=stopping_tolerance,
                              max_iterations=max_iterations)

    best_model = None
    best_qf = 9.e99
    for i in range(annealing_steps):
        gamma0 = lorenz96_utils.get_random_affiliations(
            statistics_size, n_clusters)
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
                           annealing_steps, fraction, stopping_tolerance=1.e-2,
                           max_iterations=500, use_trial_step=True,
                           verbose=False):

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
            try:
                (model, qf) = fit_single_spa_model(
                    training_data, k, eps,
                    annealing_steps,
                    stopping_tolerance=stopping_tolerance,
                    max_iterations=max_iterations,
                    use_trial_step=use_trial_step, verbose=verbose)
                approx_err = evaluate_approximation_error(model,
                                                          validation_data)
            except ValueError:
                model = None
                qf = np.NaN
                approx_err = np.NaN
            except RuntimeError:
                model = None
                qf = np.NaN
                approx_err = np.NaN

            results[idx] = {"k": k, "eps": eps, "model": model,
                            "qf": qf, "avg_approx_err": approx_err}
            idx += 1

    return results

def fit_spa_model_with_kfold(dataset, clusters, regularizations,
                             annealing_steps, n_folds, stopping_tolerance=1.e-2,
                             max_iterations=500, use_trial_step=True,
                             verbose=False):

    if n_folds < 2:
        raise ValueError("number of folds must be at least 2")

    statistics_size = dataset.shape[0]
    fold_indices = np.random.permutation(statistics_size)
    folds = n_folds * [None]
    fold_size = int(np.floor(statistics_size / n_folds))
    for i in range(n_folds - 1):
        folds[i] = np.sort(fold_indices[i * fold_size : (i+1) * fold_size])
    folds[n_folds - 1] = np.sort(fold_indices[(n_folds - 1) * fold_size:])

    results = np.size(clusters) * np.size(regularizations) * [None]
    idx = 0
    for k in clusters:
        for eps in regularizations:
            avg_approx_err = 0.0
            best_model = None
            best_qf = 9.e99
            fits = 0
            for i in range(n_folds):
                cv_mask = np.ones(len(dataset), dtype=bool)
                cv_mask[folds[i]] = False
                training_data = dataset[cv_mask]
                validation_data = dataset[folds[i],:]
                try:
                    (model, qf) = fit_single_spa_model(
                        training_data, k, eps,
                        annealing_steps, stopping_tolerance=stopping_tolerance,
                        max_iterations=max_iterations,
                        use_trial_step=use_trial_step, verbose=verbose)
                    if best_model is None:
                        best_model = model
                        best_qf = qf
                    else:
                        if qf < best_qf:
                            best_model = model
                            best_qf = qf

                    approx_err = evaluate_approximation_error(model,
                                                              validation_data)
                    avg_approx_err = ((approx_err + fits * avg_approx_err)
                                      / (fits + 1))
                    fits += 1
                except RuntimeError:
                    continue
                except ValueError:
                    continue

            if best_model is None:
                best_qf = np.NaN
                avg_approx_error = np.NaN

            results[idx] = {"k": k, "eps": eps, "model": best_model,
                            "qf": best_qf, "avg_approx_err": avg_approx_err}
            idx += 1

    return results

def fit_spa_model(dataset, clusters=[2], regularizations=[1e-3],
                  annealing_steps=5, cv_method="oos", fraction=0.75,
                  stopping_tolerance=1.e-2,
                  max_iterations=500, use_trial_step=True, verbose=False):
    statistics_size = dataset.shape[0]

    if cv_method == "oos":
        return fit_spa_model_with_oos(dataset, clusters, regularizations,
                                      annealing_steps, fraction,
                                      stopping_tolerance=stopping_tolerance,
                                      max_iterations=max_iterations,
                                      use_trial_step=use_trial_step,
                                      verbose=verbose)
    elif cv_method == "kfold":
        validation_fraction = 1.0 - fraction
        k = int(np.floor(1.0 / validation_fraction))
        if k < 2:
            if verbose:
                print("Warning: k-fold CV has fewer than 2 subdivisions")
            k = 2
        return fit_spa_model_with_kfold(dataset, clusters, regularizations,
                                        annealing_steps, k,
                                        stopping_tolerance=stopping_tolerance,
                                        max_iterations=max_iterations,
                                        use_trial_step=use_trial_step,
                                        verbose=verbose)
    else:
        raise ValueError("unrecognized cross-validation method")

def get_file_header():
    return "# k,eps,L_eps,avg_approx_err"

def write_approximation_errors(fit_results, output_file=""):
    header = get_file_header()
    if output_file:
        header = header + "\n"

    lines = [header]
    fmt_string="{:d},{:<14.8e},{:<14.8e},{:<14.8e}"
    for r in fit_results:
        line = fmt_string.format(
            r["k"], r["eps"], r["qf"], r["avg_approx_err"])
        if output_file:
            line = line + "\n"
        lines.append(line)

    if output_file:
        with open(output_file, "w") as ofs:
            ofs.writelines(lines)
    else:
        for l in lines:
            print(l)

def write_models(fit_results, models_output_file):
    with shelve.open(models_output_file,
                     protocol=pickle.DEFAULT_PROTOCOL) as db:
        for i, r in enumerate(fit_results):
            db[str(i)] = r

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
    parser.add_argument("--discard-fraction", dest="discard_fraction",
                        type=float, default=0.1,
                        help="fraction of initial time series to discard")
    parser.add_argument("--output-file", dest="output_file", default="",
                        help="name of file to output results to")
    parser.add_argument("--models-output-file", dest="models_output_file",
                        default="",
                        help="name of file to output fitted models to")

    return parser.parse_args()

def main():
    args = parse_cmd_line_args()

    regularizations = np.logspace(args.min_log10_eps, args.max_log10_eps,
                                  args.n_regularizations)

    clusters = np.arange(args.min_clusters, args.max_clusters + 1)

    (t, x, y) = lorenz96_utils.read_data(args.input_file)
    data = lorenz96_utils.build_dataset(x, y, args.discard_fraction)

    results = fit_spa_model(data, clusters=clusters,
                            regularizations=regularizations,
                            annealing_steps=args.n_annealing,
                            cv_method=args.cv_method,
                            fraction=args.train_fraction)

    write_approximation_errors(results, args.output_file)

    if args.models_output_file:
        write_models(results, args.models_output_file)

if __name__ == "__main__":
    main()
