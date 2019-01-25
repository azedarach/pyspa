import argparse
import numpy as np
import time

from pyspa.spa import EuclideanSPAModel

import lorenz96_utils

def time_spa_model(data, clusters=2, regularizations=1.e-3,
                   annealing_steps=5, stopping_tolerance=1.e-2,
                   max_iterations=500, use_trial_step=True):

    if np.isscalar(clusters):
        k_vals = clusters * np.ones(1)
    else:
        k_vals = clusters

    if np.isscalar(regularizations):
        eps_vals = regularizations * np.ones(1)
    else:
        eps_vals = regularizations

    statistics_size = data.shape[0]
    run_times = np.size(k_vals) * np.size(eps_vals) * [None]
    idx = 0
    for k in k_vals:
        for eps in eps_vals:
            model = EuclideanSPAModel(data, k, eps_s_sq=eps,
                                      stopping_tol=stopping_tolerance,
                                      max_iterations=max_iterations,
                                      use_trial_step=use_trial_step)
            timings = []
            for i in range(annealing_steps):
                gamma0 = lorenz96_utils.get_random_affiliations(
                    statistics_size, k)
                try:
                    start_time = time.perf_counter()
                    model.find_optimal_approx(gamma0)
                    end_time = time.perf_counter()
                    timings.append(end_time - start_time)
                except RuntimeError:
                    continue
                except ValueError:
                    continue
            run_times[idx] = {"T": statistics_size,
                              "k": k,
                              "eps": eps,
                              "tol": stopping_tolerance,
                              "max_iters": max_iterations,
                              "times": timings}
            idx += 1

    return run_times

def get_file_header():
    return "# T,k,eps,tol,max_iters,min_time_sec,mean_time_sec,max_time_sec"

def write_timings(timing_results, output_file=""):
    header = get_file_header()
    lines = [header]
    fmt_string = "{:d},{:d},{:<14.8e},{:14.8e},{:d},{:14.8e},{:14.8e},{:<14.8e}"
    for run in timing_results:
        if np.size(run["times"]) > 0:
            min_time = np.min(run["times"])
            mean_time = np.mean(run["times"])
            max_time = np.mean(run["times"])
            line = fmt_string.format(
                run["T"], run["k"], run["eps"], run["tol"], run["max_iters"],
                min_time, mean_time, max_time)
            if output_file:
                line = line + "\n"
            lines.append(line)

    if output_file:
        with open(output_file, "w") as ofs:
            ofs.writelines(lines)
    else:
        for l in lines:
            print(l)

def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description="Measure time taken to train on input data-sets")

    parser.add_argument("input_file", nargs="+",
                        help="data files to time")
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
    parser.add_argument("--discard-fraction", dest="discard_fraction",
                        type=float, default=0.1,
                        help="fraction of initial time series to discard")
    parser.add_argument("--output-file", dest="output_file", default="",
                        help="name of file to output results to")

    return parser.parse_args()

def main():
    args = parse_cmd_line_args()

    regularizations = np.logspace(args.min_log10_eps, args.max_log10_eps,
                                  args.n_regularizations)

    clusters = np.arange(args.min_clusters, args.max_clusters + 1)

    timing_results = []
    for f in args.input_file:
        (t, x, y) = lorenz96_utils.read_data(f)
        data = lorenz96_utils.build_dataset(x, y, args.discard_fraction)
        timings = time_spa_model(data, clusters=clusters,
                                 regularizations=regularizations,
                                 annealing_steps=args.n_annealing)
        timing_results += timings

    write_timings(timing_results, args.output_file)

if __name__ == "__main__":
    main()
