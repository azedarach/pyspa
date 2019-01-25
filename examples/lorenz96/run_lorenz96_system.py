import argparse
import numpy as np

from math import pi

from lorenz_systems import Lorenz96II

def periodic_forcing(t, x, y, amp=10.0, period=80.0):
    return amp * np.sin(2.0 * pi * t / period) * np.ones(np.size(x))

def constant_forcing(t, x, y, F=10.0):
    return F

def get_initial_conditions(n, m):
    x0 = np.random.random((n,))
    y0 = np.random.random((n,m))
    return (x0, y0)

def generate_data(model, x0, y0, length, time_step, t0=0):
    t1 = t0 + (length - 1.0) * time_step
    t_vals = np.linspace(t0, t1, length)
    return model.evolve(x0, y0, t1, t0, t_eval=t_vals)

def get_header_line(n, m):
    header = "# t,"
    header = header + ",".join(["xt" + str(i) for i in range(1, n + 1)])
    header = header + "," + ",".join(["yt" + str(i) + str(j)
                                      for i in range(1, n + 1)
                                      for j in range(1, m + 1)])
    header = header + "\n"
    return header

def get_data_lines(t, x, y):
    length = np.size(t)
    n = x.shape[0]
    m = y.shape[1]

    lines = []
    for i in range(length):
        line = "{:<14.8e}".format(t[i])
        xdat = ",".join(["{:<14.8e}".format(val) for val in x[:,i]])
        ydat = ",".join(["{:<14.8e}".format(y[j,k,i]) for j in range(n)
                         for k in range(m)])
        line = line + "," + xdat + "," + ydat + "\n"
        lines.append(line)

    return lines

def write_data(t, x, y, output_file=""):
    n = x.shape[0]
    m = y.shape[1]

    header_line = get_header_line(n, m)
    data_lines = get_data_lines(t, x, y)
    lines = [header_line] + data_lines

    if output_file:
        with open(output_file, "w") as ofs:
            ofs.writelines(lines)
    else:
        for line in lines:
            print(line)

def parse_cmd_line_args():
    parser = argparse.ArgumentParser(
        description="Generate time-series of Lorenz 96 model")
    parser.add_argument("--length", dest="length", type=int,
                        default=1000, help="length of time-series")
    parser.add_argument("--time-step", dest="time_step", type=float,
                        default=0.01, help="time-series time step")
    parser.add_argument("-n", dest="n", type=int,
                        default=8, help="value of parameter n")
    parser.add_argument("-m", dest="m", type=int,
                        default=4, help="value of parameter m")
    parser.add_argument("--h-value", dest="h", type=float,
                        default=1.0, help="value of parameter h")
    parser.add_argument("--b-value", dest="b", type=float,
                        default=10.0, help="value of parameter b")
    parser.add_argument("--c-value", dest="c", type=float,
                        default=10.0, help="value of parameter c")
    parser.add_argument("--forcing-type", dest="forcing_type",
                        choices=["periodic", "constant"],
                        default="periodic", help="form of forcing term")
    parser.add_argument("--forcing-amp", dest="forcing_amp", type=float,
                        default=10.0, help="size of forcing")
    parser.add_argument("--output-file", dest="output_file", default="",
                        help="output data file")

    args = parser.parse_args()

    return args

def main():
    args = parse_cmd_line_args()

    if args.forcing_type == "periodic":
        forcing = lambda t, x, y : periodic_forcing(t, x, y,
                                                    amp=args.forcing_amp)
    else:
        forcing = lambda t, x, y : constant_forcing(t, x, y,
                                                    F=args.forcing_amp)

    model = Lorenz96II(n=args.n, m=args.m, h=args.h, b=args.b, c=args.c,
                       F=forcing)

    (x0, y0) = get_initial_conditions(args.n, args.m)
    (t, x, y) = generate_data(model, x0, y0, args.length, args.time_step)

    write_data(t, x, y, args.output_file)

if __name__ == "__main__":
    main()
