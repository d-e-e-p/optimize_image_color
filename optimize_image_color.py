#!/usr/bin/python3.8
#
#
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

from exitstatus import ExitStatus

from optimize_color import opt

#
# problem to run for optimization
#
operations = {}
# leave out rgb_equation_multi_ccm for now
operations['wb_ccm'] = "dng_wb dng_ccm".split()
operations['rt_opt'] = """
    rgb_equation_spline
    rgb_curves
    channel_mixer
    color_toning
    luminance_curve
    hsv_equalizer
""".split();
# expanded options (also form valid groupings for arguments)
# ie calling wb_ccm will invoke both dng_wb and dng_ccm
operations['all'] = operations['wb_ccm'] + operations['rt_opt']
operations['default'] = "dng_ccm rgb_curves".split()

operations_choices = operations['all'] + list(operations.keys())

#
# solver to use to reduce cost function
#
solvers = {}
solvers['local'] = """
    nelder-mead
    powell
    cg
    bfgs
    l-bfgs-b
    tnc
    cobyla
    slsqp
    trust-constr
""".split()

solvers['global'] = """
    differential_evolution
    dual_annealing
    basinhopping
""".split()

# expanded options
# finetune must take initial seed x0 solution and carry it over...
solvers['finetune'] = "basinhopping nelder-mead powell cobyla slsqp".split()
solvers['all'] = solvers['global'] + solvers['local'] 
solvers['default'] = "nelder-mead cobyla".split()
solvers_choices = solvers['all'] + list(solvers.keys())

def parse_args() -> argparse.Namespace:
    """Parse user command line arguments."""
    parser = argparse.ArgumentParser(
        description='run color correction optimization on macbeth image')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--tag')
    parser.add_argument('--src')
    parser.add_argument('--dst')
    parser.add_argument("--operation", action="extend", nargs="+", type=str, choices=operations_choices)
    parser.add_argument("--solver", action="extend", nargs="+", type=str, choices=solvers_choices)
    return parser.parse_args()


def main() -> ExitStatus:
    args = parse_args()
    print(args)
    opt.execute(args, operations,  solvers)

    return ExitStatus.success


if __name__ == "__main__":
    sys.exit(main())
