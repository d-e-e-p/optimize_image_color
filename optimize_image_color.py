#!/usr/bin/python3.8
#
#
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

from exitstatus import ExitStatus

from optimize_color import opt

operations = """
dng_wb dng_ccm
rgb_equation_spline rgb_equation_multi_ccm
rgb_curves channel_mixer color_toning luminance_curve hsv_equalizer
""".split()

solvers = """
nelder-mead  powell  cg  bfgs  
l-bfgs-b  tnc  cobyla  slsqp  trust-constr 
basinhopping
""".split()


def parse_args() -> argparse.Namespace:
    """Parse user command line arguments."""
    parser = argparse.ArgumentParser(
        description='run color correction optimization on macbeth image')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--tag')
    parser.add_argument('--src')
    parser.add_argument('--dst')
    parser.add_argument('--operation', choices=operations)
    parser.add_argument("--solver", action="extend", nargs="+", type=str, choices=solvers)
    return parser.parse_args()


def main() -> ExitStatus:
    args = parse_args()
    print(args)
    opt.execute(args)

    return ExitStatus.success


if __name__ == "__main__":
    sys.exit(main())
