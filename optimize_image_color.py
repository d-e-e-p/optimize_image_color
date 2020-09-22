#!/usr/bin/python3.8
#
# improve color in raw images by optimizing color correction matrix 
# and other methods
#
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

from exitstatus import ExitStatus

from optimize_color import opt

__author__      = 'deep@tensorfield.ag'
__copyright__   = 'Copyright (C) 2020 - Tensorfield Ag '
__license__     = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__  = 'deep'
__status__      = 'Development'


#
# problem to run for optimization
#
operations = {}

# white balance and color correction matrix
operations['dng_all'] = "dng_wb dng_wb_bl dng_ccm dng_ccm_bl dng_wb_ccm dng_wb_ccm_bl".split()

# use rt to solve color issues
operations['rt_all'] = """
    rt_rgb_curves
    rt_channel_mixer
    rt_color_toning
    rt_luminance_curve
    rt_hsv_equalizer
""".split();

# operate equation on rgb input values
operations['eq_all'] = """
    eq_spline
    eq_root_polynomial
    eq_multi_ccm2
    eq_multi_ccm4
""".split();

# expanded options (also form valid groupings for arguments)
# ie calling wb_ccm will invoke both dng_wb and dng_ccm
operations['all'] = operations['dng_all'] + operations['rt_all'] + operations['eq_all']
operations['default'] = "dng_ccm_bl rt_rgb_curves".split()

# expand keys of dict as options, eg -operation default
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
