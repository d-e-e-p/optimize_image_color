import os, sys
from scipy.optimize import basinhopping
from scipy.optimize import minimize
import pdb

from optimize_color import cmd


def run_opt(e,method):
    if (method == 'basinhopping'):
        res = basinhopping(e.fun, e.x0, disp=True)
    else:
        res = minimize(e.fun, e.x0, method=method, bounds=e.bounds, constraints=e.constraints, 
            tol=0.001, options={'disp': True, 'maxiter':10000})
    print(res)
    return res


def execute(args):
    #pdb.set_trace()
    # setup problem to solve
    e = cmd.Exp(args)
    for method in args.solver:
        res = run_opt(e,method)
        if res.success:
            e.x0 = res.x
        else:
            raise ValueError(res.message)
    print(args)
    print("done")
    


#

