import os, sys
import pdb

# global
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing

# global
from scipy.optimize import minimize

from optimize_color import cmd

def run_opt(e):
    if    (e.method == 'basinhopping'):           res = basinhopping(e.fun, e.x0, disp=True)
    elif  (e.method == 'differential_evolution'): res = differential_evolution(e.fun, bounds=e.bounds, disp=True, polish=True)
    elif  (e.method == 'dual_annealing'):         res = dual_annealing(e.fun, bounds=e.bounds, no_local_search=False )
    else:
        res = minimize(e.fun, e.x0, method=e.method, bounds=e.bounds, constraints=e.constraints, 
            tol=0.001, options={'disp': True, 'maxiter':10000})
    print(res)

    if hasattr(res, 'lowest_optimization_result'):
        lowest_optimization_result = res.lowest_optimization_result
    elif hasattr(res, 'fun'):
        lowest_optimization_result = res.fun
    else:
        lowest_optimization_result = 0

    return lowest_optimization_result, res.x

#
# expand keys in list if entry in dict, eg
#   adict = {'f': [4, 5]}
#   alist = ['a','f']
#   blist = ['a', 4, 5]
# only 1 trip allowed!
#
def expand_keys(alist, adict):
    blist = []
    for elem in alist:
        if elem in adict:
            blist.extend(adict[elem])
        else:
            blist.append(elem)
    return blist


def execute(args, operations,  solvers):
    # setup problem to solve
    # meta methods
    if args.operation is None: 
        operation_sequence = expand_keys(['default'], operations)
    else:
        operation_sequence = expand_keys(args.operation, operations)

    if args.solver is None: 
        solver_sequence = expand_keys(['default'], solvers)
    else:
        solver_sequence = expand_keys(args.solver, solvers)

    print(f"operation_sequence: {operation_sequence}")
    print(f"solver_sequence: {solver_sequence}")
    for op in operation_sequence:
        e = cmd.Exp(args, op) # setup bounds and x0 etc
        for e.method in solver_sequence:
            print(f"running operation: {op} using  method: {e.method} ------------------")
            #pdb.set_trace()
            lowest_optimization_result, e.x0 = run_opt(e)
            print(f"lowest_optimization_result: {lowest_optimization_result} operation: {op} method: {e.method}")
    print(args)
    print("done")


#

