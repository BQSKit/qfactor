import pickle
import signal
import argparse
import numpy as np
import itertools as it
from timeit import default_timer as timer

from scipy.stats import unitary_group

import qfactor
from qfactor import CnotGate, Gate, optimize
from qfactor.tensors import CircuitTensor

import qsearch

import qfast
from qfast.decomposition.models.fixedmodel import FixedModel
from qfast.decomposition.optimizers.lbfgs import LBFGSOptimizer

from param_problem import ParamOptimizationProblem


def qfactor_solve ( pt ):
    dist = 1
    pt.data[ "retries" ] = 0
    pt.data[ "retry_times" ] = []
    while dist > 1e-8:
        start = timer()
        pt.data[ "retries" ] += 1
        res = optimize( pt.get_qfactor(), pt.target, min_iters = 0 )
        dist = qfactor.get_distance( res, pt.target )
        end = timer()
        pt.data[ "retry_times" ].append( end - start )


def qfast_solve ( pt ):
    dist = 1
    pt.data[ "retries" ] = 0
    pt.data[ "retry_times" ] = []
    while dist > 1e-8:
        start = timer()
        pt.data[ "retries" ] += 1
        model = pt.get_qfast()
        model.optimize( fine = True )
        dist = model.distance()
        end = timer()
        pt.data[ "retry_times" ].append( end - start )
    

def qsearch_solve ( pt ):
    solver = qsearch.solvers.LeastSquares_Jac_SolverNative()
    options = qsearch.options.Options()
    options.target  = pt.target
    pt.data[ "retries" ] = 0
    dist = 1
    circ = pt.get_qsearch()
    pt.data[ "retry_times" ] = []
    while dist > 1e-8:
        start = timer()
        pt.data[ "retries" ] += 1
        U, xopts = solver.solve_for_unitary( circ, options )
        dist = 1 - ( np.abs( np.trace( pt.target.conj().T @ U ) ) / U.shape[0] )
        end = timer()
        pt.data[ "retry_times" ].append( end - start )


class TrialTerminatedException ( Exception ):
    """Custom timeout or interrupt Exception."""


def term_trial ( signal_number, frame ):
    """Terminate a Trial"""

    msg = "Error"

    if signal_number == signal.SIGINT:
        msg = "Manually Interrupted"

    if signal_number == signal.SIGALRM:
        msg = "Timed-out"

    print( msg )
    raise TrialTerminatedException()


def run_benchmark ( num_qubits, gate_size, length, test_qfactor, timeout = 120*60 ):
    
    # Register Signal Handlers
    signal.signal( signal.SIGALRM, term_trial )
    signal.signal( signal.SIGINT, term_trial )

    # Set Seed
    np.random.seed( 21211411 )
    pts = []

    # Solve Param Problems
    num_solved = 0
    signal.alarm( timeout )
    try:
        while True:
            pt = ParamOptimizationProblem.generate_circuit( gate_size = gate_size,
                                                            num_qubits = num_qubits,
                                                            length = length )
            pts.append( pt )
            if test_qfactor:
                qfactor_solve( pt )
            elif gate_size == 1:
                qsearch_solve( pt )
            else:
                qfast_solve( pt )
            num_solved += 1
    except TrialTerminatedException:
        print( "Times Up! Solved: %d " % num_solved )

    # Save results
    filename = "%dq_%dg_%dd_%ds.dat" % ( num_qubits, gate_size, length, timeout )
    if test_qfactor:
        filename = "qfactor_" + filename
    elif gate_size == 1:
        filename = "qsearch_" + filename
    else:
        filename = "qfast_" + filename
    with open( filename, "wb" ) as f:
        pickle.dump( num_solved, f )
        pickle.dump( pts, f )

if __name__ == "__main__":
    description_info = "Generate and optimize random circuits."

    parser = argparse.ArgumentParser( description = description_info )

    parser.add_argument( "numqubits", type = int,
                         help = "Total number of qubits." )

    parser.add_argument( "gatesize", type = int,
                         help = "Gate Size (1 for native gates)." )

    parser.add_argument( "length", type = int,
                         help = "Length of ansatz." )

    parser.add_argument( "timeout", type = int,
                         help = "Timeout in seconds for each trial." )
    
    parser.add_argument( "--testqfactor", action = "store_true",
                         help = "Test Qfactor or the other stuff." )

    args = parser.parse_args()

    run_benchmark( args.numqubits, args.gatesize, args.length, args.testqfactor, args.timeout )
