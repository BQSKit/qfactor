"""This module implements the main optimize function."""

import logging

import numpy as np

from qfactor import utils
from qfactor.gates import Gate
from qfactor.tensors import CircuitTensor


logger = logging.getLogger( "qfactor" )


def optimize ( circuit, target, diff_tol = 1e-10, dist_tol = 1e-10,
               max_iters = 100000, min_iters = 1000, slowdown_factor = 0 ):
    """
    Optimize distance between circuit and target unitary.

    Args:
        circuit (list[Gate]): The circuit to optimize.

        target (np.ndarray): The target unitary matrix.

        diff_tol (float): Terminate when the difference in distance
            between iterations is less than this threshold.

        dist_tol (float): Terminate when the distance is less than
            this threshold.

        max_iters (int): Maximum number of iterations.

        min_iters (int): Minimum number of iterations.

        slowdown_factor (int): The larger this factor, the slower the
            optimization happens.

    Returns:
        (list[Gate]): The optimized circuit.
    """

    if not isinstance( circuit, list ):
        raise TypeError( "The circuit argument is not a list of gates." )

    if not all( [ isinstance( g, Gate ) for g in circuit ] ):
        raise TypeError( "The circuit argument is not a list of gates." )

    if not utils.is_unitary( target ):
        raise TypeError( "The target matrix is not unitary." )

    if not isinstance( diff_tol, float ) or diff_tol > 0.5:
        raise TypeError( "Invalid difference threshold." )

    if not isinstance( dist_tol, float ) or dist_tol > 0.5:
        raise TypeError( "Invalid distance threshold." )

    if not isinstance( max_iters, int ) or max_iters < 0:
        raise TypeError( "Invalid maximum number of iterations." )

    if not isinstance( min_iters, int ) or min_iters < 0:
        raise TypeError( "Invalid minimum number of iterations." )

    if not isinstance( slowdown_factor, int ) or slowdown_factor < 0:
        raise TypeError( "Invalid slowdown factor." )

    ct = CircuitTensor( target, circuit )

    c1 = 0
    c2 = 1
    it = 0

    while it < min_iters or np.abs(c1 - c2) > diff_tol and it < max_iters:
        it += 1

        # from right to left
        for k in range( len( circuit ) ):
            rk = len( circuit ) - 1 - k

            # Remove current gate from right of circuit tensor
            ct.apply_right( circuit[rk], inverse = True )

            # Update current gate
            if not circuit[rk].fixed:
                env = ct.calc_env_matrix( circuit[rk].location )
                circuit[rk].update( env, slowdown_factor )

            # Add updated gate to left of circuit tensor
            ct.apply_left( circuit[rk] )

        # from left to right
        for k in range( len( circuit ) ):

            # Remove current gate from left of circuit tensor
            ct.apply_left( circuit[k], inverse = True )

            # Update current gate
            if not circuit[k].fixed:
                env = ct.calc_env_matrix( circuit[k].location )
                circuit[k].update( env, slowdown_factor )

            # Add updated gate to right of circuit tensor
            ct.apply_right( circuit[k] )

        c2 = c1
        c1 = np.real( np.trace( ct.utry ) )
        c1 = 1 - ( c1 / ( 2 ** ct.num_qubits ) )

        if c1 <= dist_tol:
            logger.info( f"Terminated c1 = {c1} <= dist_tol." )
            return circuit

        if it % 100 == 0:
            logger.info( f"iteration: {it}, cost: {c1}" )

        if it % 40 == 0:
            ct.reinitialize()

    if it >= max_iters:
        logger.info( "Iteration limit reached." )

    if np.abs(c1 - c2) <= diff_tol:
        diff = np.abs(c1 - c2)
        logger.info( f"Terminated |c1 - c2| = {diff} <= diff_tol." )

    return circuit

