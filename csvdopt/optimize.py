"""This module implements the main optimize function."""

import logging

import numpy as np
import scipy.linalg as la

from csvdopt import utils
from csvdopt.gate import Gate
from csvdopt.tensors import CircuitTensor


logger = logging.getLogger( "csvdopt" )


def optimize ( circuit, target, threshold = 1e-10,
               max_iters = 100000, slowdown_factor = 0 ):
    """
    Optimize distance between circuit and target unitary.

    Args:
        circuit (list[Gate]): The circuit to optimize.

        target (np.ndarray): The target unitary matrix.

        threshold (float): Terminate when the difference in distance
            between iterations is less than threshold.

        max_iters (int): Maximum number of iterations.

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

    if not isinstance( threshold, float ) or threshold > 0.5:
        raise TypeError( "Invalid threshold." )

    if not isinstance( max_iters, int ) or max_iters < 0:
        raise TypeError( "Invalid maximum number of iterations." )

    if not isinstance( slowdown_factor, int ) or slowdown_factor < 0:
        raise TypeError( "Invalid slowdown factor." )

    ct = CircuitTensor( target, circuit )

    c1 = 0
    c2 = 1
    it = 0

    while np.abs(c1 - c2) > threshold and it < max_iters:
        it += 1

        # from right to left
        for k in range( len( circuit ) ):
            rk = len( circuit ) - 1 - k

            # Remove current gate from right of circuit tensor
            inv_gate = circuit[rk].get_inverse()
            ct.apply_right( inv_gate )

            # Update current gate
            if not circuit[rk].fixed:
                env = ct.calc_env_matrix( circuit[rk].location )
                u, _, v = la.svd( env + slowdown_factor * inv_gate.utry )
                circuit[rk] = Gate( v.conj().T @ u.conj().T, circuit[rk].location )

            # Add updated gate to left of circuit tensor
            ct.apply_left( circuit[rk] )

        # from left to right
        for k in range( len( circuit ) ):

            # Remove current gate from left of circuit tensor
            inv_gate = circuit[k].get_inverse()
            ct.apply_left( inv_gate )

            # Update current gate
            if not circuit[k].fixed:
                env = ct.calc_env_matrix( circuit[k].location )
                u, _, v = la.svd( env + slowdown_factor * inv_gate.utry )
                circuit[k] = Gate( v.conj().T @ u.conj().T, circuit[k].location )

            # Add updated gate to right of circuit tensor
            ct.apply_right( circuit[k] )

        c2 = c1
        c1 = np.trace( ct.utry )
        c1 = (2 ** (ct.num_qubits+1)) - (2 * np.real( c1 ))

        if it % 100 == 0:
            logger.info( f"iteration: {it}, cost: {c1}" )

        if it % 1000 == 0 and it > 0:
            ct.reinitialize()

    return circuit

