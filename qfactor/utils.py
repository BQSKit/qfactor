"""This module contains various utility functions."""

import logging

import numpy as np


logger = logging.getLogger( "qfactor" )


def get_num_qubits ( M ):
    """Returns the size of the square matrix, M, in qubits."""

    if not is_square_matrix( M ):
        raise TypeError( "Invalid matrix." )

    return int( np.log2( len( M ) ) )


def is_valid_location ( location, num_qubits = None ):
    """
    Checks if the location is valid.

    Args:
        location (Tuple[int]): The location to check.

        num_qubits (int or None): The total number of qubits. All qubits
            should be less than this. If None, don't check.

    Returns:
        (bool): Valid or not
    """

    if not isinstance( location, tuple ):
        logger.debug( "Location is not a tuple." )
        return False

    if not all( [ isinstance( qubit, int ) for qubit in location ] ):
        logger.debug( "Location is not a tuple of ints." )
        return False

    if len( location ) != len( set( location ) ):
        logger.debug( "Location has duplicates." )
        return False

    if not all( [ x == y for x, y in zip( location, sorted( location ) ) ] ):
        logger.debug( "Location not sorted." )
        return False

    if num_qubits is not None:
        if not all( [ qubit < num_qubits for qubit in location ] ):
            logger.debug( "Location has an invalid qubit." )
            return False

    return True


def is_matrix ( M ):
    """Checks if M is a matrix."""

    if not isinstance( M, np.ndarray ): 
        logger.debug( "M is not an numpy array." )
        return False

    if len( M.shape ) != 2:
        logger.debug( "M is not an 2-dimensional array." )
        return False

    return True


def is_square_matrix ( M ):
    """Checks if M is a square matrix."""

    if not is_matrix( M ):
        return False

    if M.shape[0] != M.shape[1]:
        return False

    return True


def is_unitary ( U, tol = 1e-14 ):
    """Checks if U is a unitary matrix."""

    if not is_square_matrix( U ):
        return False

    X = U @ U.conj().T
    Y = U.conj().T @ U
    I = np.identity( X.shape[0] )

    if not np.allclose( X, I, rtol = 0, atol = tol ):
        if logger.isEnabledFor( logging.DEBUG ):
            norm = np.linalg.norm( X - I )
            logger.debug( "Failed unitary condition, ||UU^d - I|| = %e" % norm )
        return False
    
    if not np.allclose( Y, I, rtol = 0, atol = tol ):
        if logger.isEnabledFor( logging.DEBUG ):
            norm = np.linalg.norm( Y - I )
            logger.debug( "Failed unitary condition, ||U^dU - I|| = %e" % norm )
        return False
    
    return True

