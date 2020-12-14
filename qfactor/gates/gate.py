"""This module implements the Gate class."""


import numpy as np
import scipy.linalg as la

from qfactor import utils


class Gate():
    """A Gate is a unitary operation applied to a set of qubits."""

    def __init__ ( self, utry, location, fixed = False, check_params = True ):
        """
        Gate Constructor

        Args:
            utry (np.ndarray): The gate's unitary operation.

            location (tuple[int]): Set of qubits this gate is applied to.

            fixed (bool): True if the gate's unitary operation is immutable.

            check_params (bool): True implies parameters are checked for
                correctness.
        """

        if check_params:
            if not utils.is_unitary( utry ):
                raise TypeError( "Specified matrix is not unitary." )

            if not utils.is_valid_location( location ):
                raise TypeError( "Specified location is not valid."  )

            if len( location ) != utils.get_num_qubits( utry ):
                raise ValueError( "Location size does not match unitary." )

            if not isinstance( fixed, bool ):
                raise TypeError( "Invalid fixed parameter." )

        self.utry = utry
        self.location = location
        self.gate_size = len( location )
        self.fixed = fixed

    def get_inverse ( self ):
        """Returns the inverse of this gate."""
        return Gate( self.utry.conj().T, self.location, self.fixed, False )

    def update ( self, env, slowdown_factor ):
        """
        Update this gate with respect to an enviroment.

        This method updates this gate's unitary to maximize:
            Re( Tr( env * self.utry ) )

        Args:
            env (np.ndarray): The enviromental matrix.

            slowdown_factor (float): A positive number less than 1. 
                The larger this factor, the slower the optimization.
        """

        u, _, v = la.svd( ( 1 - slowdown_factor ) * env
                          + slowdown_factor * self.utry.conj().T )
        self.utry = v.conj().T @ u.conj().T

    def get_tensor_format ( self, compress_left = False,
                            compress_right = False ):
        """
        Converts the gate's operation into a tensor network format.

        Indices are counted top to bottom, left to right:
             .-----.
          0 -|     |- n
          1 -|     |- n+1
             .     .
             .     .
             .     .
        n-1 -|     |- 2n-1
             '-----'

        Args:
            compress_left (bool): Compress the left indices into one.

            compress_right (bool): Compress the right indices into one.

        Returns
            (np.ndarray): A tensor representing this gate.
        """

        if not isinstance( compress_left, bool ):
            raise TypeError( "Invalid compress_left parameter." )

        if not isinstance( compress_right, bool ):
            raise TypeError( "Invalid compress_right parameter." )

        dim = len( self.utry )
        left = [ dim ] if compress_left else [2] * ( self.gate_size )
        right = [ dim ] if compress_right else [2] * ( self.gate_size )

        return self.utry.reshape( left + right )

    def __str__ ( self ):
        """Gets the gate's string representation."""

        return str( self.location ) + ":" + str( self.utry )

    def __repr__ ( self ):
        """Gets a simple gate string representation."""

        return str( self.location )       \
               + ": [["                   \
               + str( self.utry[0][0] )   \
               + " ... "                  \
               + str( self.utry[-1][-1] ) \
               + "]]"

