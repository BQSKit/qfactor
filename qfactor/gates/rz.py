"""This module implements the RzGate class."""


import numpy as np

from qfactor import utils
from qfactor.gates import Gate


class RzGate ( Gate ):
    """A RzGate is a Quantum Z-rotation applied to a qubit."""

    def __init__ ( self, theta, location, fixed = False, check_params = True ):
        """
        Gate Constructor

        Args:
            theta (float): The gate's angle of rotation.

            location (int or tuple[int]): The qubit this gate is applied to.

            fixed (bool): True if the gate's unitary operation is immutable.

            check_params (bool): True implies parameters are checked for
                correctness.
        """

        if check_params:
            if not isinstance( theta, float ):
                raise TypeError( "Invalid theta angle, not a float." )

            if not isinstance( location, int ):
                if ( not utils.is_valid_location( location )
                     or len( location ) != 1 ):
                    raise TypeError( "Specified location is not valid."  )
            elif location < 0:
                raise ValueError( "Invalid location value." )

            if not isinstance( fixed, bool ):
                raise TypeError( "Invalid fixed parameter." )

        self.theta = theta
        if isinstance( location, int ):
            self.location = tuple( [ location ] )
        else:
            self.location = location
        self.gate_size = len( self.location )
        self.fixed = fixed

    @property
    def utry ( self ):
        return np.array( [ [ 1, 0 ],
                           [ 0, np.exp( 1j * self.theta ) ] ] )

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

        if self.fixed:
            return

        a = np.real( env[1, 1] )
        b = np.imag( env[1, 1] )
        arctan = np.arctan( b / a )

        if a < 0 and b > 0:
            arctan += np.pi
        elif a < 0 and b < 0:
            arctan -= np.pi

        new_theta = -arctan
        self.theta = ( ( 1 - slowdown_factor ) * new_theta
                       + slowdown_factor * self.theta )

    def __repr__ ( self ):
        """Gets a simple gate string representation."""

        return str( self.location )       \
               + ": Rz("                  \
               + str( self.theta )        \
               + ")"

