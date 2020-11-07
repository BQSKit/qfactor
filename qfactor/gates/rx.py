"""This module implements the RxGate class."""


import numpy as np

from qfactor import utils
from qfactor.gates import Gate


class RxGate ( Gate ):
    """A RxGate is a Quantum X-rotation applied to a qubit."""

    def __init__ ( self, theta, location, fixed = False, check_params = True ):
        """
        Gate Constructor

        Args:
            theta (float): The gate's angle of rotation.

            location (int): The qubit this gate is applied to.

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
        cos = np.cos( self.theta / 2 )
        sin = np.sin( self.theta / 2 )
        return np.array( [ [ cos, -1j * sin ],
                           [ -1j * sin, cos ] ] )

    def update ( self, env, slowdown_factor ):
        """
        Update this gate with respect to an enviroment.

        This method updates this gate's unitary to maximize:
            Re( Tr( env * self.utry ) )

        Args:
            env (np.ndarray): The enviromental matrix.

            slowdown_factor (int): The larger this factor, the slower
                the optimization happens.
        """

        if self.fixed:
            return

        a = np.real( env[0, 0] + env[1, 1] )
        b = np.imag( env[0, 1] + env[1, 0] )
        self.theta = 2 * np.arccos( a / np.sqrt( a ** 2 + b ** 2 ) )

    def __repr__ ( self ):
        """Gets a simple gate string representation."""

        return str( self.location )       \
               + ": Rx("                  \
               + str( self.theta )        \
               + ")"

