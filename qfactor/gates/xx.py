"""This module implements the XXGate class."""


import numpy as np

from qfactor import utils
from qfactor.gates import Gate


class XXGate ( Gate ):
    """A XXGate is a Quantum XX-rotation applied to a qubit."""

    def __init__ ( self, theta, location, fixed = False, check_params = True ):
        """
        Gate Constructor

        Args:
            theta (float): The gate's angle of rotation.

            location (tuple[int]): The qubits this gate is applied to.

            fixed (bool): True if the gate's unitary operation is immutable.

            check_params (bool): True implies parameters are checked for
                correctness.
        """

        if check_params:
            if not isinstance( theta, float ):
                raise TypeError( "Invalid theta angle, not a float." )

            if not isinstance( location, tuple ):
                raise TypeError( "Specified location is not valid."  )

            if not utils.is_valid_location( location ):
                raise TypeError( "Specified location is not valid."  )

            if len( location ) != 2:
                raise TypeError( "Specified location is not valid."  )

            if not isinstance( fixed, bool ):
                raise TypeError( "Invalid fixed parameter." )

        self.theta = theta
        self.location = location
        self.gate_size = len( self.location )
        self.fixed = fixed

    @property
    def utry ( self ):
        cos = np.cos( self.theta / 2 )
        isin = -1j * np.sin( self.theta / 2 )
        return np.array( [ [ cos, 0, 0, isin ],
                           [ 0, cos, isin, 0 ],
                           [ 0, isin, cos, 0 ],
                           [ isin, 0, 0, cos ] ] )

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

        a = np.real( env[0, 0] + env[1, 1] + env[2, 2] + env[3, 3] )
        b = np.imag( env[0, 3] + env[1, 2] + env[2, 1] + env[3, 0] )
        new_theta = np.arccos( a / np.sqrt( a ** 2 + b ** 2 ) )
        new_theta *= -2 if b < 0 else 2
        self.theta = ( ( 1 - slowdown_factor ) * new_theta
                       + slowdown_factor * self.theta )

    def __repr__ ( self ):
        """Gets a simple gate string representation."""

        return str( self.location )       \
               + ": XX("                  \
               + str( self.theta )        \
               + ")"

