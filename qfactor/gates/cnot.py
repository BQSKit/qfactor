"""This module implements the CnotGate class."""


import numpy as np

from qfactor import utils
from qfactor.gates import Gate


class CnotGate ( Gate ):
    """A CnotGate is a controlled-not applied to a pair of qubit."""

    def __init__ ( self, control, target, check_params = True ):
        """
        Gate Constructor

        Args:
            control (int): The index of the control qubit.

            target (int): The index of the target qubit.

            check_params (bool): True implies parameters are checked for
                correctness.
        """

        if check_params:
            if not isinstance( control, int ) or control < 0:
                raise TypeError( "Invalid control qubit." )

            if not isinstance( target, int ) or target < 0:
                raise TypeError( "Invalid target qubit." )

        self.utry = np.array( [ [ 1, 0, 0, 0 ],
                                [ 0, 1, 0, 0 ],
                                [ 0, 0, 0, 1 ],
                                [ 0, 0, 1, 0 ] ] )
        self.location = tuple( [ control, target ] )
        self.gate_size = 2
        self.fixed = True

    def update ( self, env, slowdown_factor ):
        """
        Update this gate with respect to an enviroment.
        Note this gate is fixed and never updates. This
        function is implemented to satisfy the API.

        Args:
            env (np.ndarray): The enviromental matrix.

            slowdown_factor (float): A positive number less than 1. 
                The larger this factor, the slower the optimization.
        """
        return

    def __repr__ ( self ):
        """Gets a simple gate string representation."""

        return str( self.location )       \
               + ": CNOT"

