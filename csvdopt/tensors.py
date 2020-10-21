"""This module implements the CircuitTensor class."""

import logging

import numpy as np

from csvdopt import utils
from csvdopt.gate import Gate


logger = logging.getLogger( "csvdopt" )


class CircuitTensor():
    """A CircuitTensor tracks an entire circuit as a tensor."""

    def __init__ ( self, utry_target, gate_list ):
        """
        CircuitTensor Constructor

        Args:
            utry_target (np.ndarray): Unitary target matrix

            gate_list (list[Gate]): The circuit's gate list.
        """

        if not utils.is_unitary( utry_target ):
            raise TypeError( "Specified target matrix is not unitary." )

        if not isinstance( gate_list, list ):
            raise TypeError( "Gate list is not a list." )

        if not all( [ isinstance( gate, Gate ) for gate in gate_list ] ):
            raise TypeError( "Gate list contains non-gate objects." )

        self.utry_target = utry_target
        self.num_qubits = utils.get_num_qubits( self.utry_target )

        if not all( [ utils.is_valid_location( gate.location, self.num_qubits )
                      for gate in gate_list ] ):
            raise ValueError( "Gate location mismatch with circuit tensor." )

        self.gate_list = gate_list
        self.reinitialize()

    def reinitialize ( self ):
        """Reconstruct the circuit tensor."""
        logger.debug( "Reinitializing CircuitTensor" )

        self.tensor = self.utry_target.conj().T
        self.tensor = self.tensor.reshape( [2] * 2 * self.num_qubits )

        for gate in self.gate_list:
            self.apply_right( gate )

    @property
    def utry ( self ):
        """Calculates this circuit tensor's unitary representation."""
        num_elems = 2 ** self.num_qubits
        return self.tensor.reshape( ( num_elems, num_elems ) )

    def apply_right ( self, gate ):
        """
        Apply the specified gate on the right on the circuit.

             .-----.   .------.
          0 -|     |---|      |-
          1 -|     |---| gate |-
             .     .   '------'
             .     .
             .     .
        n-1 -|     |------------
             '-----'
        
        Note that apply the gate on the right is equivalent to
        multiplying on the gate on the left of the tensor.
        This operation is performed using tensor contraction.

        Args:
            gate (Gate): The gate to apply.
        """

        if not isinstance( gate, Gate ):
            raise TypeError( "Invalid gate." )

        left_perm = list( range( self.num_qubits ) )
        mid_perm = [ x + self.num_qubits for x in left_perm if x not in gate.location ]
        right_perm = [ x + self.num_qubits for x in gate.location ]

        perm = left_perm + mid_perm + right_perm
        self.tensor = self.tensor.transpose( perm )
        self.tensor = self.tensor.reshape( ( 2 ** len( right_perm ), -1 ) )
        self.tensor = gate.utry @ self.tensor

        self.tensor = self.tensor.reshape( [2] * 2 * self.num_qubits )
        inv_perm = np.argsort( perm )
        self.tensor = self.tensor.transpose( inv_perm )


    def apply_left ( self, gate ):
        """
        Apply the specified gate on the left on the circuit.

             .------.   .-----.
          0 -|      |---|     |-
          1 -| gate |---|     |-
             '------'   .     .
                        .     .
                        .     .
        n-1 ------------|     |-
                        '-----'
        
        Note that apply the gate on the left is equivalent to
        multiplying on the gate on the right of the tensor.
        This operation is performed using tensor contraction.

        Args:
            gate (Gate): The gate to apply.
        """

        if not isinstance( gate, Gate ):
            raise TypeError( "Invalid gate." )

        left_perm = list( range( self.num_qubits ) )
        mid_perm = [ x + self.num_qubits for x in left_perm if x not in gate.location ]
        right_perm = [ x + self.num_qubits for x in gate.location ]

        perm = left_perm + mid_perm + right_perm
        self.tensor = self.tensor.transpose( perm )
        self.tensor = self.tensor.reshape( ( -1, 2 ** len( right_perm ) ) )
        self.tensor = self.tensor @ gate.utry

        self.tensor = self.tensor.reshape( [2] * 2 * self.num_qubits )
        inv_perm = np.argsort( perm )
        self.tensor = self.tensor.transpose( inv_perm )

    def calc_env_matrix ( self, location ):
        """
        Calculates the environmental matrix of the tensor with
        respect to the specified location.

        Args:
            location (iterable): Calculate the environment for this
                set of qubits.

        Returns:
            (np.ndarray): The environmental matrix.
        """

        if not utils.is_valid_location( location, self.num_qubits ):
            raise TypeError( "Invalid location." )

        left_perm = list( range( self.num_qubits ) )
        left_perm = [ x for x in left_perm if x not in location ]
        left_perm = left_perm + [ x + self.num_qubits for x in left_perm ]
        right_perm = list( location ) + [ x + self.num_qubits
                                          for x in location ]

        perm = left_perm + right_perm
        a = np.transpose( self.tensor, perm )
        a = np.reshape( a, ( 2 ** ( self.num_qubits - len( location ) ),
                             2 ** ( self.num_qubits - len( location ) ),
                             2 ** len( location ),
                             2 ** len( location ) ) )
        return np.trace( a )

