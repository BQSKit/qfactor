import numpy as np
import itertools as it
from scipy.stats import unitary_group

import qfactor
from qfactor import CnotGate, Gate, optimize
from qfactor.tensors import CircuitTensor

import qsearch

import qfast
from qfast.decomposition.models.fixedmodel import FixedModel
from qfast.decomposition.optimizers.lbfgs import LBFGSOptimizer

class ParamOptimizationProblem():
    
    def __init__ ( self, gate_size, num_qubits, locations, native ):
        self.gate_size = gate_size
        self.num_qubits = num_qubits
        self.locations = [ tuple( [ int( x ) for x in location ] )
                           for location in locations ]
        self.native = native

        self.target = CircuitTensor( np.identity( 2 ** self.num_qubits ),
                                     self.get_qfactor() ).utry

        self.data = {}

    @staticmethod
    def generate_circuit ( gate_size, num_qubits, length ):
        native = gate_size <= 1
        gate_size = 2 if native else gate_size
        locations = list( it.combinations( range( num_qubits ), gate_size ) )
        locations = np.array( locations )
        idxs = np.random.choice( len( locations ), length, replace = True )
        locations = locations[ idxs ]
        return ParamOptimizationProblem( gate_size, num_qubits, locations, native )
    
    def get_qfactor ( self ):
        circuit = []

        if self.native:
            for pair in self.locations:
                circuit.append( CnotGate( pair[0], pair[1] ) )
                circuit.append( Gate( unitary_group.rvs( 2 ), ( pair[0], ) ) )
                circuit.append( Gate( unitary_group.rvs( 2 ), ( pair[1], ) ) )
            return circuit

        for location in self.locations:
            circuit.append( Gate( unitary_group.rvs( 2 ** self.gate_size ),
                                  location ) )
        return circuit
    
    def get_qsearch ( self ):
        if not self.native:
            return None

        steps = []
        u30 = qsearch.gates.U3Gate()
        u31 = qsearch.gates.U3Gate()
        for pair in self.locations:
            min_idx = min( pair )
            max_idx = max( pair )

            cnot = qsearch.gates.NonadjacentCNOTGate( max_idx - min_idx + 1,
                                                      pair[0] - min_idx,
                                                      pair[1] - min_idx )
            if max_idx - min_idx == 1:
                u_layer = qsearch.gates.KroneckerGate( u30, u31 )
            else:
                mid_layer = qsearch.gates.IdentityGate( max_idx - min_idx - 1 )
                u_layer = qsearch.gates.KroneckerGate( u30, mid_layer, u31 )
            p_layer = qsearch.gates.ProductGate( cnot, u_layer )


            up = qsearch.gates.IdentityGate( min_idx )
            down = qsearch.gates.IdentityGate( self.num_qubits - max_idx - 1 )
            steps.append( qsearch.gates.KroneckerGate( up, p_layer, down ) )
        return qsearch.gates.ProductGate( *steps )

    def get_qfast ( self ):
        return FixedModel( self.target, self.gate_size, [],
                           LBFGSOptimizer(), success_threshold = 1e-8,
                           structure = self.locations )
