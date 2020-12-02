import pickle
import signal
import argparse
import numpy as np
import itertools as it
from timeit import default_timer as timer

from scipy.stats import unitary_group

import qfactor
from qfactor import CnotGate, Gate, optimize
from qfactor.tensors import CircuitTensor

import qsearch

import qfast
from qfast.decomposition.models.fixedmodel import FixedModel
from qfast.decomposition.optimizers.lbfgs import LBFGSOptimizer


np.random.seed( 21211411 )

class TrialTerminatedException ( Exception ):
    """Custom timeout or interrupt Exception."""

def term_trial ( signal_number, frame ):
    """Terminate a Trial"""

    msg = "Error"

    if signal_number == signal.SIGINT:
        msg = "Manually Interrupted"

    if signal_number == signal.SIGALRM:
        msg = "Timed-out"

    print( msg )
    raise TrialTerminatedException()

class CircuitDataPoint():
    
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
        locations = list( it.combinations( range( num_qubits ), 2 ) )
        locations = np.array( locations )
        idxs = np.random.choice( len( locations ), length, replace = True )
        locations = locations[ idxs ]
        return CircuitDataPoint( gate_size, num_qubits, locations, native )
    
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

    def count_qfactor_tries ( self ):
        dist = 1
        tries = 0
        self.data[ "qfactor_retry_times" ] = []
        while dist > 1e-8:
            start = timer()
            tries += 1
            res = optimize( self.get_qfactor(), self.target, min_iters = 0 )
            dist = qfactor.get_distance( res, self.target )
            end = timer()
            self.data[ "qfactor_retry_times" ].append( end - start )
        self.data[ "qfactor_retries" ] = tries

    def count_qfast_tries ( self ):
        dist = 1
        tries = 0
        self.data[ "qfast_retry_times" ] = []
        while dist > 1e-8:
            start = timer()
            tries += 1
            model = self.get_qfast()
            model.optimize( fine = True )
            dist = model.distance()
            end = timer()
            self.data[ "qfast_retry_times" ].append( end - start )
        self.data[ "qfast_retries" ] = tries

    def count_qsearch_tries ( self ):
        solver = qsearch.solvers.LeastSquares_Jac_SolverNative()
        options = qsearch.options.Options()
        options.target  = self.target
        tries = 0
        dist = 1
        circ = self.get_qsearch()
        self.data[ "qsearch_retry_times" ] = []
        while dist > 1e-8:
            start = timer()
            tries += 1
            U, xopts = solver.solve_for_unitary( circ, options )
            dist = 1 - ( np.abs( np.trace( self.target.conj().T @ U ) ) / U.shape[0] )
            end = timer()
            self.data[ "qsearch_retry_times" ].append( end - start )
        self.data[ "qsearch_retries" ] = tries


def run_benchmark ( gate_size, length, num_qubits, num_circs, timeout = 60*60 ):
    
    # Register Signal Handlers
    signal.signal( signal.SIGALRM, term_trial )
    signal.signal( signal.SIGINT, term_trial )

    # Generate Circuits
    pts = [ CircuitDataPoint.generate_circuit( gate_size = gate_size,
                                               num_qubits = num_qubits,
                                               length = length )
            for x in range( num_circs ) ]

    # Optimize Circuits
    for pt in pts:

        try:
            # Count Qfactor Tries
            signal.alarm( timeout )
            pt.count_qfactor_tries()

            # Count either QFAST or QSearch
            signal.alarm( timeout )
            if gate_size == 1:
                pt.count_qsearch_tries()
            else:
                pt.count_qfast_tries()

        except TrialTerminatedException:
            pt.data[ "failed" ] = True

        print( pt.data )

    filename = "%dq_%dg_%dd_%dp.dat" % ( num_qubits, gate_size,
                                         length, num_circs )
    with open( filename, "wb" ) as f:
        pickle.dump( pts, f )

if __name__ == "__main__":
    description_info = "Generate and optimize random circuits."

    parser = argparse.ArgumentParser( description = description_info )

    parser.add_argument( "numqubits", type = int,
                         help = "Total number of qubits." )

    parser.add_argument( "gatesize", type = int,
                         help = "Gate Size (1 for native gates)." )

    parser.add_argument( "length", type = int,
                         help = "Length of ansatz." )

    parser.add_argument( "numcircs", type = int,
                         help = "Total number of circuits or data points." )

    parser.add_argument( "timeout", type = int,
                         help = "Timeout in seconds for each trial." )

    args = parser.parse_args()

    run_benchmark( args.gatesize, args.length, args.numqubits, args.numcircs, args.timeout )

