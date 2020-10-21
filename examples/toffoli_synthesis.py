"""Optimize a random 3-qubit circuit to be a toffoli gate."""

import numpy as np

from csvdopt import Gate, optimize


def gen_random_gate( gate_size, num_qubits ):
    num_elems = 2 ** gate_size
    X = np.random.random( (num_elems, num_elems) )
    Y = np.random.random( (num_elems, num_elems) )
    utry, _, _ = np.linalg.svd( X - 0.5 + 1j * (Y - 0.5) )
    location = sorted( np.random.choice( range( num_qubits ),
                                         gate_size,
                                         replace = False ) )
    location = tuple( [ int( x ) for x in location ] )
    return Gate( utry, location )

utry_target = np.array( [ [ 1, 0, 0, 0, 0, 0, 0, 0 ],
                          [ 0, 1, 0, 0, 0, 0, 0, 0 ],
                          [ 0, 0, 1, 0, 0, 0, 0, 0 ],
                          [ 0, 0, 0, 1, 0, 0, 0, 0 ],
                          [ 0, 0, 0, 0, 1, 0, 0, 0 ],
                          [ 0, 0, 0, 0, 0, 1, 0, 0 ],
                          [ 0, 0, 0, 0, 0, 0, 0, 1 ],
                          [ 0, 0, 0, 0, 0, 0, 1, 0 ] ] )

circuit = [ gen_random_gate( 2, 3 ) for i in range( 7 ) ]

print( optimize( circuit, utry_target, 1e-12, 100000, 0 ) )

