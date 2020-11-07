"""
Optimize a 3-qubit circuit to be a toffoli gate.

In this example, we use a more native structure and extract
angles for RzGates from the optimized result.
"""

import numpy as np

from qfactor import Gate, optimize, CnotGate, RzGate


# The next two lines start qfactor's logger.
import logging
logging.getLogger( "qfactor" ).setLevel( logging.INFO )

# We will optimize towards the toffoli unitary.
toffoli = np.array( [ [ 1, 0, 0, 0, 0, 0, 0, 0 ],
                      [ 0, 1, 0, 0, 0, 0, 0, 0 ],
                      [ 0, 0, 1, 0, 0, 0, 0, 0 ],
                      [ 0, 0, 0, 1, 0, 0, 0, 0 ],
                      [ 0, 0, 0, 0, 1, 0, 0, 0 ],
                      [ 0, 0, 0, 0, 0, 1, 0, 0 ],
                      [ 0, 0, 0, 0, 0, 0, 0, 1 ],
                      [ 0, 0, 0, 0, 0, 0, 1, 0 ] ] )

# Start with the circuit structure
# and an initial guess for the gate's parameters.
# Here, the hadamards and cnots are fixed
# and the RzGates start with theta equal to a random number.

H = np.sqrt(1/2) * np.array( [ [ 1, 1 ],
                               [ 1, -1 ] ] )

circuit = [ Gate( H, (2,), fixed = True ),
            CnotGate( 1, 2 ),
            RzGate( np.random.random(), 2 ),
            CnotGate( 0, 2 ),
            RzGate( np.random.random(), 2 ),
            CnotGate( 1, 2 ),
            RzGate( np.random.random(), 2 ),
            CnotGate( 0, 2 ),
            RzGate( np.random.random(), 1 ),
            RzGate( np.random.random(), 2 ),
            CnotGate( 0, 1 ),
            Gate( H, (2,), fixed = True ),
            RzGate( np.random.random(), 0 ),
            RzGate( np.random.random(), 1 ),
            CnotGate( 0, 1 ) ]

# Call the optimize function
ans = optimize( circuit, toffoli, # <--- These are the only required args
                diff_tol_a = 1e-12,   # Stopping criteria for distance change
                diff_tol_r = 1e-6,    # Relative criteria for distance change
                dist_tol = 1e-12,     # Stopping criteria for distance
                max_iters = 100000,   # Maximum number of iterations
                min_iters = 1000,     # Minimum number of iterations
                slowdown_factor = 0 ) # Larger numbers slowdown optimization
                                      # to avoid local minima


# The result "ans" is another circuit object (list[Gate])
# with the gate's unitaries changed from the input circuit.
print( ans )

# If you would like to convert the unitary operations to native gates,
# you should use the KAK decomposition for 2 qubit unitaries, or
# qsearch or qfast for 3+ qubit unitaries.

