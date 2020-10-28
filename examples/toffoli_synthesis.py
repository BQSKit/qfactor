"""Optimize a 3-qubit circuit to be a toffoli gate."""

import numpy as np
from scipy.stats import unitary_group

from qfactor import Gate, optimize


# The next two lines start qfactor's logger.
import logging
logging.getLogger( "qfactor" ).setLevel( logging.DEBUG )

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
# and an initial guess for the gate's unitaries.
# Here we use randomly generated unitaries for initial guess.
circuit = [ Gate( unitary_group.rvs(4), (1, 2) ),
            Gate( unitary_group.rvs(4), (0, 2) ),
            Gate( unitary_group.rvs(4), (1, 2) ),
            Gate( unitary_group.rvs(4), (0, 2) ),
            Gate( unitary_group.rvs(4), (0, 1) ) ]

# Note: the Gate object also has an optional boolean parameter "fixed"
# If "fixed" is set to true, that gate's unitary will not change.

# Call the optimize function
ans = optimize( circuit, toffoli, # <--- These are the only required args
                diff_tol = 1e-12,     # Stopping criteria for distance change
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

