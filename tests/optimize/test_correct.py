import numpy as np
import unittest as ut

from scipy.stats import unitary_group

from csvdopt.optimize import optimize, Gate


class TestOptimizeFixed ( ut.TestCase ):

    def test_optimize_fixed ( self ):
        u1 = unitary_group.rvs( 8 )
        g1 = Gate( u1, (0, 1, 2) )
        circ = optimize( [ g1 ], u1 )

        self.assertTrue( np.allclose( circ[0].utry, g1.utry ) )


if __name__ == "__main__":
    ut.main()

