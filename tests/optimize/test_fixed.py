import numpy as np
import unittest as ut

from scipy.stats import unitary_group

from csvdopt.optimize import optimize, Gate


class TestOptimizeFixed ( ut.TestCase ):

    def test_optimize_fixed ( self ):
        g1 = Gate( unitary_group.rvs( 4 ), (0, 1), True )
        g2 = Gate( unitary_group.rvs( 8 ), (1, 2, 3), True )
        circ = optimize( [ g1, g2 ], unitary_group.rvs( 16 ) )

        self.assertTrue( np.allclose( circ[0].utry, g1.utry ) )
        self.assertTrue( np.allclose( circ[1].utry, g2.utry ) )


if __name__ == "__main__":
    ut.main()

