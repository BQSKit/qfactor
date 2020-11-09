import numpy as np
import unittest as ut

from scipy.stats import unitary_group

from qfactor.gates import Gate


class TestGetInverse ( ut.TestCase ):

    def test_get_inverse ( self ):
        utry = unitary_group.rvs( 8 )
        gate = Gate( utry, (0, 1, 2) )
        inv_gate = gate.get_inverse()
        self.assertTrue( np.allclose( inv_gate.utry @ gate.utry,
                                      np.identity( 8 ) ) )
        self.assertTrue( np.allclose( gate.utry @ inv_gate.utry,
                                      np.identity( 8 ) ) )


if __name__ == "__main__":
    ut.main()
