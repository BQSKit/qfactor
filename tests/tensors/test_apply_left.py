import numpy as np
import unittest as ut

from scipy.stats import unitary_group

from qfactor.gate import Gate
from qfactor.tensors import CircuitTensor


class TestApplyLeft ( ut.TestCase ):

    def test_apply_left ( self ):
        u1 = unitary_group.rvs( 8 )
        u2 = unitary_group.rvs( 4 ) 
        g = Gate( u2, (0, 1) )
        ct = CircuitTensor( u1, [] )

        ct.apply_left( g )
        prod = u1 @ np.kron( u2, np.identity( 2 ) )
        prod_test = ct.utry
        self.assertTrue( np.allclose( prod, prod_test ) )

        ct.apply_left( g )
        prod = prod @ np.kron( u2, np.identity( 2 ) )
        prod_test = ct.utry
        self.assertTrue( np.allclose( prod, prod_test ) )

    def test_apply_left_invalid ( self ):
        u1 = unitary_group.rvs( 8 )
        ct = CircuitTensor( u1, [] )
        self.assertRaises( TypeError, ct.apply_left, "a" )


if __name__ == "__main__":
    ut.main()

