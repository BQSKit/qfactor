import numpy    as np
import scipy    as sp
import unittest as ut

from scipy.stats import unitary_group

from qfactor.utils import is_unitary

class TestIsUnitary ( ut.TestCase ):
    
    def test_is_unitary1 ( self ):
        for i in range( 1, 10 ):
            U = unitary_group.rvs( 2 * i )
            self.assertTrue( is_unitary( U, tol = 1e-14 ) )

    def test_is_unitary2 ( self ):
        for i in range( 1, 10 ):
            U = unitary_group.rvs( 2 * i )
            U += 1e-13 * np.ones( ( 2 * i, 2 * i ) )
            self.assertTrue( is_unitary( U, tol = 1e-12 ) )
   
    def test_is_unitary_invalid ( self ):
        self.assertFalse( is_unitary( 1j * np.ones( ( 4, 4 ) ) ) )
        self.assertFalse( is_unitary( np.ones( ( 4, 3 ) ) ) )
        self.assertFalse( is_unitary( np.ones( ( 4, ) ) ) )
        self.assertFalse( is_unitary( 1 ) )
        self.assertFalse( is_unitary( "a" ) )


if __name__ == '__main__':
    ut.main()
