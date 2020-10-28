import numpy    as np
import unittest as ut

from scipy.stats import unitary_group

from qfactor.utils import is_matrix

class TestIsMatrix ( ut.TestCase ):
    
    def test_is_matrix1 ( self ):
        for i in range( 1, 10 ):
            self.assertTrue( is_matrix( unitary_group.rvs( 2 * i ) ) )
   
    def test_is_matrix2 ( self ):
        self.assertTrue( is_matrix( 1j * np.ones( ( 4, 4 ) ) ) )
        self.assertTrue( is_matrix( np.ones( ( 4, 3 ) ) ) )

    def test_is_matrix_invalid ( self ):
        self.assertFalse( is_matrix( np.ones( ( 4, ) ) ) )
        self.assertFalse( is_matrix( 1 ) )
        self.assertFalse( is_matrix( "a" ) )


if __name__ == '__main__':
    ut.main()
