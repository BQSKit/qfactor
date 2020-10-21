import numpy    as np
import unittest as ut

from scipy.stats import unitary_group

from csvdopt.utils import is_square_matrix

class TestIsSquareMatrix ( ut.TestCase ):
    
    def test_is_square_matrix1 ( self ):
        for i in range( 10 ):
            self.assertTrue( is_square_matrix( unitary_group.rvs( 2 * i ) ) )
   
    def test_is_square_matrix2 ( self ):
        self.assertTrue( is_square_matrix( 1j * np.ones( ( 4, 4 ) ) ) )

    def test_is_square_matrix_invalid ( self ):
        self.assertFalse( is_square_matrix( np.ones( ( 4, 3 ) ) ) )
        self.assertFalse( is_square_matrix( np.ones( ( 4, ) ) ) )
        self.assertFalse( is_square_matrix( 1 ) )
        self.assertFalse( is_square_matrix( "a" ) )


if __name__ == '__main__':
    ut.main()
