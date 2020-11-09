import numpy as np
import unittest as ut

from scipy.stats import unitary_group

from qfactor.gates import Gate


class TestGetTensorFormat ( ut.TestCase ):

    def test_get_tensor_format ( self ):
        utry = unitary_group.rvs( 8 )
        gate = Gate( utry, (0, 1, 2) )
        tensor = gate.get_tensor_format()
        self.assertTrue( len( tensor.shape ) == 6 )
        self.assertTrue( all( [ x == 2 for x in tensor.shape ] ) )
        self.assertTrue( np.allclose( np.reshape( tensor, (8, 8) ), utry ) )

    def test_get_tensor_format_left ( self ):
        utry = unitary_group.rvs( 8 )
        gate = Gate( utry, (0, 1, 2) )
        tensor = gate.get_tensor_format( compress_left = True )
        self.assertTrue( len( tensor.shape ) == 4 )
        self.assertTrue( all( [ x == 2 for x in tensor.shape[1:] ] ) )
        self.assertTrue( tensor.shape[0] == 8 )
        self.assertTrue( np.allclose( np.reshape( tensor, (8, 8) ), utry ) )

    def test_get_tensor_format_right ( self ):
        utry = unitary_group.rvs( 8 )
        gate = Gate( utry, (0, 1, 2) )
        tensor = gate.get_tensor_format( compress_right = True )
        self.assertTrue( len( tensor.shape ) == 4 )
        self.assertTrue( all( [ x == 2 for x in tensor.shape[:-1] ] ) )
        self.assertTrue( tensor.shape[-1] == 8 )
        self.assertTrue( np.allclose( np.reshape( tensor, (8, 8) ), utry ) )

    def test_get_tensor_format_all ( self ):
        utry = unitary_group.rvs( 8 )
        gate = Gate( utry, (0, 1, 2) )
        tensor = gate.get_tensor_format( compress_left = True,
                                         compress_right = True )
        self.assertTrue( len( tensor.shape ) == 2 )
        self.assertTrue( tensor.shape[0] == 8 )
        self.assertTrue( tensor.shape[1] == 8 )
        self.assertTrue( np.allclose( tensor, utry ) )


if __name__ == "__main__":
    ut.main()
