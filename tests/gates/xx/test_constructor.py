import scipy
import numpy    as np
import unittest as ut

from qfactor import get_distance
from qfactor.gates import XXGate

class TestXXGateConstructor ( ut.TestCase ):

    def test_xxgate_constructor_invalid ( self ):
        self.assertRaises( TypeError, XXGate, 1, 0 )
        self.assertRaises( TypeError, XXGate, "a", 0 )
        self.assertRaises( TypeError, XXGate, [0, 1], 0 )

        self.assertRaises( TypeError, XXGate, np.pi/2, -1 )
        self.assertRaises( TypeError, XXGate, np.pi/2, [0, 1] )
        self.assertRaises( TypeError, XXGate, np.pi/2, (0, 1, 2) )
        self.assertRaises( TypeError, XXGate, np.pi/2, ("a") )

        self.assertRaises( TypeError, XXGate, np.pi/2, 0, 0 )

    def test_xxgate_constructor_valid ( self ):
        gate = XXGate( np.pi, (0, 1), True )
        X = np.array( [ [ 0, 1 ], [ 1, 0 ] ] )
        XX = np.kron( X, X )
        RXX = scipy.linalg.expm( -1j * np.pi/2 * XX )
        self.assertTrue( get_distance( [ gate ], RXX ) < 1e-15 )
        self.assertTrue( np.array_equal( gate.location, (0,1) ) )
        self.assertEqual( gate.gate_size, 2 )
        self.assertTrue( gate.fixed )


if __name__ == '__main__':
    ut.main()

