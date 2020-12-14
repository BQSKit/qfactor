import scipy
import numpy    as np
import unittest as ut

from qfactor import get_distance
from qfactor.gates import RxGate

class TestRxGateConstructor ( ut.TestCase ):

    def test_rxgate_constructor_invalid ( self ):
        self.assertRaises( TypeError, RxGate, 1, 0 )
        self.assertRaises( TypeError, RxGate, "a", 0 )
        self.assertRaises( TypeError, RxGate, [0, 1], 0 )

        self.assertRaises( ValueError, RxGate, np.pi/2, -1 )
        self.assertRaises( TypeError, RxGate, np.pi/2, [0, 1] )
        self.assertRaises( TypeError, RxGate, np.pi/2, (0, 1) )
        self.assertRaises( TypeError, RxGate, np.pi/2, ("a") )

        self.assertRaises( TypeError, RxGate, np.pi/2, 0, 0 )

    def test_rxgate_constructor_valid ( self ):
        gate = RxGate( np.pi, 0, True )
        X = np.array( [ [ 0, 1 ], [ 1, 0 ] ] )
        Rx = scipy.linalg.expm( -1j * np.pi/2 * X )
        self.assertTrue( get_distance( [ gate ], Rx ) < 1e-15 )
        self.assertTrue( np.array_equal( gate.location, (0,) ) )
        self.assertEqual( gate.gate_size, 1 )
        self.assertTrue( gate.fixed )


if __name__ == '__main__':
    ut.main()

