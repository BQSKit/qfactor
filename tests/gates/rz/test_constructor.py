import scipy
import numpy    as np
import unittest as ut

from qfactor import get_distance
from qfactor.gates import RzGate

class TestRzGateConstructor ( ut.TestCase ):

    def test_rzgate_constructor_invalid ( self ):
        self.assertRaises( TypeError, RzGate, 1, 0 )
        self.assertRaises( TypeError, RzGate, "a", 0 )
        self.assertRaises( TypeError, RzGate, [0, 1], 0 )

        self.assertRaises( ValueError, RzGate, np.pi/2, -1 )
        self.assertRaises( TypeError, RzGate, np.pi/2, [0, 1] )
        self.assertRaises( TypeError, RzGate, np.pi/2, (0, 1) )
        self.assertRaises( TypeError, RzGate, np.pi/2, ("a") )

        self.assertRaises( TypeError, RzGate, np.pi/2, 0, 0 )

    def test_rzgate_constructor_valid ( self ):
        gate = RzGate( np.pi, 0, True )
        Z = np.array( [ [ 1, 0 ], [ 0, -1 ] ] )
        Rz = scipy.linalg.expm( -1j * np.pi/2 * Z )
        self.assertTrue( get_distance( [ gate ], Rz ) < 1e-15 )
        self.assertTrue( np.array_equal( gate.location, (0,) ) )
        self.assertEqual( gate.gate_size, 1 )
        self.assertTrue( gate.fixed )


if __name__ == '__main__':
    ut.main()

