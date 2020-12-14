import scipy
import numpy    as np
import unittest as ut

from qfactor import get_distance
from qfactor.gates import RyGate

class TestRyGateConstructor ( ut.TestCase ):

    def test_rygate_constructor_invalid ( self ):
        self.assertRaises( TypeError, RyGate, 1, 0 )
        self.assertRaises( TypeError, RyGate, "a", 0 )
        self.assertRaises( TypeError, RyGate, [0, 1], 0 )

        self.assertRaises( ValueError, RyGate, np.pi/2, -1 )
        self.assertRaises( TypeError, RyGate, np.pi/2, [0, 1] )
        self.assertRaises( TypeError, RyGate, np.pi/2, (0, 1) )
        self.assertRaises( TypeError, RyGate, np.pi/2, ("a") )

        self.assertRaises( TypeError, RyGate, np.pi/2, 0, 0 )

    def test_rygate_constructor_valid ( self ):
        gate = RyGate( np.pi, 0, True )
        Y = np.array( [ [ 0, -1j ], [ 1j, 0 ] ] )
        Ry = scipy.linalg.expm( -1j * np.pi/2 * Y )
        self.assertTrue( get_distance( [ gate ], Ry ) < 1e-15 )
        self.assertTrue( np.array_equal( gate.location, (0,) ) )
        self.assertEqual( gate.gate_size, 1 )
        self.assertTrue( gate.fixed )


if __name__ == '__main__':
    ut.main()

