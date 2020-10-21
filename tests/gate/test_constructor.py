import numpy    as np
import unittest as ut

from csvdopt.gate import Gate

class TestGateConstructor ( ut.TestCase ):

    TOFFOLI = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    INVALID = np.asarray(
            [[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
            [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]] )

    def test_gate_constructor_invalid ( self ):
        self.assertRaises( TypeError, Gate, 1, (0, 1) )
        self.assertRaises( TypeError, Gate, np.array( [ 0, 1 ] ), (0, 1) )
        self.assertRaises( TypeError, Gate, np.array( [ [ [ 0 ] ] ] ), (0, 1) )

        self.assertRaises( TypeError, Gate, self.TOFFOLI, 1 )
        self.assertRaises( TypeError, Gate, self.TOFFOLI, ("a", "b") )
        self.assertRaises( TypeError, Gate, self.TOFFOLI, (1, 1) )

        self.assertRaises( ValueError, Gate, self.TOFFOLI, (0, 1) )
        self.assertRaises( ValueError, Gate, self.TOFFOLI, (0, 1, 2, 3) )

        self.assertRaises( TypeError, Gate, self.TOFFOLI, (0, 1, 2), "a" )

        invalid_utry_matrix = np.copy( self.TOFFOLI )
        invalid_utry_matrix[4][4] = 2112.+0.j

        self.assertRaises( TypeError, Gate, invalid_utry_matrix, (0, 1, 2) )
        self.assertRaises( TypeError, Gate, self.INVALID, (0, 1, 2) )

    def test_gate_constructor_valid ( self ):
        gate = Gate( self.TOFFOLI, (0, 1, 2), True )
        self.assertTrue( np.array_equal( gate.utry, self.TOFFOLI ) )
        self.assertTrue( np.array_equal( gate.location, (0, 1, 2) ) )
        self.assertEqual( gate.gate_size, 3 )
        self.assertTrue( gate.fixed )


if __name__ == '__main__':
    ut.main()

