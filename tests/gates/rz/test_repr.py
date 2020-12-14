import numpy    as np
import unittest as ut

from qfactor.gates import RzGate


class TestRzGateRepr ( ut.TestCase ):

    def test_rzgate_repr_1 ( self ):
        gate = RzGate( 0., 0 )
        self.assertEqual( repr( gate ), "(0,): Rz(0.0)" )

    def test_rzgate_repr_2 ( self ):
        gate = RzGate( 2., 1 )
        self.assertEqual( repr( gate ), "(1,): Rz(2.0)" )


if __name__ == '__main__':
    ut.main()

