import numpy    as np
import unittest as ut

from qfactor.gates import RyGate


class TestRyGateRepr ( ut.TestCase ):

    def test_rygate_repr_1 ( self ):
        gate = RyGate( 0., 0 )
        self.assertEqual( repr( gate ), "(0,): Ry(0.0)" )

    def test_rygate_repr_2 ( self ):
        gate = RyGate( 2., 1 )
        self.assertEqual( repr( gate ), "(1,): Ry(2.0)" )


if __name__ == '__main__':
    ut.main()

