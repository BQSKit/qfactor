import numpy    as np
import unittest as ut

from qfactor.gates import XXGate


class TestXXGateRepr ( ut.TestCase ):

    def test_xxgate_repr_1 ( self ):
        gate = XXGate( 0., (0,1) )
        self.assertEqual( repr( gate ), "(0, 1): XX(0.0)" )

    def test_xxgate_repr_2 ( self ):
        gate = XXGate( 2., (1,3) )
        self.assertEqual( repr( gate ), "(1, 3): XX(2.0)" )


if __name__ == '__main__':
    ut.main()

