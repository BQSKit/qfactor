import numpy    as np
import unittest as ut

from qfactor.gates import RxGate


class TestRxGateRepr ( ut.TestCase ):

    def test_rxgate_repr_1 ( self ):
        gate = RxGate( 0., 0 )
        self.assertEqual( repr( gate ), "(0,): Rx(0.0)" )

    def test_rxgate_repr_2 ( self ):
        gate = RxGate( 2., 1 )
        self.assertEqual( repr( gate ), "(1,): Rx(2.0)" )


if __name__ == '__main__':
    ut.main()

