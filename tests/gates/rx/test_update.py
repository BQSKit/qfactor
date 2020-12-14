import numpy    as np
import unittest as ut

from qfactor.gates import RxGate


class TestRxGateUpdate ( ut.TestCase ):

    def test_rxgate_update_1 ( self ):
        env = RxGate( np.pi/3, 0 ).utry
        gate = RxGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, -np.pi/3 ) )

    def test_rxgate_update_2 ( self ):
        env = RxGate( 2*np.pi/3, 0 ).utry
        gate = RxGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, -2*np.pi/3 ) )

    def test_rxgate_update_3 ( self ):
        env = RxGate( -2*np.pi/3, 0 ).utry
        gate = RxGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, 2*np.pi/3 ) )

    def test_rxgate_update_4 ( self ):
        env = RxGate( -np.pi/3, 0 ).utry
        gate = RxGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, np.pi/3 ) )


if __name__ == '__main__':
    ut.main()

