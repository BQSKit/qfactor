import numpy    as np
import unittest as ut

from qfactor.gates import RzGate


class TestRzGateUpdate ( ut.TestCase ):

    def test_rzgate_update_1 ( self ):
        env = RzGate( np.pi/3, 0 ).utry
        gate = RzGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, -np.pi/3 ) )

    def test_rzgate_update_2 ( self ):
        env = RzGate( 2*np.pi/3, 0 ).utry
        gate = RzGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, -2*np.pi/3 ) )

    def test_rzgate_update_3 ( self ):
        env = RzGate( -2*np.pi/3, 0 ).utry
        gate = RzGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, 2*np.pi/3 ) )

    def test_rzgate_update_4 ( self ):
        env = RzGate( -np.pi/3, 0 ).utry
        gate = RzGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, np.pi/3 ) )


if __name__ == '__main__':
    ut.main()

