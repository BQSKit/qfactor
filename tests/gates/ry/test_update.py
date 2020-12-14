import numpy    as np
import unittest as ut

from qfactor.gates import RyGate


class TestRyGateUpdate ( ut.TestCase ):

    def test_rygate_update_1 ( self ):
        env = RyGate( np.pi/3, 0 ).utry
        gate = RyGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, -np.pi/3 ) )

    def test_rygate_update_2 ( self ):
        env = RyGate( 2*np.pi/3, 0 ).utry
        gate = RyGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, -2*np.pi/3 ) )

    def test_rygate_update_3 ( self ):
        env = RyGate( -2*np.pi/3, 0 ).utry
        gate = RyGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, 2*np.pi/3 ) )

    def test_rygate_update_4 ( self ):
        env = RyGate( -np.pi/3, 0 ).utry
        gate = RyGate( 0., 0 )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, np.pi/3 ) )


if __name__ == '__main__':
    ut.main()

