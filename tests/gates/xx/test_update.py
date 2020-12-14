import numpy    as np
import unittest as ut

from qfactor.gates import XXGate


class TestXXGateUpdate ( ut.TestCase ):

    def test_xxgate_update_1 ( self ):
        env = XXGate( np.pi/3, (0,1) ).utry
        gate = XXGate( 0., (0,1) )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, -np.pi/3 ) )

    def test_xxgate_update_2 ( self ):
        env = XXGate( 2*np.pi/3, (0,1) ).utry
        gate = XXGate( 0., (0,1) )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, -2*np.pi/3 ) )

    def test_xxgate_update_3 ( self ):
        env = XXGate( -2*np.pi/3, (0,1) ).utry
        gate = XXGate( 0., (0,1) )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, 2*np.pi/3 ) )

    def test_xxgate_update_4 ( self ):
        env = XXGate( -np.pi/3, (0,1) ).utry
        gate = XXGate( 0., (0,1) )
        gate.update( env, 0 )
        self.assertTrue( np.allclose( gate.theta, np.pi/3 ) )


if __name__ == '__main__':
    ut.main()

