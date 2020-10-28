import numpy    as np
import unittest as ut

from qfast.perm import calc_permutation_matrix

from qfactor import Gate, optimize
from qfactor.tensors import CircuitTensor

class TestToffoliTensor ( ut.TestCase ):

    def test_toffoli_tensor ( self ):
        toffoli = np.array( [ [ 1, 0, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 1, 0, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 1, 0, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 1, 0, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 1, 0, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 1, 0, 0 ],
                              [ 0, 0, 0, 0, 0, 0, 0, 1 ],
                              [ 0, 0, 0, 0, 0, 0, 1, 0 ] ] )

        p12 = calc_permutation_matrix( 3, (1, 2) )
        p02 = calc_permutation_matrix( 3, (0, 2) )

        cnot = np.array( [ [ 1, 0, 0, 0 ],
                           [ 0, 1, 0, 0 ],
                           [ 0, 0, 0, 1 ],
                           [ 0, 0, 1, 0 ] ] )

        H = (np.sqrt(2)/2) * np.array( [ [ 1, 1 ],
                                         [ 1, -1 ] ] )

        T = np.array( [ [ 1, 0 ], [ 0, np.exp( 1j * np.pi/4 ) ] ] )
        I = np.identity( 2 )

        u1 = np.kron( I, T.conj().T ) @ cnot @ np.kron( I, H )
        u2 = np.kron( I, T ) @ cnot
        u3 = np.kron( I, T.conj().T ) @ cnot
        u4 = np.kron( I, H @ T ) @ cnot
        u5 = cnot @ np.kron( T, T.conj().T ) @ cnot @ np.kron( I, T )

        circuit = [ Gate( u1, (1, 2) ),
                    Gate( u2, (0, 2) ),
                    Gate( u3, (1, 2) ),
                    Gate( u4, (0, 2) ),
                    Gate( u5, (0, 1) ) ]


        c1 = p12 @ np.kron( u1, I ) @ p12.T
        c2 = p02 @ np.kron( u2, I ) @ p02.T
        c3 = p12 @ np.kron( u3, I ) @ p12.T
        c4 = p02 @ np.kron( u4, I ) @ p02.T
        c5 = np.kron( u5, I )
        self.assertTrue( np.allclose( toffoli, c5 @ c4 @ c3 @ c2 @ c1 ) )


        ct = CircuitTensor( toffoli, [] )
        self.assertTrue( np.allclose( ct.utry, toffoli.conj().T  ) )
        ct.apply_right( circuit[0] )
        self.assertTrue( np.allclose( ct.utry, c1 @ toffoli.conj().T ) )
        ct.apply_right( circuit[1] )
        self.assertTrue( np.allclose( ct.utry, c2 @ c1 @ toffoli.conj().T ) )
        ct.apply_right( circuit[2] )
        self.assertTrue( np.allclose( ct.utry, c3 @ c2 @ c1 @ toffoli.conj().T ) )
        ct.apply_right( circuit[3] )
        self.assertTrue( np.allclose( ct.utry, c4 @ c3 @ c2 @ c1 @ toffoli.conj().T ) )
        ct.apply_right( circuit[4] )
        self.assertTrue( np.allclose( ct.utry, c5 @ c4 @ c3 @ c2 @ c1 @ toffoli.conj().T ) )
        self.assertTrue( np.allclose( ct.utry, np.identity( 8 ) ) )
        ct = CircuitTensor( toffoli, circuit )
        self.assertTrue( np.allclose( ct.utry, np.identity( 8 ) ) )


if __name__ == "__main__":
    ut.main()

