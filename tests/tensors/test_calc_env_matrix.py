import numpy as np
import unittest as ut

from scipy.stats import unitary_group

from qfactor.gates import Gate
from qfactor.tensors import CircuitTensor


class TestCalcEnvMatrix ( ut.TestCase ):

    def test_calc_env_matrix ( self ):
        u1 = unitary_group.rvs( 8 )
        u2 = u1.conj().T

        ct = CircuitTensor( u1, [] )

        env = ct.calc_env_matrix( [0, 1, 2] )
        self.assertTrue( np.allclose( env, u2  ) )


    def test_calc_env_matrix_invalid ( self ):
        u1 = unitary_group.rvs( 8 )
        u2 = u1.conj().T

        ct = CircuitTensor( u1, [] )

        self.assertRaises( ValueError, ct.calc_env_matrix, [0, 1, 2, 3] )
        self.assertRaises( TypeError, ct.calc_env_matrix, "a" )


if __name__ == "__main__":
    ut.main()

