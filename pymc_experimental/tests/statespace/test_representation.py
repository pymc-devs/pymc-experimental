import unittest

import numpy as np
import pytensor.tensor as pt
from numpy.testing import assert_allclose

from pymc_experimental.statespace.core.representation import PytensorRepresentation
from pymc_experimental.tests.statespace.utilities.test_helpers import make_test_inputs


class BasicFunctionality(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(10)[:, None]

    def test_numpy_to_pytensor(self):
        ssm = PytensorRepresentation(data=np.zeros((4, 1)), k_states=5, k_posdef=1)
        X = np.eye(5)
        X_pt = ssm._numpy_to_pytensor("transition", X)
        self.assertTrue(isinstance(X_pt, pt.TensorVariable))

    def test_default_shapes_full_rank(self):
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=5)
        p = ssm.data.shape[1]
        m = ssm.k_states
        r = ssm.k_posdef

        self.assertTrue(ssm.data.shape == (10, 1, 1))
        self.assertTrue(ssm["design"].eval().shape == (p, m))
        self.assertTrue(ssm["transition"].eval().shape == (m, m))
        self.assertTrue(ssm["selection"].eval().shape == (m, r))
        self.assertTrue(ssm["state_cov"].eval().shape == (r, r))
        self.assertTrue(ssm["obs_cov"].eval().shape == (p, p))

    def test_default_shapes_low_rank(self):
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=2)
        p = ssm.data.shape[1]
        m = ssm.k_states
        r = ssm.k_posdef

        self.assertTrue(ssm.data.shape == (10, 1, 1))
        self.assertTrue(ssm["design"].eval().shape == (p, m))
        self.assertTrue(ssm["transition"].eval().shape == (m, m))
        self.assertTrue(ssm["selection"].eval().shape == (m, r))
        self.assertTrue(ssm["state_cov"].eval().shape == (r, r))
        self.assertTrue(ssm["obs_cov"].eval().shape == (p, p))

    def test_matrix_assignment(self):
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=2)

        ssm["design", 0, 0] = 3.0
        ssm["transition", 0, :] = 2.7
        ssm["selection", -1, -1] = 9.9

        self.assertTrue(ssm["design"].eval()[0, 0] == 3.0)
        self.assertTrue(np.all(ssm["transition"].eval()[0, :] == 2.7))
        self.assertTrue(ssm["selection"].eval()[-1, -1] == 9.9)

    def test_build_representation_from_data(self):
        p, m, r, n = 3, 6, 1, 10
        inputs = [data, a0, P0, T, Z, R, H, Q] = make_test_inputs(p, m, r, n, missing_data=0)
        ssm = PytensorRepresentation(
            data=data,
            k_states=m,
            k_posdef=r,
            design=Z,
            transition=T,
            selection=R,
            state_cov=Q,
            obs_cov=H,
            initial_state=a0,
            initial_state_cov=P0,
        )
        names = [
            "initial_state",
            "initial_state_cov",
            "transition",
            "design",
            "selection",
            "obs_cov",
            "state_cov",
        ]
        for name, X in zip(names, inputs[1:]):
            self.assertTrue(np.allclose(X, ssm[name].eval()))

    def test_assign_time_varying_matrices(self):
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=2)
        n = self.data.shape[0]

        ssm["design", 0, 0] = 3.0
        ssm["transition", 0, :] = 2.7
        ssm["selection", -1, -1] = 9.9

        ssm["state_intercept"] = np.zeros((5, 1, self.data.shape[0]))
        ssm["state_intercept", 0, 0, :] = np.arange(n)

        self.assertTrue(ssm["design"].eval()[0, 0] == 3.0)
        self.assertTrue(np.all(ssm["transition"].eval()[0, :] == 2.7))
        self.assertTrue(ssm["selection"].eval()[-1, -1] == 9.9)
        self.assertTrue(np.allclose(ssm["state_intercept"][0, 0, :].eval(), np.arange(n)))

    def test_invalid_key_name_raises(self):
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=1)
        with self.assertRaises(IndexError) as e:
            X = ssm["invalid_key"]
        msg = str(e.exception)
        self.assertEqual(msg, "invalid_key is an invalid state space matrix name")

    def test_non_string_key_raises(self):
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=1)
        with self.assertRaises(IndexError) as e:
            X = ssm[0]
        msg = str(e.exception)
        self.assertEqual(msg, "First index must the name of a valid state space matrix.")

    def test_invalid_key_tuple_raises(self):
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=1)
        with self.assertRaises(IndexError) as e:
            X = ssm[0, 1, 1]
        msg = str(e.exception)
        self.assertEqual(msg, "First index must the name of a valid state space matrix.")

    def test_slice_statespace_matrix(self):
        T = np.eye(5)
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=1, transition=T)
        T_out = ssm["transition", :3, :]
        assert_allclose(T[:3], T_out.eval())

    def test_update_matrix_via_key(self):
        T = np.eye(5)
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=1)
        ssm["transition"] = T

        assert_allclose(T, ssm["transition"].eval())

    def test_update_matrix_with_invalid_shape_raises(self):
        T = np.eye(10)
        ssm = PytensorRepresentation(data=self.data, k_states=5, k_posdef=1)
        with self.assertRaises(ValueError) as e:
            ssm["transition"] = T
        msg = str(e.exception)
        self.assertEqual(msg, "Array provided for transition has shape (10, 10), expected (5, 5)")


if __name__ == "__main__":
    unittest.main()
