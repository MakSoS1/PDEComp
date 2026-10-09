"""Integration regressions; run with EPDE_PATH pointing at a modern checkout."""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get('EPDE_PATH'):
    sys.path.insert(0, os.environ['EPDE_PATH'])
import clean_run_metrics as driver
from utils.protocols import NATIVE_PROTOCOL, FIXED_PROTOCOL


class ModernEpdeTests(unittest.TestCase):
    def test_driver_uses_grouped_configuration_adapter(self):
        module = driver.load_framework_module('epde')
        self.assertTrue(hasattr(module, 'search_config'), 'legacy keyword wrapper is still selected')

    def test_truth_metrics_work_without_removed_thesis_workspace(self):
        equation = '-4.0 * u{power: 1.0} + -1.0 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + 1.5 * x{power: 1.0, dim: 0.0} = d^2u/dx0^2{power: 1.0}'
        result = driver.epde_structural_metrics('ode_data.npy', [equation])
        self.assertEqual(result['hamming'], 0)
        self.assertAlmostEqual(result['coefficient_error'], 0.0)

    def test_native_oscillator_represents_sin_two_t_as_one_frequency(self):
        module = driver.load_module('modern_test_adapter', ROOT / 'epde/run_config_api.py')
        params = module.resolve_params('ode_data.npy', NATIVE_PROTOCOL, None)
        self.assertEqual(params['trig_tokens_freq'], (2.0, 2.0))

    def test_single_frequency_family_represents_exact_oscillator_forcing(self):
        import numpy as np
        module = driver.load_module('frequency_test_adapter', ROOT / 'epde/run_config_api.py')
        t = np.linspace(0, 2, 30)
        params = module.resolve_params('ode_data.npy', NATIVE_PROTOCOL, None)
        families = module.build_tokens(['TrigonometricTokens'], [t], [t], params['trig_tokens_freq'])
        family = families[0].token_family
        self.assertEqual(family.token_params['freq'], (2.0, 2.0))

    def test_fixed_protocol_keeps_shared_axis_major_derivatives(self):
        import numpy as np
        module = driver.load_module('derivatives_test_adapter', ROOT / 'epde/run_config_api.py')
        t = np.linspace(0, 1, 20)
        x = np.linspace(-1, 1, 15)
        data = t[:, None] ** 2 + x[None, :] ** 3
        derivatives = module.axis_major_derivatives(data, x, None, None, t, (1, 2))[0]
        self.assertEqual(derivatives.shape, (300, 3))
        np.testing.assert_allclose(derivatives[:, 0].reshape(20, 15)[2:-2, 2:-2], np.broadcast_to(2*t[:, None], data.shape)[2:-2, 2:-2], atol=1e-10)


if __name__ == '__main__':
    unittest.main()
