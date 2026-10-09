"""EPDE wrapper for the grouped-configuration interface of EPDE.

The wrapper in ``run.py`` builds the search with the former keyword interface
(``EpdeSearch(use_pic=..., boundary=..., coordinate_tensors=...)`` and
``fit(variable_names=..., derivs=...)``). The current EPDE rejects ``use_pic``
and expects domains, trajectories and one configuration object. This module
keeps the benchmark contract of ``run.py`` -- the same ``run_epde`` signature,
the same per-dataset settings from ``data/config.py`` and the same result
dictionary -- and only changes how the search is built:

* settings go into ``EpdeSearch(config=...)`` (``use_pic`` -> ``second_objective``);
* data enter through ``createDomain`` / ``createTrajectory``;
* in the fixed protocol the shared derivatives are passed in EPDE's axis-major
  layout (pure derivatives of every axis, orders 1..k), not the full multi-index
  product, which for PDEs also contains mixed derivatives;
* explicit benchmark metadata declares exactly one trigonometric frequency.
  Broad search intervals remain intervals; no width heuristic is applied.

Usage mirrors ``run.py``: ``python epde/run_config_api.py ode_data.npy burgers_data.mat``.
"""

import json
import sys
import time
import subprocess
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.epde_runtime import configure_epde
EPDE_CHECKOUT = configure_epde()
_revision = subprocess.run(["git", "-C", str(EPDE_CHECKOUT), "rev-parse", "HEAD"], capture_output=True, text=True)
EPDE_REVISION = _revision.stdout.strip() if _revision.returncode == 0 else None
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epde import EpdeSearch
from epde.evaluators import trigonometric_evaluator
from epde.interface.prepared_tokens import PreparedTokens, TrigonometricTokens
from epde.interface.token_family import TokenFamily

from data.config import COMMON_PARAMS, NATIVE_EPDE_DEFAULTS, epde_params
from utils.derivatives import compute_multi_derivative, get_data_axes
from utils.protocols import FIXED_PROTOCOL, NATIVE_PROTOCOL, validate_protocol

import importlib.util
_helper_spec = importlib.util.spec_from_file_location("epde_legacy_helpers", Path(__file__).with_name("run.py"))
legacy_wrapper = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(legacy_wrapper)
from utils.randomness import seed_everything
DATASETS = list(epde_params)

# One admissible frequency per dataset and protocol.
# Protocol metadata declares ONE frequency; interval width is not evidence of it.
FIXED_TRIG_FREQUENCIES = {name: 2.0 for name in epde_params}
FIXED_TRIG_FREQUENCIES["ODE_simple_discovery"] = 1.0
NATIVE_TRIG_FREQUENCIES = {name: 1.0 for name in epde_params}
NATIVE_TRIG_FREQUENCIES["ode_data.npy"] = 2.0


def axis_major_derivatives(data, x, y, z, t, max_deriv_order):
    """Shared numerical derivatives in the column layout EPDE expects:
    for every axis in order, its pure derivatives of orders 1..k."""
    arrays = data if isinstance(data, list) else [data]
    arrays = [np.asarray(values, dtype=float) for values in arrays]
    axes = get_data_axes(arrays[0], x, y, z, t)
    if len(max_deriv_order) != len(axes):
        raise ValueError(f"expected {len(axes)} derivative orders, got {max_deriv_order}")
    stacks = []
    for values in arrays:
        columns = []
        for axis, max_order in enumerate(max_deriv_order):
            for order in range(1, int(max_order) + 1):
                orders = tuple(order if i == axis else 0 for i in range(len(axes)))
                columns.append(compute_multi_derivative(values, axes, orders).ravel())
        stacks.append(np.column_stack(columns))
    return stacks


def fixed_frequency_tokens(freq, dimensionality):
    """sin/cos with exactly one frequency.

    EPDE treats trigonometric tokens whose frequencies differ by more than 5 % of
    the declared interval width as different tokens. On an interval such as
    (2 - 1e-8, 2 + 1e-8) one function becomes many numerically identical tokens,
    and the search assembles identities like a*u'sin(2t) + b*u'sin(2t) = u'sin(2t)
    with zero error and zero instability. A single admissible value keeps one
    token and the same printed form ``sin{power: 1.0, freq: 2.0, dim: 0.0}``.
    """
    class FixedTrigonometricTokens(PreparedTokens):
        def __init__(self):
            self._token_family = TokenFamily(token_type='trigonometric')
            self._token_family.set_status(unique_specific_token=True, unique_token_type=True,
                                          meaningful=False)
            params = OrderedDict([('power', (1, 1)), ('freq', (float(freq), float(freq))),
                                  ('dim', (0, dimensionality))])
            self._token_family.set_params(['sin', 'cos'], params, {'power': 0, 'freq': 1e-3, 'dim': 0})
            self._token_family.set_evaluator(trigonometric_evaluator)
    return FixedTrigonometricTokens()


def build_tokens(additional_tokens, grid, data, trig_tokens_freq, fix_frequency=True):
    if trig_tokens_freq is not None and fix_frequency and trig_tokens_freq[0] == trig_tokens_freq[1]:
        # TrigonometricTokens validates a strictly increasing interval. Only use
        # it to identify the configured families, then replace before fitting.
        frequency = float(trig_tokens_freq[0])
        factory_interval = (frequency - 0.1, frequency + 0.1)
    else:
        return legacy_wrapper.get_additional_tokens(additional_tokens, grid, data, trig_tokens_freq)
    tokens = legacy_wrapper.get_additional_tokens(additional_tokens, grid, data, factory_interval)
    low, high = trig_tokens_freq
    out = []
    for family in tokens:
        if isinstance(family, TrigonometricTokens):
            dimensionality = np.asarray(data[0]).ndim - 1
            out.append(fixed_frequency_tokens(0.5 * (low + high), dimensionality))
        else:
            out.append(family)
    return out


def resolve_params(filename, protocol, native_options):
    configured = epde_params[filename]
    if protocol != NATIVE_PROTOCOL:
        frequency = FIXED_TRIG_FREQUENCIES[filename]
        return {**configured, "trig_tokens_freq": (frequency, frequency)}
    params = {
        **NATIVE_EPDE_DEFAULTS,
        **(native_options or {}),
        "variable_names": configured["variable_names"],
        "max_deriv_order": configured.get("max_deriv_order", COMMON_PARAMS["max_deriv_order"]),
        "coordinate_tensors": configured.get("coordinate_tensors"),
        "additional_tokens": (None if len(configured["variable_names"]) > 1
                              else ["GridTokens", "TrigonometricTokens"]),
        "trig_tokens_freq": (NATIVE_TRIG_FREQUENCIES[filename], NATIVE_TRIG_FREQUENCIES[filename]),
        "use_solver": False,
        "fourier_layers": False,
    }
    if native_options and native_options.get("max_iterations") is not None:
        params["training_epochs"] = int(native_options["max_iterations"])
    return params


def search_config(params, protocol, bounds, max_deriv_order, device):
    second = 'instability' if params.get("use_pic", True) else 'complexity'
    objectives = {
        'multiobjective_mode': params.get("multiobjective_mode", True),
        'discrepancy_metric': params.get("discrepancy_metric", "wape"),
        'second_objective': second,
    }
    if protocol == FIXED_PROTOCOL:
        objectives['sparsity_cls'] = 'vwsr'
    return {
        'domain': {'boundary_width': list(bounds) if isinstance(bounds, tuple) else bounds},
        'preprocessing': {'default_preprocessor_type': params["default_preprocessor_type"],
                          'preprocessor_kwargs': {},
                          'max_deriv_order': list(max_deriv_order)},
        'search_space': {'data_fun_pow': params.get("data_fun_pow", COMMON_PARAMS["data_fun_pow"]),
                         'deriv_fun_pow': params.get("deriv_fun_pow", 1),
                         'equation_terms_max_number': params["equation_terms_max_number"],
                         'equation_factors_max_number': params["equation_factors_max_number"]},
        'objectives': objectives,
        'solver': {'use_solver': params["use_solver"], 'device': device},
        'evolution': {'population_size': params["population_size"],
                      'training_epochs': params["training_epochs"]},
        'runtime': {'verbose_params': {'show_iter_idx': False}, 'memory_for_cache': 2},
    }


def run_epde(data, x, y, z, t, filename, device="cpu", solution_index=0, only_print=True,
             visualize=False, return_all=False, protocol=FIXED_PROTOCOL, native_options=None,
             fix_frequency=True):
    """Same contract as ``run.run_epde``; builds the search with the configuration interface."""
    start = time.perf_counter()
    validate_protocol(protocol)
    if native_options and "seed" in native_options:
        seed_everything(int(native_options["seed"]))
    params = resolve_params(filename, protocol, native_options)
    epde_data = data if isinstance(data, list) else [data]
    variable_names = params["variable_names"]

    grid = legacy_wrapper.get_coordinate_tensors(params.get("coordinate_tensors"), t, x, y, z)
    grid = list(grid) if isinstance(grid, (list, tuple)) else [grid]
    bounds = legacy_wrapper.resolve_boundary(params["boundary"], grid)
    max_deriv_order = legacy_wrapper.normalize_epde_max_deriv_order(
        params.get("max_deriv_order", COMMON_PARAMS["max_deriv_order"]), epde_data[0])

    derivs = params.get("derivs")
    if protocol == FIXED_PROTOCOL and derivs is None:
        derivs = axis_major_derivatives(epde_data, x, y, z, t, max_deriv_order)

    config = search_config(params, protocol, bounds, max_deriv_order, device)
    search = EpdeSearch(config=config)
    _, domain = search.createDomain(grid, ID=0)
    _, trajectory = search.createTrajectory(dict(zip(variable_names, epde_data)), domain,
                                            cache_id=0, derivs=derivs)
    tokens = build_tokens(params.get("additional_tokens"), grid, epde_data,
                          params.get("trig_tokens_freq"), fix_frequency)
    fit_kwargs = {}
    if config['objectives'].get('sparsity_cls') == 'lasso':
        fit_kwargs['eq_sparsity_interval'] = params["eq_sparsity_interval"]
    search.fit(data=[trajectory], additional_tokens=tokens, **fit_kwargs)

    if only_print:
        search.equations(only_print=True, num=1)
    elapsed = time.perf_counter() - start
    if visualize:
        search.visualize_solutions()

    def finish(result):
        result["protocol"] = protocol
        result["epde_checkout"] = str(EPDE_CHECKOUT)
        result["epde_revision"] = EPDE_REVISION
        result["search_config"] = config
        result["trig_frequency"] = params.get("trig_tokens_freq")
        result["derivative_source"] = "shared_axis_major" if protocol == FIXED_PROTOCOL else "epde_preprocessor"
        result["algorithm_seed"] = (native_options or {}).get("seed", None)
        if protocol == NATIVE_PROTOCOL:
            result["library_sizes"] = {target: "" for target in result["targets"]}
            result["library_size"] = ""
        return result

    if return_all:
        candidates = [finish(legacy_wrapper.extract_result(solution, filename, elapsed))
                      for solution in legacy_wrapper.pareto_solutions(search)]
        return {"dataset": filename.split(".")[0], "candidates": candidates, "time": elapsed,
                "model": "\n\n".join(c.get("model", "") for c in candidates), "protocol": protocol}
    solution = legacy_wrapper.select_solution(search, solution_index)
    return finish(legacy_wrapper.extract_result(solution, filename, elapsed))


if __name__ == "__main__":
    from utils.dataloader import load_data
    selected = sys.argv[1:] or list(epde_params)
    results = []
    for dataset in selected:
        print(f"\n=== {dataset} ===")
        data, x, y, z, t = load_data(dataset)
        results.append(run_epde(data, x, y, z, t, dataset, device="cpu"))
    legacy_wrapper.save_combined_results(results)
