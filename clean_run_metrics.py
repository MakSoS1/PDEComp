import argparse
import contextlib
import csv
import importlib.util
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from data.config import TRUE_COEFFICIENTS, TRUE_COEFFICIENT_ALTERNATIVES
from utils.dataloader import load_data
from utils.protocols import (
    FIXED_PROTOCOL,
    NATIVE_PROTOCOL,
    PROTOCOLS,
    ProtocolSkippedError,
    ProtocolUnavailableError,
    benchmark_exit_code,
)
from utils.randomness import seed_everything
from utils.epde_runtime import configure_epde


ROOT = Path(__file__).resolve().parent
COEFFICIENT_TOLERANCE = 1e-12
DEFAULT_DATASETS = list(TRUE_COEFFICIENTS)
EPDE_DATASET_REGISTRY = {
    "ode_data.npy": "ode", "vdp_data.npy": "vdp", "lotka_data.npy": "lv",
    "lorenz_data.npy": "lorenz", "burgers_data.mat": "burgers",
    "burgers_sln_100_data.csv": "burgers_inviscid", "ac_data.npy": "ac",
    "kdv_data.mat": "kdv", "kdv_periodic_data.npy": "kdv_cossin",
    "wave_data.csv": "wave", "pde_divide_data.npy": "pde_divide",
    "pde_compound_data.npy": "pde_compound", "ks_data.mat": "ks", "ns_data.mat": "ns",
}
_THESIS_METRICS = None
_EPDE_TRUTH_CACHE = {}


def load_module(module_name, path, extra_paths=()):
    for extra_path in reversed(extra_paths):
        sys.path.insert(0, str(extra_path))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_framework_module(framework, epde_path=None):
    if framework == "pysindy":
        return load_module("pysindy_run", ROOT / "pysindy" / "run.py")
    if framework == "deepmod":
        return load_module("deepmod_run", ROOT / "deepmod" / "run.py")
    if framework == "epde":
        configure_epde(epde_path)
        return load_module("epde_run", ROOT / "epde" / "run_config_api.py")
    if framework == "discover":
        return load_module(
            "discover_run",
            ROOT / "discover" / "run.py",
            extra_paths=(ROOT / "discover",),
        )
    if framework == "edl":
        return load_module(
            "edl_run",
            ROOT / "edl" / "run.py",
            extra_paths=(ROOT / "edl" / "EDL" / "evaluation",),
        )
    if framework == "vwsr":
        return load_module(
            "vwsr_run",
            ROOT / "vwsr" / "run.py",
            extra_paths=(ROOT / "epde" / "EPDE",),
        )
    raise ValueError(f"Unknown framework: {framework}")


def load_thesis_metrics():
    """Compatibility name for callers; metrics now come from epde_bench."""
    global _THESIS_METRICS
    if _THESIS_METRICS is None:
        configure_epde()
        from epde_bench import metrics
        _THESIS_METRICS = metrics
    return _THESIS_METRICS


def epde_truth_text_alternatives(dataset):
    if dataset in _EPDE_TRUTH_CACHE:
        return _EPDE_TRUTH_CACHE[dataset]
    name = EPDE_DATASET_REGISTRY.get(dataset)
    if name is None:
        return []
    configure_epde()
    from epde_bench import datasets
    problem = datasets.load(name)
    alternatives = problem.truth_systems
    # PDEComp's record uses half the coefficients of the current EPDE record.
    # Keep its original law and window; never silently change the benchmark.
    if dataset == "pde_divide_data.npy":
        alternatives = [["-1.0 * du/dx1{power: 1.0} + 0.25 * d^2u/dx1^2{power: 1.0} * x{power: 1.0, dim: 1.0} = du/dx0{power: 1.0} * x{power: 1.0, dim: 1.0}"]]
    _EPDE_TRUTH_CACHE[dataset] = alternatives
    return alternatives


def epde_canonical_alternatives(dataset, single_equation=False):
    metrics = load_thesis_metrics()
    text_alternatives = epde_truth_text_alternatives(dataset)
    if single_equation:
        canonical = []
        for alternative in text_alternatives:
            for equation_text in alternative:
                canonical.append(metrics.canonical_tokens([equation_text]))
        return canonical
    return [metrics.canonical_tokens(alternative) for alternative in text_alternatives]


def epde_text_alternatives(dataset, single_equation=False):
    text_alternatives = epde_truth_text_alternatives(dataset)
    if single_equation:
        return [[equation_text] for alternative in text_alternatives for equation_text in alternative]
    return text_alternatives


def epde_structural_metrics(dataset, equation_texts, single_equation=False):
    truth_alternatives = epde_canonical_alternatives(dataset, single_equation=single_equation)
    truth_text_alternatives = epde_text_alternatives(dataset, single_equation=single_equation)
    if not truth_alternatives:
        return None

    metrics = load_thesis_metrics()
    discovered = metrics.canonical_tokens(equation_texts)
    hamming_value = metrics.hamming_best(discovered, truth_alternatives)
    coefficient_error = metrics.coefficient_error_best(equation_texts, truth_text_alternatives)
    if coefficient_error != coefficient_error:
        coefficient_error = ""
    return {
        "hamming": hamming_value,
        "success": hamming_value == 0,
        "coefficient_error": coefficient_error,
    }


def dataset_axis_names(dataset):
    if dataset == "ns_data.mat":
        return {0: "t", 1: "y", 2: "x"}
    if dataset in {"ode_data.npy", "vdp_data.npy", "lorenz_data.npy", "lotka_data.npy", "ODE_simple_discovery"}:
        return {0: "t"}
    return {0: "t", 1: "x"}


def derivative_suffix(axis_name, order):
    return axis_name * order


def epde_variable_name(dataset, variable):
    if dataset == "lotka_data.npy":
        return {"u": "x0", "v": "x1"}.get(variable, variable)
    if dataset == "lorenz_data.npy":
        return {"u": "x0", "v": "x1", "w": "x2"}.get(variable, variable)
    return variable


def format_frequency(value):
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-3:
        return "" if rounded == 1 else f"{rounded} "
    return f"{float(value):.4g} "


def canonical_epde_factor(factor, dataset):
    axes = dataset_axis_names(dataset)
    factor = factor.strip()

    derivative_match = re.match(
        r"d(?:\^(\d+))?([A-Za-z]\w*)/dx(\d+)(?:\^\d+)?\{power: ([0-9.]+)",
        factor,
    )
    if derivative_match:
        order = int(derivative_match.group(1) or 1)
        variable = epde_variable_name(dataset, derivative_match.group(2))
        axis_name = axes.get(int(derivative_match.group(3)), f"x{derivative_match.group(3)}")
        power = int(round(float(derivative_match.group(4))))
        name = f"{variable}_{derivative_suffix(axis_name, order)}"
        return name if power == 1 else f"{name}^{power}"

    grid_match = re.match(r"x_(\d+)\{power: ([0-9.]+)(?:,[^}]*)?", factor)
    if grid_match:
        axis_name = axes.get(int(grid_match.group(1)), f"x{grid_match.group(1)}")
        power = int(round(float(grid_match.group(2))))
        return axis_name if power == 1 else f"{axis_name}^{power}"

    variable_match = re.match(r"([A-Za-z]\w*)\{power: ([0-9.]+)", factor)
    if variable_match and variable_match.group(1) not in {"sin", "cos"}:
        variable = epde_variable_name(dataset, variable_match.group(1))
        power = int(round(float(variable_match.group(2))))
        return variable if power == 1 else f"{variable}^{power}"

    trig_match = re.match(
        r"(sin|cos)\{power: ([0-9.]+), freq: ([0-9.eE+-]+), dim: ([0-9.]+)",
        factor,
    )
    if trig_match:
        trig_name = trig_match.group(1)
        power = int(round(float(trig_match.group(2))))
        frequency = format_frequency(trig_match.group(3))
        axis_name = axes.get(int(round(float(trig_match.group(4)))), "t")
        name = f"{trig_name}({frequency}{axis_name})"
        return name if power == 1 else f"{name}^{power}"

    return factor


def canonical_epde_feature(feature, dataset):
    factors = [
        canonical_epde_factor(factor, dataset)
        for factor in feature.split(" * ")
    ]

    def sort_key(name):
        if re.fullmatch(r"[A-Za-z]\w*(\^\d+)?", name):
            return (0, name)
        if "_" in name:
            return (1, name)
        if name.startswith(("sin(", "cos(")):
            return (2, name)
        return (3, name)

    return " ".join(sorted(factors, key=sort_key))


def identity_feature(feature, dataset):
    return feature


def normalize_result(result):
    targets = result.get("targets") or [result["target"]]
    features = result["features"]
    coefficients = result["coefficients"]
    equation_texts = result.get("equation_texts") or []

    if len(targets) == 1 and features and isinstance(features[0], str):
        features = [features]
    if len(targets) == 1 and np.asarray(coefficients, dtype=object).ndim == 1:
        coefficients = [coefficients]
    if not equation_texts:
        model = result.get("model", "")
        equation_texts = [line for line in str(model).splitlines() if line.strip()]

    return {
        **result,
        "targets": targets,
        "features": features,
        "coefficients": coefficients,
        "equation_texts": equation_texts,
    }


def coefficient_by_feature(result, target_index, dataset=None, feature_normalizer=identity_feature):
    result = normalize_result(result)
    coefficients = np.asarray(result["coefficients"][target_index], dtype=float)
    features = result["features"][target_index]
    coefficient_map = {}
    for feature, coefficient in zip(features, coefficients):
        normalized = feature_normalizer(feature, dataset)
        coefficient_map[normalized] = coefficient_map.get(normalized, 0.0) + float(coefficient)
    return coefficient_map


def active_features(coefficient_map, tolerance=COEFFICIENT_TOLERANCE):
    return {
        feature
        for feature, coefficient in coefficient_map.items()
        if abs(coefficient) > tolerance
    }


def relative_error_sum(fitted_coefficients, true_coefficients):
    total = 0.0
    for feature, true_value in true_coefficients.items():
        if abs(true_value) <= np.finfo(float).eps:
            raise ValueError(
                f"Ground-truth coefficient for active term {feature!r} must be non-zero"
            )
        fitted_value = fitted_coefficients.get(feature, 0.0)
        total += abs(fitted_value - true_value) / abs(true_value)
    return total


def true_coefficient_alternatives(dataset, target):
    primary = TRUE_COEFFICIENTS.get(dataset, {}).get(target)
    if primary is None:
        return []
    return [
        primary,
        *TRUE_COEFFICIENT_ALTERNATIVES.get(dataset, {}).get(target, []),
    ]


def structure_hamming(active, expected):
    return len(active - expected) + len(expected - active)


def best_true_coefficient_match(dataset, target, fitted_coefficients, tolerance=COEFFICIENT_TOLERANCE):
    alternatives = true_coefficient_alternatives(dataset, target)
    if not alternatives:
        return None

    active = active_features(fitted_coefficients, tolerance=tolerance)
    matches = []
    for true_coefficients in alternatives:
        expected = set(true_coefficients)
        matches.append({
            "true_coefficients": true_coefficients,
            "expected": expected,
            "missing": expected - active,
            "extra": active - expected,
            "hamming": structure_hamming(active, expected),
            "relative_error_sum": relative_error_sum(fitted_coefficients, true_coefficients),
        })
    return min(matches, key=lambda match: (match["hamming"], match["relative_error_sum"]))


def format_feature_list(features):
    return "; ".join(sorted(features))


def framework_feature_normalizer(framework):
    if framework == "epde":
        return canonical_epde_feature
    return identity_feature


def normalize_target(framework, target, dataset):
    if framework == "epde":
        return canonical_epde_feature(target, dataset)
    return target


def run_framework(framework, module, dataset, args, quiet=True):
    data, x, y, z, t = load_data(dataset)
    protocol = getattr(args, "protocol", FIXED_PROTOCOL)
    native_options = {
        "device": args.device,
        "allow_external_llm": getattr(args, "allow_external_llm", False),
        "seed": getattr(args, "algorithm_seed", 0),
    }
    if getattr(args, "native_max_iterations", None) is not None:
        native_options["max_iterations"] = args.native_max_iterations
    if getattr(args, "native_max_samples", None) is not None:
        native_options["max_samples"] = args.native_max_samples
    if framework == "pysindy":
        return module.run_sindy(data, x, y, z, t, dataset, protocol=protocol, native_options=native_options)
    if framework == "deepmod":
        return module.run_deepmod(data, x, y, z, t, dataset, protocol=protocol, native_options=native_options)
    if framework == "epde":
        return module.run_epde(
            data,
            x,
            y,
            z,
            t,
            dataset,
            device=args.device,
            solution_index=args.solution_index,
            only_print=not quiet,
            return_all=getattr(args, "epde_pareto_oracle", False),
            protocol=protocol,
            native_options=native_options,
        )
    if framework == "discover":
        return module.run_discover(
            data, x, y, z, t, dataset, only_print=not quiet,
            protocol=protocol, native_options=native_options,
        )
    if framework == "edl":
        return module.run_edl(data, x, y, z, t, dataset, protocol=protocol, native_options=native_options)
    if framework == "vwsr":
        return module.run_vwsr(data, x, y, z, t, dataset, protocol=protocol, native_options=native_options)
    raise ValueError(f"Unknown framework: {framework}")


def select_epde_pareto_oracle(dataset, result):
    """Select against ground truth for diagnostic upper-bound experiments only."""

    candidates = result.get("candidates")
    if not candidates:
        return result

    scored = []
    for index, candidate in enumerate(candidates):
        candidate = normalize_result(candidate)
        structure = epde_structural_metrics(
            dataset,
            candidate.get("equation_texts") or [],
            single_equation=False,
        )
        if structure is None:
            score = (10**9, index)
            structure = {}
        else:
            coeff_error = structure.get("coefficient_error", "")
            coeff_score = float(coeff_error) if coeff_error != "" else 10**6
            score = (structure["hamming"], coeff_score, index)
        scored.append((score, candidate, structure))

    scored.sort(key=lambda item: item[0])
    best_score, best_candidate, best_structure = scored[0]
    return {
        **best_candidate,
        "candidate_count": len(candidates),
        "selected_candidate_index": best_score[-1],
        "best_structure_hamming": best_structure.get("hamming", ""),
        "best_structure_success": best_structure.get("success", ""),
        "selection_policy": "ground_truth_pareto_oracle",
    }


def summarize_target(framework, dataset, result, target_index, runtime_seconds):
    result = normalize_result(result)
    raw_target = result["targets"][target_index]
    target = normalize_target(framework, raw_target, dataset)
    features = result["features"][target_index]
    reported_library_sizes = result.get("library_sizes", {})
    library_size = reported_library_sizes.get(raw_target, len(features))
    feature_normalizer = framework_feature_normalizer(framework)
    fitted_coefficients = coefficient_by_feature(
        result,
        target_index,
        dataset=dataset,
        feature_normalizer=feature_normalizer,
    )
    active = active_features(fitted_coefficients)
    truth_match = best_true_coefficient_match(dataset, target, fitted_coefficients)

    structure_hamming = ""
    structure_success = ""
    coefficient_error = ""
    if truth_match is None:
        expected = set()
        missing = set()
        extra = active
        error_sum = ""
        truth_defined = False
    else:
        expected = truth_match["expected"]
        missing = truth_match["missing"]
        extra = truth_match["extra"]
        error_sum = truth_match["relative_error_sum"]
        truth_defined = True
        structure_hamming = truth_match["hamming"]
        structure_success = structure_hamming == 0
        if structure_success:
            coefficient_error = error_sum

    if framework == "epde":
        equation_texts = result.get("equation_texts") or []
        if target_index < len(equation_texts):
            structure = epde_structural_metrics(
                dataset,
                [equation_texts[target_index]],
                single_equation=True,
            )
            if structure is not None:
                structure_hamming = structure["hamming"]
                structure_success = structure["success"]
                coefficient_error = structure["coefficient_error"]
                truth_defined = True
                error_sum = coefficient_error
                if structure_success:
                    missing = set()
                    extra = set()

    return {
        "framework": framework,
        "protocol": result.get("protocol", FIXED_PROTOCOL),
        "status": "ok",
        "dataset": dataset,
        "target": target,
        "raw_target": raw_target,
        "runtime_seconds": runtime_seconds,
        "library_size": library_size,
        "truth_defined": truth_defined,
        "true_terms_count": len(expected),
        "active_terms_count": len(active),
        "relative_error_sum": error_sum,
        "missing_terms": format_feature_list(missing),
        "extra_terms": format_feature_list(extra),
        "expected_terms": format_feature_list(expected),
        "active_terms": format_feature_list(active),
        "structure_hamming": structure_hamming,
        "structure_success": structure_success,
        "coefficient_error": coefficient_error,
        "candidate_count": result.get("candidate_count", ""),
        "selected_candidate_index": result.get("selected_candidate_index", ""),
        "selection_policy": result.get("selection_policy", ""),
        "model": result.get("model", ""),
    }


def summarize_system(framework, dataset, target_rows, runtime_seconds, result=None):
    error_values = [
        row["relative_error_sum"]
        for row in target_rows
        if row["relative_error_sum"] != ""
    ]
    truth_defined = len(error_values) == len(target_rows)
    missing_parts = [row["missing_terms"] for row in target_rows if row["missing_terms"]]
    extra_parts = [row["extra_terms"] for row in target_rows if row["extra_terms"]]
    hamming_values = [
        row["structure_hamming"]
        for row in target_rows
        if row["structure_hamming"] != ""
    ]
    structure_hamming = (
        sum(hamming_values) if len(hamming_values) == len(target_rows) else ""
    )
    structure_success = (
        all(row["structure_success"] is True for row in target_rows)
        if structure_hamming != ""
        else ""
    )
    coefficient_values = [
        row["coefficient_error"]
        for row in target_rows
        if row["coefficient_error"] != ""
    ]
    coefficient_error = (
        sum(coefficient_values)
        if structure_success and len(coefficient_values) == len(target_rows)
        else ""
    )
    if framework == "epde" and result is not None:
        equation_texts = normalize_result(result).get("equation_texts") or []
        structure = epde_structural_metrics(
            dataset,
            equation_texts,
            single_equation=False,
        )
        if structure is not None:
            structure_hamming = structure["hamming"]
            structure_success = structure["success"]
            coefficient_error = structure["coefficient_error"]
            truth_defined = True
            error_values = [coefficient_error] if coefficient_error != "" else []
            if structure_success:
                missing_parts = []
                extra_parts = []
    library_sizes = [row["library_size"] for row in target_rows]
    system_library_size = (
        sum(library_sizes)
        if all(isinstance(size, (int, float, np.integer, np.floating)) for size in library_sizes)
        else ""
    )
    return {
        "framework": framework,
        "protocol": target_rows[0].get("protocol", FIXED_PROTOCOL),
        "status": "ok",
        "dataset": dataset,
        "target": "__system__",
        "raw_target": "__system__",
        "runtime_seconds": runtime_seconds,
        "library_size": system_library_size,
        "truth_defined": truth_defined,
        "true_terms_count": sum(row["true_terms_count"] for row in target_rows),
        "active_terms_count": sum(row["active_terms_count"] for row in target_rows),
        "relative_error_sum": sum(error_values) if truth_defined and error_values else "",
        "missing_terms": " | ".join(missing_parts),
        "extra_terms": " | ".join(extra_parts),
        "expected_terms": "",
        "active_terms": "",
        "structure_hamming": structure_hamming,
        "structure_success": structure_success,
        "coefficient_error": coefficient_error,
        "candidate_count": result.get("candidate_count", "") if result else "",
        "selected_candidate_index": result.get("selected_candidate_index", "") if result else "",
        "selection_policy": result.get("selection_policy", "") if result else "",
        "model": "",
    }


def summarize_dataset(framework, module, dataset, args, quiet=True):
    start = time.perf_counter()
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_framework(framework, module, dataset, args, quiet=True)
    else:
        result = run_framework(framework, module, dataset, args, quiet=False)
    runtime_seconds = time.perf_counter() - start
    if hasattr(args, "raw_results"):
        args.raw_results.append({"framework": framework, "dataset": dataset, "result": result})

    if framework == "epde":
        if getattr(args, "epde_pareto_oracle", False):
            result = select_epde_pareto_oracle(dataset, result)
        else:
            result = {
                **result,
                "selection_policy": f"framework_solution_index:{args.solution_index}",
            }
    result = normalize_result(result)
    rows = [
        summarize_target(framework, dataset, result, target_index, runtime_seconds)
        for target_index, _ in enumerate(result["targets"])
    ]
    if len(rows) > 1:
        rows.append(summarize_system(framework, dataset, rows, runtime_seconds, result=result))
    return rows


def error_row(framework, dataset, error, protocol=FIXED_PROTOCOL, status="error"):
    return {
        "framework": framework,
        "protocol": protocol,
        "status": status,
        "dataset": dataset,
        "target": "",
        "raw_target": "",
        "runtime_seconds": "",
        "library_size": "",
        "truth_defined": False,
        "true_terms_count": "",
        "active_terms_count": "",
        "relative_error_sum": "",
        "missing_terms": "",
        "extra_terms": "",
        "expected_terms": "",
        "active_terms": "",
        "structure_hamming": "",
        "structure_success": "",
        "coefficient_error": "",
        "candidate_count": "",
        "selected_candidate_index": "",
        "selection_policy": "",
        "model": "",
        "error": str(error),
    }


def write_rows(rows, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "framework",
        "protocol",
        "status",
        "dataset",
        "target",
        "raw_target",
        "runtime_seconds",
        "library_size",
        "truth_defined",
        "true_terms_count",
        "active_terms_count",
        "relative_error_sum",
        "missing_terms",
        "extra_terms",
        "expected_terms",
        "active_terms",
        "structure_hamming",
        "structure_success",
        "coefficient_error",
        "candidate_count",
        "selected_candidate_index",
        "selection_policy",
        "model",
        "error",
    ]
    with open(output_file, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_output(framework, protocol=FIXED_PROTOCOL):
    if framework == "all":
        name = "clean_run_metrics.csv" if protocol == FIXED_PROTOCOL else "native_clean_run_metrics.csv"
        return ROOT / "results" / name
    if protocol == NATIVE_PROTOCOL:
        return ROOT / "results" / framework / "native" / "clean_run_metrics.csv"
    return ROOT / "results" / framework / "clean_run_metrics.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("framework", choices=["pysindy", "deepmod", "epde", "discover", "edl", "vwsr", "all"])
    parser.add_argument("--protocol", choices=PROTOCOLS, default=FIXED_PROTOCOL)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show-equations", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epde-path", help="Modern EPDE checkout (or set EPDE_PATH).")
    parser.add_argument("--solution-index", type=int, default=0)
    parser.add_argument(
        "--epde-pareto-oracle",
        action="store_true",
        help="Diagnostic upper bound: choose an EPDE Pareto candidate using ground truth; not valid for primary comparisons.",
    )
    parser.add_argument("--native-max-iterations", type=int, default=None)
    parser.add_argument("--native-max-samples", type=int, default=None)
    parser.add_argument("--allow-external-llm", action="store_true")
    parser.add_argument("--algorithm-seed", type=int, default=0)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Return success even when every selected run is skipped or unsupported.",
    )
    return parser.parse_args()


def selected_frameworks(framework):
    if framework == "all":
        return ["pysindy", "deepmod", "epde", "discover", "edl", "vwsr"]
    return [framework]


def main():
    args = parse_args()
    output_file = Path(args.output) if args.output else default_output(args.framework, args.protocol)
    all_rows = []
    args.raw_results = []

    for framework in selected_frameworks(args.framework):
        try:
            module = load_framework_module(framework, epde_path=args.epde_path)
        except Exception as error:
            datasets = args.datasets or DEFAULT_DATASETS
            all_rows.extend(error_row(framework, dataset, error, args.protocol) for dataset in datasets)
            print(f"\n=== Unable to load {framework}: {error} ===")
            continue
        datasets = args.datasets or getattr(module, "DATASETS", DEFAULT_DATASETS)
        for dataset in datasets:
            print(f"\n=== Measuring {framework} / {dataset} ===")
            try:
                seed_everything(args.algorithm_seed)
                rows = summarize_dataset(
                    framework,
                    module,
                    dataset,
                    args,
                    quiet=not args.show_equations,
                )
                all_rows.extend(rows)
                for row in rows:
                    print(
                        f"  {row['target']}: time={row['runtime_seconds']:.4f}s, "
                        f"library={row['library_size']}, "
                        f"rel_error_sum={row['relative_error_sum']}"
                    )
            except ProtocolSkippedError as error:
                all_rows.append(error_row(framework, dataset, error, args.protocol, status="skipped"))
                print(f"  skipped: {error}")
            except ProtocolUnavailableError as error:
                all_rows.append(error_row(framework, dataset, error, args.protocol, status="unsupported"))
                print(f"  unsupported: {error}")
            except Exception as error:
                all_rows.append(error_row(framework, dataset, error, args.protocol))
                print(f"  error: {error}")

    write_rows(all_rows, output_file)
    json_file = output_file.with_suffix(".json")
    with json_file.open("w", encoding="utf-8") as handle:
        json.dump({"protocol": args.protocol, "algorithm_seed": args.algorithm_seed,
                   "runs": args.raw_results, "metrics": all_rows}, handle, indent=2)
    print(f"\nSaved clean-run metrics to {output_file}")
    return benchmark_exit_code(all_rows, allow_empty=args.allow_empty)


if __name__ == "__main__":
    sys.exit(main())
