# -*- coding: utf-8 -*-
"""
Regularized nonlinear conjugate-gradient inversion.

This module contains only the inversion-side code extracted and cleaned from
``1D_inversion_RCG.ipynb``.  Forward modelling, synthetic true-model creation,
plotting, and notebook-specific execution cells are intentionally removed.

The forward model must be supplied by the user in this form:

    calculated_data = forward_model(model, x_data)

where ``model`` is a one-dimensional parameter vector.  If you want to invert
both resistivity and layer thickness, concatenate them into one vector and split
inside your own ``forward_model``.
"""

import time
from typing import Callable, Dict, Optional

import numpy as np

ArrayLike = np.ndarray


def modellog(linear_model: ArrayLike, model_min: ArrayLike, model_max: ArrayLike) -> np.ndarray:
    """
    Map bounded model parameters to an unconstrained logit-like space.

        x = log((m - m_min) / (m_max - m))

    Parameters are clipped slightly inside the bounds to avoid infinities.
    """
    linear_model = np.asarray(linear_model, dtype=float)
    model_min = np.asarray(model_min, dtype=float)
    model_max = np.asarray(model_max, dtype=float)

    linear_model = np.clip(linear_model, model_min + 1e-12, model_max - 1e-12)
    return np.log((linear_model - model_min) / (model_max - linear_model))


def return_log(log_model: ArrayLike, model_min: ArrayLike, model_max: ArrayLike) -> np.ndarray:
    """
    Map parameters from the unconstrained space back to the bounded model space.

        m(x) = (m_min + exp(x) * m_max) / (1 + exp(x))
    """
    log_model = np.asarray(log_model, dtype=float)
    model_min = np.asarray(model_min, dtype=float)
    model_max = np.asarray(model_max, dtype=float)

    # Avoid overflow in exp for very large updates.
    log_model = np.clip(log_model, -700.0, 700.0)
    exp_model = np.exp(log_model)
    return (model_min + exp_model * model_max) / (1.0 + exp_model)


def rmspe(calculated_data: ArrayLike, observed_data: ArrayLike) -> float:
    """
    Root mean squared percentage error.
    """
    calculated_data = np.asarray(calculated_data, dtype=float)
    observed_data = np.asarray(observed_data, dtype=float)
    observed_safe = np.clip(observed_data, 1e-30, np.inf)

    return float(
        np.sqrt(np.mean(((calculated_data - observed_data) / observed_safe) ** 2))
        * 100.0
    )


def _safe_log(data: ArrayLike) -> np.ndarray:
    """Take log after clipping non-positive values to a very small positive value."""
    return np.log(np.clip(np.asarray(data, dtype=float), 1e-30, np.inf))


def finite_difference_log_jacobian(
    forward_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_data: ArrayLike,
    observed_data: ArrayLike,
    log_model: ArrayLike,
    model_min: ArrayLike,
    model_max: ArrayLike,
    diff_step: float = 0.05,
) -> np.ndarray:
    """
    Approximate the Jacobian of the log residual by central differences.

    The residual is

        r = log(calculated_data) - log(observed_data)

    and the derivative is taken with respect to the unconstrained model
    parameter ``log_model``.
    """
    observed_log = _safe_log(observed_data)
    log_model = np.asarray(log_model, dtype=float)
    model_min = np.asarray(model_min, dtype=float)
    model_max = np.asarray(model_max, dtype=float)

    n_data = np.asarray(observed_data, dtype=float).size
    n_param = log_model.size
    jacobian = np.zeros((n_data, n_param), dtype=float)

    for j in range(n_param):
        log_plus = log_model.copy()
        log_minus = log_model.copy()
        log_plus[j] += diff_step
        log_minus[j] -= diff_step

        model_plus = return_log(log_plus, model_min, model_max)
        model_minus = return_log(log_minus, model_min, model_max)

        residual_plus = _safe_log(forward_model(model_plus, x_data)) - observed_log
        residual_minus = _safe_log(forward_model(model_minus, x_data)) - observed_log

        jacobian[:, j] = (residual_plus - residual_minus) / (2.0 * diff_step)

    return jacobian


def _weight_diagonals(jacobian: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the diagonal entries corresponding to Wd^2 and Wm^2.

    In the original notebook, these were formed as diagonal matrices:

        Wd = diag(diag(J @ J.T) ** (1/2))
        Wm = diag(diag(J.T @ J) ** (1/4))

    This implementation stores only the diagonal entries to avoid unnecessary
    dense matrix operations.
    """
    row_power = np.sum(jacobian * jacobian, axis=1)
    col_power = np.sum(jacobian * jacobian, axis=0)

    wd2_diag = np.clip(row_power, 1e-30, np.inf)
    wm2_diag = np.sqrt(np.clip(col_power, 1e-30, np.inf))

    return wd2_diag, wm2_diag


def regularized_nlcg_inversion(
    forward_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_data: ArrayLike,
    observed_data: ArrayLike,
    initial_model: ArrayLike,
    model_min: ArrayLike,
    model_max: ArrayLike,
    max_iter: int = 1000,
    stop_rmspe: float = 1.0,
    diff_step: float = 0.05,
    q: float = 0.67,
    p: float = 0.96,
    alpha_init: Optional[float] = None,
    reference_log_model: Optional[ArrayLike] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray | float | int]:
    """
    Solve a nonlinear inverse problem using a regularized nonlinear CG method.

    Parameters
    ----------
    forward_model:
        Function that calculates predicted data from ``model`` and ``x_data``.
    x_data:
        Input data used by ``forward_model``.  This can be any NumPy-compatible
        array needed by the supplied forward model.
    observed_data:
        Observed data to fit.  Values are evaluated in log space, so they should
        be positive.
    initial_model:
        Initial model parameter vector in the original bounded parameter space.
    model_min, model_max:
        Lower and upper bounds of model parameters.
    max_iter:
        Maximum number of CG iterations.
    stop_rmspe:
        Stop when RMSPE becomes smaller than this value.
    diff_step:
        Central-difference step in the unconstrained model space.
    q:
        Decay factor of the regularization parameter.
    p:
        Damping factor applied to the model update.
    alpha_init:
        Initial regularization parameter.  If omitted, it is estimated from the
        first trial update, following the original notebook logic.
    reference_log_model:
        Reference model in the unconstrained parameter space.  If omitted, a
        zero vector is used, matching the original notebook's ``m_apr = 0``.
    verbose:
        If True, print iteration progress.

    Returns
    -------
    dict
        Dictionary containing the final model, calculated data, histories, final
        RMSPE, elapsed time, and number of iterations.
    """
    start = time.time()

    x_data = np.asarray(x_data, dtype=float)
    observed_data = np.asarray(observed_data, dtype=float)
    initial_model = np.asarray(initial_model, dtype=float)
    model_min = np.asarray(model_min, dtype=float)
    model_max = np.asarray(model_max, dtype=float)

    if initial_model.ndim != 1:
        raise ValueError("initial_model must be a one-dimensional array.")
    if not (initial_model.shape == model_min.shape == model_max.shape):
        raise ValueError("initial_model, model_min, and model_max must have the same shape.")
    if np.any(model_min >= model_max):
        raise ValueError("Every model_min value must be smaller than model_max.")
    if not (0.0 < q <= 1.0):
        raise ValueError("q must satisfy 0 < q <= 1.")
    if not (0.0 < p <= 1.0):
        raise ValueError("p must satisfy 0 < p <= 1.")

    log_model = modellog(initial_model, model_min, model_max)
    if reference_log_model is None:
        reference_log_model = np.zeros_like(log_model)
    else:
        reference_log_model = np.asarray(reference_log_model, dtype=float)
        if reference_log_model.shape != log_model.shape:
            raise ValueError("reference_log_model must have the same shape as initial_model.")

    rmspe_hist: list[float] = []
    alpha_hist: list[float] = []
    model_hist: list[np.ndarray] = []

    # Initial calculation.
    model = return_log(log_model, model_min, model_max)
    calculated_data = np.asarray(forward_model(model, x_data), dtype=float)
    residual = _safe_log(calculated_data) - _safe_log(observed_data)

    jacobian = finite_difference_log_jacobian(
        forward_model=forward_model,
        x_data=x_data,
        observed_data=observed_data,
        log_model=log_model,
        model_min=model_min,
        model_max=model_max,
        diff_step=diff_step,
    )
    wd2_diag, wm2_diag = _weight_diagonals(jacobian)

    # First CG direction: data-misfit term only, following the notebook.
    gradient = jacobian.T @ (wd2_diag * residual)
    previous_gradient = gradient.copy()
    previous_direction = gradient.copy()

    beta = np.dot(gradient, gradient) / max(np.dot(previous_gradient, previous_gradient), 1e-30)
    direction = gradient + beta * previous_direction

    denominator = np.dot(np.sqrt(wd2_diag) * (jacobian @ direction), np.sqrt(wd2_diag) * (jacobian @ direction))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise FloatingPointError("Invalid first CG step denominator.")

    step_length = np.dot(direction, gradient) / denominator
    log_model = log_model - p * step_length * direction
    model = return_log(log_model, model_min, model_max)
    calculated_data = np.asarray(forward_model(model, x_data), dtype=float)
    residual = _safe_log(calculated_data) - _safe_log(observed_data)

    previous_gradient = gradient
    previous_direction = direction

    if alpha_init is None:
        data_norm = np.linalg.norm(np.sqrt(wd2_diag) * residual) ** 2
        model_norm = np.linalg.norm(np.sqrt(wm2_diag) * (log_model - reference_log_model)) ** 2
        alpha_0 = data_norm / max(model_norm, 1e-30)
    else:
        alpha_0 = float(alpha_init)

    rmspe_now = rmspe(calculated_data, observed_data)
    rmspe_hist.append(rmspe_now)
    alpha_hist.append(alpha_0)
    model_hist.append(model.copy())

    if verbose:
        print(f"iteration=1, RMSPE={rmspe_now:.6g}, alpha={alpha_0:.6g}")

    if rmspe_now < stop_rmspe:
        elapsed = time.time() - start
        return {
            "model_final": model,
            "calculated_final": calculated_data,
            "rmspe_hist": np.asarray(rmspe_hist),
            "alpha_hist": np.asarray(alpha_hist),
            "model_hist": np.asarray(model_hist),
            "rmspe_final": float(rmspe_hist[-1]),
            "elapsed": elapsed,
            "iterations": len(rmspe_hist),
        }

    # Main CG loop.
    for iteration in range(2, int(max_iter) + 1):
        jacobian = finite_difference_log_jacobian(
            forward_model=forward_model,
            x_data=x_data,
            observed_data=observed_data,
            log_model=log_model,
            model_min=model_min,
            model_max=model_max,
            diff_step=diff_step,
        )
        wd2_diag, _ = _weight_diagonals(jacobian)

        alpha = alpha_0 * (q ** (iteration - 1))

        gradient = (
            jacobian.T @ (wd2_diag * residual)
            + alpha * wm2_diag * (log_model - reference_log_model)
        )

        beta = np.dot(gradient, gradient) / max(np.dot(previous_gradient, previous_gradient), 1e-30)
        direction = gradient + beta * previous_direction

        data_term = np.linalg.norm(np.sqrt(wd2_diag) * (jacobian @ direction)) ** 2
        model_term = alpha * (np.linalg.norm(np.sqrt(wm2_diag) * direction) ** 2)
        denominator = data_term + model_term

        if denominator <= 0.0 or not np.isfinite(denominator):
            if verbose:
                print(f"iteration={iteration}: invalid denominator; stopping.")
            break

        step_length = np.dot(direction, gradient) / denominator
        log_model = log_model - p * step_length * direction
        model = return_log(log_model, model_min, model_max)
        calculated_data = np.asarray(forward_model(model, x_data), dtype=float)
        residual = _safe_log(calculated_data) - _safe_log(observed_data)

        previous_gradient = gradient
        previous_direction = direction

        rmspe_now = rmspe(calculated_data, observed_data)
        rmspe_hist.append(rmspe_now)
        alpha_hist.append(alpha)
        model_hist.append(model.copy())

        if verbose:
            print(
                f"iteration={iteration}, RMSPE={rmspe_now:.6g}, "
                f"alpha={alpha:.6g}, step={step_length:.6g}"
            )

        if rmspe_now < stop_rmspe:
            break

    elapsed = time.time() - start

    return {
        "model_final": model,
        "calculated_final": calculated_data,
        "rmspe_hist": np.asarray(rmspe_hist),
        "alpha_hist": np.asarray(alpha_hist),
        "model_hist": np.asarray(model_hist),
        "rmspe_final": float(rmspe_hist[-1]),
        "elapsed": elapsed,
        "iterations": len(rmspe_hist),
    }
