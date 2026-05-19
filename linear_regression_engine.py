# Made by Rushil Jain

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Constants ---
N_SAMPLES = 100
X_RANGE = (0, 15)
NOISE_STD = 0.5
TRUE_BETA_0 = 2
TRUE_BETA_1 = 6

# --- Data Generation (modern NumPy RNG) ---
rng = np.random.default_rng(42)
X = rng.uniform(*X_RANGE, N_SAMPLES)
epsilon = rng.normal(0, NOISE_STD, N_SAMPLES)
Y = TRUE_BETA_0 + TRUE_BETA_1 * X + epsilon


def simple_linear_regression(X, Y):
    """
    Estimate OLS coefficients for simple linear regression.

    Parameters
    ----------
    X : np.ndarray
        1D array of predictor values.
    Y : np.ndarray
        1D array of response values.

    Returns
    -------
    beta_0 : float
        Estimated intercept.
    beta_1 : float
        Estimated slope.

    Raises
    ------
    ValueError
        If X and Y have different lengths, fewer than 2 points, or X has zero variance.
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    if len(X) < 2:
        raise ValueError("Need at least 2 data points")
    if np.var(X) == 0:
        raise ValueError("X has zero variance — regression is undefined")

    x_mean, y_mean = np.mean(X), np.mean(Y)

    numerator = np.sum((X - x_mean) * (Y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    beta_1 = numerator / denominator

    beta_0 = y_mean - beta_1 * x_mean

    return beta_0, beta_1


def evaluate_model(X, Y, beta_0, beta_1):
    """
    Compute evaluation metrics for a fitted simple linear regression model.

    Parameters
    ----------
    X : np.ndarray
        1D array of predictor values.
    Y : np.ndarray
        1D array of observed response values.
    beta_0 : float
        Estimated intercept.
    beta_1 : float
        Estimated slope.

    Returns
    -------
    dict with keys:
        RSS           : Residual Sum of Squares
        RSE           : Residual Standard Error (estimate of sigma)
        se_beta_0     : Standard error of beta_0
        se_beta_1     : Standard error of beta_1
        t_stat_beta_0 : t-statistic for H0: beta_0 = 0
        t_stat_beta_1 : t-statistic for H0: beta_1 = 0
        p_value_beta_0: Two-sided p-value for beta_0
        p_value_beta_1: Two-sided p-value for beta_1
        R_squared     : Coefficient of determination

    Raises
    ------
    ValueError
        If X and Y have different lengths or fewer than 3 points.
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")
    if len(X) < 3:
        raise ValueError("Need at least 3 data points to evaluate the model")

    n = len(X)

    Y_pred = beta_0 + beta_1 * X
    residuals = Y - Y_pred
    RSS = np.sum(residuals ** 2)

    x_mean = np.mean(X)
    s_xx = np.sum((X - x_mean) ** 2)
    RSE = np.sqrt(RSS / (n - 2))
    se_beta_1 = RSE / np.sqrt(s_xx)
    se_beta_0 = RSE * np.sqrt(1 / n + x_mean ** 2 / s_xx)

    t_stat_beta_1 = beta_1 / se_beta_1
    p_value_beta_1 = 2 * (1 - stats.t.cdf(abs(t_stat_beta_1), df=n - 2))

    t_stat_beta_0 = beta_0 / se_beta_0
    p_value_beta_0 = 2 * (1 - stats.t.cdf(abs(t_stat_beta_0), df=n - 2))

    TSS = np.sum((Y - np.mean(Y)) ** 2)
    R_squared = 1 - (RSS / TSS)

    return {
        'RSS': RSS,
        'RSE': RSE,
        'se_beta_0': se_beta_0,
        'se_beta_1': se_beta_1,
        't_stat_beta_0': t_stat_beta_0,
        't_stat_beta_1': t_stat_beta_1,
        'p_value_beta_0': p_value_beta_0,
        'p_value_beta_1': p_value_beta_1,
        'R_squared': R_squared,
    }


def fmt_pvalue(p):
    """Format p-values: fixed 4 d.p. if >= 0.0001, else scientific notation."""
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"


# --- Fit and Evaluate ---
estimated_beta_0, estimated_beta_1 = simple_linear_regression(X, Y)
results = evaluate_model(X, Y, estimated_beta_0, estimated_beta_1)

print(f"Estimated beta_0:    {estimated_beta_0:.4f}  (true: {TRUE_BETA_0})")
print(f"Estimated beta_1:    {estimated_beta_1:.4f}  (true: {TRUE_BETA_1})")
print(f"RSS:                 {results['RSS']:.4f}")
print(f"RSE:                 {results['RSE']:.4f}")
print(f"SE(beta_0):          {results['se_beta_0']:.4f}")
print(f"SE(beta_1):          {results['se_beta_1']:.4f}")
print(f"t-stat (beta_0):     {results['t_stat_beta_0']:.4f}")
print(f"p-value (beta_0):    {fmt_pvalue(results['p_value_beta_0'])}")
print(f"t-stat (beta_1):     {results['t_stat_beta_1']:.4f}")
print(f"p-value (beta_1):    {fmt_pvalue(results['p_value_beta_1'])}")
print(f"R-squared:           {results['R_squared']:.4f}")

# --- Plot ---
X_sorted = np.sort(X)
Y_line = estimated_beta_0 + estimated_beta_1 * X_sorted

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, color='blue', alpha=0.5, label='Data points')
plt.plot(X_sorted, Y_line, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
