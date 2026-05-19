# Linear Regression Engine

OLS linear regression implemented from scratch — deriving coefficients analytically and building a full evaluation pipeline (RSS, RSE, t-statistics, p-values) without using any ML libraries.

---

## What This Project Does

Given a dataset of (X, Y) pairs, this project:

- Estimates regression coefficients β₀ and β₁ using **Ordinary Least Squares (OLS)**
- Computes **RSS**, **RSE**, and **R²**
- Calculates **standard errors** for both coefficients
- Runs **t-tests** for both β₀ and β₁ with properly formatted p-values
- Validates inputs with descriptive errors
- Visualizes the data and fitted regression line

No `sklearn`. No `statsmodels`. Just NumPy and the math.

---

## Mathematical Background

The OLS estimators minimize the residual sum of squares:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}, \quad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

The **Residual Standard Error** (estimate of σ):

$$RSE = \sqrt{\frac{RSS}{n-2}}$$

Standard errors for the coefficients:

$$SE(\hat{\beta}_1) = \frac{RSE}{\sqrt{S_{xx}}}, \quad SE(\hat{\beta}_0) = RSE \sqrt{\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}}$$

t-statistics for hypothesis testing (H₀: βᵢ = 0):

$$t = \frac{\hat{\beta}_i}{SE(\hat{\beta}_i)} \sim t_{n-2}$$

---

## Project Structure

```
linear-regression-engine/
│
├── linear_regression_engine.ipynb     # Main notebook: implementation + visualization
├── requirements.txt                   # Dependencies
├── .gitignore                         # Python gitignore
└── README.md
```

---

## Getting Started

**Clone the repo**

```bash
git clone https://github.com/rushil--jain/linear-regression-engine.git
cd linear-regression-engine
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run the notebook**

Open `linear_regression_engine.ipynb` in Jupyter or VS Code and run all cells.

---

## Sample Output

The notebook generates synthetic data with known true parameters (β₀ = 2, β₁ = 6) and recovers them via OLS:

```
Estimated beta_0:    2.0447  (true: 2)
Estimated beta_1:    5.9973  (true: 6)
RSS:                 26.3201
RSE:                 0.5164
SE(beta_0):          0.1169
SE(beta_1):          0.0136
t-stat (beta_0):     17.4923
p-value (beta_0):    2.15e-32
t-stat (beta_1):     440.6574
p-value (beta_1):    1.07e-99
R-squared:           0.9995
```

---

## Implementation Notes

- **No ML libraries** — coefficients derived purely from the OLS normal equations
- **Modern NumPy RNG** — uses `np.random.default_rng(42)` for reproducibility
- **Validated inputs** — both functions raise descriptive `ValueError` exceptions for bad inputs
- **Correct p-value formatting** — scientific notation for very small p-values, fixed precision otherwise
- **Clean plot** — X is sorted before drawing the regression line to avoid a jagged artefact

---

## Dependencies

```
numpy
matplotlib
scipy
```

---

## Author

**Rushil Jain**  
[LinkedIn](https://www.linkedin.com/in/rushil--jain/)
