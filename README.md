# Computational Algebra & Cryptography Library

A comprehensive Python library developed to manipulate **Finite Fields ($\mathbb{F}_{p^n}$)** and **Polynomial Rings**, implementing algebraic primitives essential for public-key cryptography from scratch.

## Key Features

### 1. Finite Fields Arithmetic (`cuerpos_finitos.py`)
* **Object-Oriented Design:** Custom classes for elements in Z_p and extension fields $\mathbb{F}_{q} = \mathbb{F}_p[x] / <g(x)>$.
* **Polynomial Operations:** Full implementation of polynomial arithmetic (addition, multiplication, division, GCD, Extended Euclidean Algorithm).
* **Irreducibility Tests:** Rabin's test for polynomial irreducibility over finite fields.

### 2. Polynomial Factorization (`factorizacion.py`)
Implementation of the complete **Cantor-Zassenhaus Algorithm** pipeline:
* **Square-Free Factorization:** Removes repeated factors using formal derivatives.
* **Distinct-Degree Factorization:** Separates factors by degree.
* **Equal-Degree Factorization:** Probabilistic splitting of factors of the same degree.

### 3. Tonelli-Shanks Algorithm (`tonelli_shanks.py`)
*  Computes square roots modulo $p$ (Critical for point compression in Elliptic Curve Cryptography).

### 4. High-Performance Arithmetic (`algoritmos_rapidos.py`)
* **Karatsuba Algorithm:** Fast polynomial multiplication reducing complexity to $O(n^{\log_2 3})$.
* **FFT / Cooley-Tukey:** Fast Fourier Transform over finite fields for quasi-linear arithmetic operations (Work in Progress).
* **Toeplitz Matrices:** Vectorized operations for efficient polynomial processing.

## Usage Example

```python
from finite_fields import cuerpo_fp, anillo_fp_x

# Initialize field F_7
F7 = cuerpo_fp(7)
R = anillo_fp_x(F7)

# Define polynomials f = x^2 + 1 and g = x + 3
f = R.elem_de_str("x^2 + 1")
g = R.elem_de_str("x + 3")

# Compute GCD
h = R.gcd(f, g)
print(f"GCD({f}, {g}) = {h}")
