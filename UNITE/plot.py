import math
import random
import matplotlib.pyplot as plt
from sympy import primerange
from functools import reduce
from tqdm import tqdm


# ----------- Utility -----------
def prod(lst):
    result = 1
    for x in lst:
        result *= x
    return result


# ----------- 1. Theoretical FPR -----------
def theoretical_fpr(n):
    primes = list(primerange(n + 1, n * n))
    product = 1
    count = 0
    for p in primes:
        if product * p < 2**n - 1:
            product *= p
            count += 1
        else:
            break
    return count, len(primes), count / len(primes) if len(primes) > 0 else 0, product


# ----------- 2. Empirical FPR -----------
def empirical_fpr(n, K, trials=1000):
    primes = list(primerange(n + 1, n * n))
    if not primes:
        return 0
    return sum(1 for _ in range(trials) if K % random.choice(primes) == 0) / trials


# ----------- 3. Upper Bound FPR -----------
def upper_bound_fpr(n):
    log2 = math.log(2)
    logn = math.log(n)
    N = int((n * log2) / logn)
    D = max(1, int((n**2 / math.log(n**2)) - (n / math.log(n))))
    return N / D


# ----------- Plot 1: Empirical vs Theoretical -----------
def plot_empirical_theoretical(n_start=6, n_end=1000, step=20, trials=1000):
    ns = list(range(n_start, n_end + 1, step))
    theo_rates, emp_rates = [], []

    print("Generating empirical vs theoretical FPR plot...")
    for n in tqdm(ns):
        _, _, theo_rate, K = theoretical_fpr(n)
        emp_rate = empirical_fpr(n, K, trials)
        theo_rates.append(theo_rate)
        emp_rates.append(emp_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(ns, theo_rates, label="Theoretical FPR", marker="o")
    plt.plot(ns, emp_rates, label="Empirical FPR", marker="x")
    plt.xlabel("n")
    plt.ylabel("False Positive Rate")
    plt.title("Empirical vs Theoretical False Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("empirical_vs_theoretical.png", dpi=300)
    # plt.show()


# ----------- Plot 2: Upper Bound (log-log) -----------
def plot_upper_bound_loglog():
    ns = [10**i for i in range(1, 101)]
    rates = [upper_bound_fpr(n) for n in ns]

    plt.figure(figsize=(10, 6))
    plt.loglog(ns, rates, marker="s", linestyle="-")
    plt.xlabel("n (log scale)")
    plt.ylabel("Upper Bound FPR (log scale)")
    plt.title("Upper Bound False Positive Rate (n = 10 to 10^100)")
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig("upper_bound_loglog.png", dpi=300)
    # plt.show()


# ----------- Main -----------
if __name__ == "__main__":
    # Run both plots
    plot_empirical_theoretical(
        n_start=6, n_end=300, step=10, trials=1000
    )  # keep runtime manageable
    plot_upper_bound_loglog()
