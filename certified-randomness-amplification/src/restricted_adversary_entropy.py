###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from scipy.special import gammainc as P
from scipy.stats import binom
import numpy as np
from math import ceil
from scipy.stats import hypergeom


mode = 'approx'
# mode = "exact"


def CDF_exact(L_val, phi, chi):
    """Exact cumulative density function of finite fidelity sampling
    See Supplementary material of Eq. III.11
    For original reference:
    See Eq. 73 of https://doi.org/10.13140/RG.2.2.24562.94400. Uses
    approximate formula (see CDF_approx) if the value cannot be
    computed exactly.

    Args:
        m (int): sample size (number of bitstrings)
        phi (float): fidelity of samples. Could be quantum fidelity,
                    classical fidelity, or effective fidelity of
                    quantum classical mixtures
        chi (list-like): XEB score threshold.

    Returns:
        NDArray: the cumulative density function of getting SQC
        less than or equal to the input xm
    """
    if phi == 1:
        return P(2 * L_val, L_val * (chi + 1))
    length = chi.shape[0]
    l = np.arange(L_val + 1)
    pmfs = binom.pmf(l, L_val, phi)
    Ps = P(
        L_val + l.repeat(length).reshape(L_val + 1, length),
        np.tile(L_val * (chi + 1), L_val + 1).reshape(L_val + 1, length),
    )
    return Ps.T @ pmfs


def CDF_approx(L_val, phi, chi):
    """approximate cumulative density function of finite fidelity sampling
    See Eq. 93 of https://doi.org/10.13140/RG.2.2.24562.94400. This is
    suitable when sample size m is large, in which case we use an
    approximate Gamma distribution.

    Args:
        m (int): sample size (number of bitstrings)
        phi (float): fidelity of samples. Could be quantum fidelity,
                    classical fidelity, or effective fidelity of
                    quantum classical mixtures
        xm (list-like): sum of quantum value (SQC).

    Returns:
        NDArray: the cumulative density function of getting SQC
        less than or equal to the input xm
    """
    N = 1  # N factor cancels out
    N_new = N * (1 + phi) / (1 + phi * (2 - phi))  # eq. 91
    phi_new = 2 * phi**2 / (1 + phi * (2 - phi))  # eq. 92
    return P(L_val * (1 + phi_new), N_new * L_val * (chi + 1))


if mode == "approx":
    CDF = CDF_approx
elif mode == "exact":
    CDF = CDF_exact
else:
    raise ValueError(f"Invalid mode {mode}")


def get_chi(L_val, pfail, phi, tol=10**-2):
    """Get chi threshold that achieves desired pfail

    Args:
        m (int): sample size (number of bitstrings)
        pfail: tolerated probability that honest server fails
        phi (float): fidelity of honest sampler
        atol (float): tolerance on approximating the target pfail

    Returns:
        float: chi. If no chi can achieve atol, retunns np.nan
    """

    high = 2
    low = -2
    min_error = np.inf
    tentative_result = np.nan
    i = 0
    while low <= high and i < 1000:
        i += 1
        chi = (high + low) / 2
        ans = CDF(L_val, phi, np.array([chi]))[0]
        error = (ans - pfail) / pfail
        if error > 0 and error < min_error:
            min_error = np.abs(error)
            tentative_result = chi
        if error < -tol:
            low = chi
        elif error > tol:
            high = chi
        elif np.abs(error) < tol:
            return chi

    if min_error < tol:
        return tentative_result
    else:
        return np.nan


def R_Q_FP(
    epsilon_sou, L_val, chi, f_adv, tol=0.001, allow_failed=False
):
    """Approximate fraction of samples needed to be honest in order to pass a target
    false positive rate, calculated using approximate CDF. In this setting, the
    adversary samples genuine quantum samples with fidelity phi1, and the rest are
    simulated classically.

    Args:
        epsilon_sou (float): soundness parameter
            (equals to the probability that adversary XEB passes)
        m (int): sample size (number of bitstrings)
        t (float): threshold of SQC to exceed
        t_threshold: time per quantum sample, which is the same
            as time adversary has for classical simulation
        B_val: validation cost per sample
        C: adversary computational power compared to validator

    Returns:
        float: approximate fraction of samples needed to be honest
    """
    chi = np.array([chi])

    R_Q = 0
    R = R_Q + f_adv
    ans = CDF(L_val, R, chi)
    if 1 - ans > epsilon_sou:
        return R_Q

    high = 1
    low = 0
    i = 0
    min_error = np.inf
    tentative_result = np.nan
    # If element is present at the middle itself
    while low <= high and i < 100:
        i += 1
        R_Q = (high + low) / 2
        # formula given in supplement Eq. V.1
        R = R_Q + f_adv
        ans = CDF(L_val, R, chi)
        if np.isnan(ans):
            print(L_val, R, chi)
            raise ValueError("CDF is nan")
        error = ((1 - ans) - epsilon_sou) / epsilon_sou
        if error > 0 and error < min_error:
            min_error = np.abs(error)
            tentative_result = R_Q
        if error < -tol:
            low = R_Q
        elif error > tol:
            high = R_Q
        elif np.abs(error) < tol:
            return R_Q

    if min_error < tol:
        return tentative_result
    else:
        if allow_failed:
            return R_Q
        else:
            return np.nan


def int_maximize(f, low, high):
    prev_high = None
    prev_low = None
    mid = (low + high) // 2
    low_val, mid_val, high_val = f(low), f(mid), f(high)
    while True:
        if low >= high - 1:
            if low_val > high_val:
                return low, low_val
            else:
                return high, high_val
        vals = [low_val, mid_val, high_val]
        max_id = np.nanargmax(vals)
        if max_id == 0 and prev_low == None:
            low = low
            high = mid
            high_val = mid_val
            mid = (low + mid) // 2
            mid_val = f(mid)
        elif max_id == 0 and prev_low != None:
            high = mid
            high_val = mid_val
            mid = low
            mid_val = low_val
            low = prev_low
            low_val = prev_low_val
        elif max_id == 2 and prev_high == None:
            low = mid
            high = high
            low_val = mid_val
            mid = (mid + high) // 2
            mid_val = f(mid)
        elif max_id == 2 and prev_high != None:
            low = mid
            low_val = mid_val
            mid = high
            mid_val = high_val
            high = prev_high
            high_val = prev_high_val
        else:
            prev_high = high
            prev_low = low
            prev_high_val = high_val
            prev_low_val = low_val
            high = (high + mid) // 2
            low = (low + mid) // 2
            high_val, low_val = f(high), f(low)


def Q_FP_exact_hypergeometric_chernoff(
    epsilon_sou, chi, L_val, L, PhiC, allow_negative=False
):
    """Minimum number of honest quantum samples needed to obtain XEB score with
    a false positive rate, calculated using exact CDF. In this setting, the adversary
    uses q quantum samples and simulates the rest classically

    Args:
        epsilon_sou (float): target soundness parameter
            (probability that adversary passes the XEB test)
        m (int): sample size (number of bitstrings)
        M (int): total number of bitstrings
        t (int): threshold XEB
        B_val (float): computational budget to simulate each circuit
        A (float): adversarial peak-flop power
        T_tot: total time needed for all M circuits

    Returns:
        float: exact number of quantum samples needed
    """
    high = L
    low = 0
    if allow_negative:
        low = -L
    previous_Q = None
    while True:
        Q = (high + low) // 2

        if Q == previous_Q:
            return Q
        previous_Q = Q

        # replace with Chernoff's bound
        exp_Lc = min(L - Q, L * PhiC)
        delta = np.sqrt(np.log(1 / (epsilon_sou / 2)) * (3 / exp_Lc))

        Lmax = Q + (1 + delta) * exp_Lc  # maximum number of PT samples in M
        try:
            Lmax = ceil(Lmax)
        except:
            print(
                f"Lmax: {Lmax}, Q: {Q}, delta: {delta}, exp_Lc: {exp_Lc}, M: {L}, Q: {Q}"
            )
            raise

        pmf = hypergeom.pmf(np.arange(L_val + 1), L, Lmax, L_val)
        ans = 1 - np.dot(pmf, P(np.arange(L_val + 1) + L_val, L_val * (chi + 1)))

        if ans < epsilon_sou / 2:
            low = Q
        else:
            high = Q


def optimize_L_val(
    pfail, epsilon_sou, phi, xi, L_val_min, L_val_max, n_points
):
    L_val_min = max(L_val_min, 1)

    def f(L_val):
        chi = get_chi(L_val, pfail, phi)
        f_adv = min(1, L_val / xi)
        result = R_Q_FP(
            epsilon_sou, L_val, chi, f_adv, allow_failed=True
        )
        return result

    if L_val_max - L_val_max > 200:
        tentative_RQs = [f(int(L_val)) for L_val in np.linspace(L_val_min, L_val_max, n_points)]
        print(tentative_RQs)
        idx = np.nanargmax(tentative_RQs)
        this_L_val_min = int(np.linspace(L_val_min, L_val_max, n_points)[max(idx - 1, 0)])
        this_L_val_max = int(
            np.linspace(L_val_min, L_val_max, n_points)[min(idx + 1, n_points - 1)]
        )
    else:
        this_L_val_min = L_val_min
        this_L_val_max = L_val_max
    

    L_val, R_Q = int_maximize(f, this_L_val_min, this_L_val_max)
    if L_val > L_val_max - 10 and R_Q > 0.005:
        print(
            f"m={L_val} is too close to the maximum, consider increasing the range. {phi}, {R_Q}, {L_val_min}, {L_val_max}"
        )
        raise ValueError
    if L_val < L_val_min + 10 and R_Q > 0.005:
        print(
            f"m={L_val} is too close to the minimum {L_val_min}, consider increasing the range. {phi}, {R_Q}, {L_val_min}, {L_val_max}"
        )
        raise ValueError
    return L_val, R_Q