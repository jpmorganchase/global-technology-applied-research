###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
from math import ceil, floor, log2


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
            low_val = prev_low_val  # I think this is undefined
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
            high_val = prev_high_val  # I think this is undefined
        else:
            prev_high = high
            prev_low = low
            prev_high_val = high_val
            prev_low_val = low_val
            high = (high + mid) // 2
            low = (low + mid) // 2
            high_val, low_val = f(high), f(low)

# ------- Raz's extractor calculations -------


def log2_error_raz(n_1: int,
                   k_1: float,
                   k_2: float,
                   m: int,
                   l: int,
                   p: int):
    """
    Compute an upper bound on the logarithm base 2
    of the error for the efficient weak version of
    Raz's extractor presented in [Fore2025]_.

    Parameters
    ----------
    n_1 : int
        The length of the first input (in bits).
    k_1 : float
        The min-entropy of the first input.
    k_2 : float
        The min-entropy of the second input.
    m : int
        The length of the extractor output (in bits).
    l : int
        The logarithm base 2 of p'.
        **Note:** the efficient construction requires p'
        to be a power of 2, i.e. l is an integer.
    p : int
        A free parameter that is an even integer that
        satisfies p <= 2^l / m.

    Returns
    -------
    float :
        An upper bound on the logarithm base 2 of the
        extractor error.
    """
    log_gamma_bound = (n_1 - k_1)/p + max((l - n_1/2 + 1)/p,
                                          np.log2(p) - k_2/2) + 1
    return log_gamma_bound + m/2


def opt_error_raz(n_1: int,
                  k_1: float,
                  n_2: int,
                  k_2: float,
                  m: int,
                  max_tests_basic=1,
                  max_tests_detailed=1000,
                  detailed_opt=False, verbose=False):
    """
    Compute an upper bound on the logarithm base 2
    of the error for the efficient weak version of
    Raz's extractor presented in [Fore2025]_.

    Parameters
    ----------
    n_1 : int
        The length of the first input (in bits).
    k_1 : float
        The min-entropy of the first input.
    n_2 : int
        The length of the second input (in bits).
    k_2 : float
        The min-entropy of the second input.
    m : int
        The length of the extractor output (in bits).
    max_tests_basic : int
        The maximum number of interations for the
        basic parameter optimisation method, i.e. when
        detailed_opt is set to False (default: 0).
    max_tests_detailed : int
        The maximum number of interations for the
        intense parameter optimisation method, i.e.
        when detailed_opt is set to True (default: 1000).
    detailed_opt : bool
        Flag to indicate the intensity of the
        optimisation performed (default: False).
    verbose : bool
        If True, prints all parameters found (default: False).

    Returns
    -------
    float :
        An upper bound on the logarithm base 2 of the
        extractor error.
    """
    # Ensure input parameters meet required constraints.
    assert 0 < n_2 <= n_1 / 2
    assert 0 <= k_1 < n_1 and 0 <= k_2 < n_2
    assert 0 < m <= n_1 / 2
    assert n_1 % 2 == 0
    assert max_tests_basic > 0
    assert max_tests_detailed > 0

    # Compute maximum possible l value based on input parameters.
    l_max = int(n_2 + floor(log2(n_1 / 2)))

    # Cap exponent to avoid overflow in 2^x computations.
    max_pow_for_overflow = 32

    # Initialize variables to track the best (minimal)
    # log2 error and corresponding parameters.
    min_log2_error, best_l, best_p = 0, 'Not found', 'Not found'

    # Estimate a good initial value for l based on m and (n_1 - k_1).
    l_use = max(floor(log2(m * (n_1 - k_1))), 1)

    # Define the range of l values to explore around l_use.
    max_plus = min(floor(l_max - l_use),
                   (max_tests_basic - 1) // 2)
    max_minus = min(floor(l_use - ceil(log2(m)) - 1),
                    (max_tests_basic - 1) // 2)
    ls = [i for i in range(l_use - max_minus, l_use + max_plus + 1)]

    # Coarse search: iterate over candidate l values.
    for current_l in ls:
        # Compute the maximum number of p values to try for the current l.
        p_half_max = int((2**(current_l - log2(m)))//2)
        max_ps = p_half_max  # Enables the option to restrict the number of
        # candidate p values to test, e.g., max_ps = min(100000, p_half_max).

        # Compute step size to space evenly over the range [0, p_half_max - 1]
        step = (p_half_max - 1) / (max_ps - 1) if max_ps > 1 else 0

        # Generate evenly spaced p_values directly
        # p_values = [2 * int(round(i * step)) + 2 for i in range(max_ps)]

        # for current_p in p_values:
        #     # Evaluate the log2 of the error for current parameters.
        #     eps = log2_error_raz(n_1, k_1, k_2, m, current_l, current_p)
        #     # Update best found parameters if error improves.
        #     if eps < min_log2_error:
        #         min_log2_error, best_l, best_p = eps, current_l, current_p

        def f(current_p):
            return - log2_error_raz(n_1, k_1, k_2, m, current_l, current_p)
        _, negative_min_log2_error = int_maximize(f, 2, 2 * int(round((max_ps - 1) * step)) + 2)
        min_log2_error = - negative_min_log2_error

    # If detailed optimisation is enabled, perform a more exhaustive search.
    if detailed_opt:
        # Generate a list of l values to try with finer granularity.
        num_values = min(l_max, max_tests_detailed)
        step_size = (l_max - (ceil(log2(m)) + 1)) / (num_values - 1)
        ls = [
            int(round((ceil(log2(m)) + 1) + i * step_size)
                ) for i in range(num_values)]
        total = len(ls)
        # Define progress milestones for verbose output.
        milestones = {int(total * p / 100) for p in range(10, 101, 10)}
        for i, current_l in enumerate(ls):
            # Prevent overflow when computing p values.
            if current_l - log2(m) - 1 < max_pow_for_overflow:
                p_half_max = int(2**(current_l - log2(m) - 1))
            else:
                p_half_max = int(floor(2 ** max_pow_for_overflow))
            # Sample p_half values uniformly for detailed testing.
            num_values = min(p_half_max, max_tests_detailed)
            step_size = (p_half_max - 1) / num_values
            phalf_values = [
                int(round(1 + i * step_size)
                    ) for i in range(num_values + 1)]

            for current_phalf in phalf_values:
                current_p = 2*current_phalf
                eps = log2_error_raz(n_1, k_1, k_2, m, current_l, current_p)
                # Update best found parameters if error improves.
                if eps < min_log2_error:
                    min_log2_error, best_l, best_p = eps, current_l, current_p
            # Print progress if enabled and at a milestone.
            if verbose and i in milestones:
                percent = int((i / total) * 100)
                print(f'[{percent}% Completed] ({i}/{total})')

        if verbose:
            print(f'[100% Completed] ({total}/{total})')
            print('Performing final refinement...')

        # Final refinement around best_l after detailed search.
        if best_l != 'Not found':
            # Compute the range of l values to explore around best_l.
            max_plus = min(floor(l_max - best_l), max_tests_detailed // 2)
            max_minus = min(floor(best_l - ceil(log2(m)) - 1),
                            max_tests_detailed // 2)
            # Generate candidate l values around best_l.
            ls = [i for i in range(best_l - max_minus, best_l + max_plus + 1)]
            # Iterate over candidate l values for final refinement.
            for current_l in ls:
                # Prevent overflow when computing p values.
                if current_l - log2(m) - 1 < max_pow_for_overflow:
                    p_half_max = int(2**(current_l - log2(m) - 1))
                else:
                    p_half_max = int(floor(2 ** max_pow_for_overflow))
                # Sample p_half values uniformly for detailed testing.
                num_values = min(p_half_max, max_tests_detailed)
                step_size = (p_half_max - 1) / num_values
                phalf_values = [
                    int(round(1 + i * step_size)
                        ) for i in range(num_values + 1)]
                # Iterate over candidate p values.
                for current_phalf in phalf_values:
                    current_p = 2*current_phalf
                    eps = log2_error_raz(n_1,
                                         k_1,
                                         k_2,
                                         m,
                                         current_l,
                                         current_p)
                    if eps < min_log2_error:
                        min_log2_error = eps
                        best_l, best_p = current_l, current_p
    if verbose:
        print(f'log2 of minimum error found: {min_log2_error}')
        print(f'Corresponding l value (p prime = 2^l): {best_l}')
        print(f'Corresponding p value: {best_p}')
    # Return the minimal log2 error found.
    return min_log2_error


def calc_raz_out(n_1: int,
                 k_1: float,
                 n_2: int,
                 k_2: float,
                 log2_error_tol: float,
                 m_init=1,
                 m_tests=100,
                 max_tests_basic=1,
                 max_tests_detailed=1000,
                 detailed_opt=False,
                 verbose=False):
    """
    Compute the maximum output length for the efficient
    weak version of Raz's extractor presented in
    [Fore2025]_ that satisfies a given error tolerance.

    Parameters
    ----------
    n_1 : int
        The length of the first input (in bits).
    k_1 : float
        The min-entropy of the first input.
    n_2 : int
        The length of the second input (in bits).
    k_2 : float
        The min-entropy of the second input.
    log2_error_tol : float
        The logarithm base 2 of the acceptable extractor error.
        Must be negative, as the extractor error is 2^log2_error.
    m_init : int
        The initial value for the output length to test (default: 1).
    m_tests : int
        The number of output lengths to test (default: 100).
    max_tests_basic : int
        The maximum number of interations for the
        basic parameter optimisation method, i.e. when
        detailed_opt is set to False (default: 0).
    max_tests_detailed : int
        The maximum number of interations for the
        intense parameter optimisation method, i.e.
        when detailed_opt is set to True (default: 1000).
    detailed_opt : bool
        Flag to indicate the intensity of the
        optimisation performed (default: False).
    verbose : bool
        If True, prints all parameters found (default: False).

    Returns
    -------
    int :
        The maximum output length that satisfies the
        error tolerance.
    """
    max_m = 0
    ms = np.linspace(m_init, floor(k_2),
                     min(floor(k_2) - m_init,
                         m_tests), dtype=int)
    i = 0
    STOP = 0
    while STOP == 0:
        m = ms[i]
        if opt_error_raz(n_1, k_1, n_2, k_2, m,
                         max_tests_basic=max_tests_basic,
                         max_tests_detailed=max_tests_detailed,
                         detailed_opt=detailed_opt,
                         verbose=False) <= log2_error_tol:
            max_m = m
        else:
            STOP = 1
        i += 1
        if i >= len(ms):
            STOP = 1
    if detailed_opt:
        max_m += 1
        while opt_error_raz(n_1, k_1, n_2, k_2, max_m,
                            max_tests_basic=max_tests_basic,
                            max_tests_detailed=max_tests_detailed,
                            detailed_opt=detailed_opt,
                            verbose=False) <= log2_error_tol:
            max_m += 1
        max_m -= 1
    if verbose:
        print(f'Maximum output length found: {max_m}')
        opt_error_raz(n_1, k_1, n_2, k_2, max_m,
                      max_tests_basic=max_tests_basic,
                      max_tests_detailed=max_tests_detailed,
                      detailed_opt=detailed_opt,
                      verbose=True)
    if max_m > n_1 / 2:
        max_m = n_1 / 2
    return max_m

# ----- STRONG + QPROOF -----


def calc_raz_out_strong(n_1: int,
                        k_1: float,
                        n_2: int,
                        k_2: float,
                        log2_error_tol: float,
                        m_init=1,
                        m_tests=100,
                        max_tests_basic=1,
                        max_tests_detailed=1000,
                        detailed_opt=False,
                        verbose=False):
    """
    Compute the maximum output length for the efficient
    strong version of Raz's extractor
    presented in [Fore2025]_ that satisfies a
    given error tolerance.

    Parameters
    ----------
    n_1 : int
        The length of the first input (in bits).
    k_1 : float
        The min-entropy of the first input.
    n_2 : int
        The length of the second input (in bits).
    k_2 : float
        The min-entropy of the second input.
    log2_error_tol : float
        The logarithm base 2 of the acceptable extractor error.
        Must be negative, as the extractor error is 2^log2_error.
        m_init : int
        The initial value for the output length to test (default: 1).
    m_tests : int
        The number of output lengths to test (default: 100).
    max_tests_basic : int
        The maximum number of interations for the
        basic parameter optimisation method, i.e. when
        detailed_opt is set to False (default: 0).
    max_tests_detailed : int
        The maximum number of interations for the
        intense parameter optimisation method, i.e.
        when detailed_opt is set to True (default: 1000).
    detailed_opt : bool
        Flag to indicate the intensity of the
        optimisation performed (default: False).
    verbose : bool
        If True, prints all parameters found (default: False).

    Returns
    -------
    int :
        The maximum output length that satisfies the
        error tolerance.
    """
    assert 0 <= k_1 <= n_1 and 0 <= k_2 <= n_2
    k_1_init = k_1
    k_2_init = k_2
    max_m = 0
    ms = np.linspace(m_init, floor(k_2),
                     min(floor(k_2) - m_init,
                         m_tests), dtype=int)
    STOP = 0
    i = 0
    while STOP == 0:
        m = ms[i]
        raz_error = log2_error_tol - 1
        k_1 = max(k_1_init - m/2 - 2 + (log2_error_tol - m/2), 0)
        k_2 = max(k_2_init - m/2 - 2 + (log2_error_tol - m/2), 0)
        if opt_error_raz(n_1, k_1, n_2, k_2, m,
                         max_tests_basic=max_tests_basic,
                         max_tests_detailed=max_tests_detailed,
                         detailed_opt=detailed_opt,
                         verbose=False) <= raz_error:
            max_m = m
        else:
            STOP = 1
        i += 1
        if i >= len(ms):
            STOP = 1
    if verbose:
        print(f'Maximum output length found: {max_m}')
        opt_error_raz(n_1, k_1, n_2, k_2, max_m,
                      max_tests_basic=max_tests_basic,
                      max_tests_detailed=max_tests_detailed,
                      detailed_opt=detailed_opt,
                      verbose=True)
    if max_m > n_1 / 2:
        max_m = n_1 / 2
    return max_m


def calc_raz_out_strong_qproof(n_1: int,
                               k_1: float,
                               n_2: int,
                               k_2: float,
                               log2_error_tol: float,
                               m_init=1,
                               m_tests=100,
                               max_tests_basic=1,
                               max_tests_detailed=1000,
                               detailed_opt=False,
                               verbose=False):
    """
    Compute the maximum output length for the efficient
    strong and quantum-proof version of Raz's extractor
    presented in [Fore2025]_ that satisfies a
    given error tolerance.

    Parameters
    ----------
    n_1 : int
        The length of the first input (in bits).
    k_1 : float
        The min-entropy of the first input.
    n_2 : int
        The length of the second input (in bits).
    k_2 : float
        The min-entropy of the second input.
    log2_error_tol : float
        The logarithm base 2 of the acceptable extractor error.
        Must be negative, as the extractor error is 2^log2_error.
    m_init : int
        The initial value for the output length to test (default: 1).
    m_tests : int
        The number of output lengths to test (default: 100).
    max_tests_basic : int
        The maximum number of interations for the
        basic parameter optimisation method, i.e. when
        detailed_opt is set to False (default: 0).
    max_tests_detailed : int
        The maximum number of interations for the
        intense parameter optimisation method, i.e.
        when detailed_opt is set to True (default: 1000).
    detailed_opt : bool
        Flag to indicate the intensity of the
        optimisation performed (default: False).
    verbose : bool
        If True, prints all parameters found (default: False).

    Returns
    -------
    int :
        The maximum output length that satisfies the
        error tolerance.
    """
    assert 0 < k_1 <= n_1 and 0 < k_2 <= n_2
    k_1_init = k_1
    k_2_init = k_2
    max_m = 0
    ms = np.linspace(m_init, floor(k_2),
                     min(floor(k_2) - m_init,
                         m_tests), dtype=int)
    STOP = 0
    i = 0
    while STOP == 0:
        m = ms[i]
        raz_error = 2 - log2(3) - m + 2 * log2_error_tol
        k_1 = max(k_1_init - 1 + 2 * (raz_error), 0)
        k_2 = max(k_2_init - 1 + 2 * (raz_error), 0)
        if opt_error_raz(n_1, k_1, n_2, k_2, m,
                         max_tests_basic=max_tests_basic,
                         max_tests_detailed=max_tests_detailed,
                         detailed_opt=detailed_opt,
                         verbose=False) <= raz_error:
            max_m = m
        else:
            STOP = 1
        i += 1
        if i >= len(ms):
            STOP = 1

    if verbose:
        print(f'Maximum output length found: {max_m}')
        opt_error_raz(n_1, k_1, n_2, k_2, max_m,
                      max_tests_basic=max_tests_basic,
                      max_tests_detailed=max_tests_detailed,
                      detailed_opt=detailed_opt,
                      verbose=True)
    if max_m > n_1 / 2:
        max_m = n_1 / 2
    return max_m


if __name__ == "__main__":
    # Minimise the min-entropy of the weak source of randomness
    # subject to obtaining at-least 4093 bits of output after
    # applying our efficiently implemented Raz' two-source extractor.

    n_qc = 56 * 30010   # Length of the input to the 2-source extractor
    # from the quantum computer.
    k_qc = 0.04 * 56 * 30010  # Min-entropy of the input from the
    # quantum computer.

    e_ext = 10**-6  # Error of the extractor, i.e. the distance bound.
    log2_error = np.log2(e_ext)  # Logarithm base 2 of the extractor error.

    # Setting for the Raz extractor:
    strong = True
    qproof = True
    m_raz_min = 4093  # Minimum output length required from the extractor.

    # Currently supported field sizes for our efficient Raz's extractor:
    valid_n1half = [
        3, 7, 15, 31, 63, 127, 255, 521, 1279, 2281, 3217, 4423, 23209,
        44497, 110503, 132049, 756839, 859433, 3021377, 6972593, 24036583,
        25964951, 30402457, 32582657, 42643801, 43112609, 74207281
    ]
    # Set optimisation parameters.
    STOP = False
    inc = 0.005  # Increment for the min-entropy rate.

    # Two cases: when the min-entropy rate of the quantum computer input
    # is greater than 0.5, and when it is less than or equal to 0.5.
    n_1_prelim = 2 * min(x for x in valid_n1half if 2 * x >= n_qc)
    delta = 0.02  # Tolerance for the min-entropy rate 'check'
    # (i.e. 0.5 + delta is the min-entropy rate when the QC source
    # becomes the 'good' one). This will depend on how close n_qc
    # is to a supported input size. For example, with the above
    # parameters, if the rate of the qc is 0.53, a weaker min-entropy
    # source can amplified when it is still considered the second source.
    if k_qc / n_1_prelim > 0.5 + delta:
        # The QC source is the 'good' source:
        n_1 = n_1_prelim  # Adjusted length of the QC input to the extractor.
        k_1 = k_qc  # Min-entropy of the input from the quantum computer.
        n_2 = n_1_prelim / 2
        alpha_wsr = 0.005  # Initial min-entropy rate assumption of the weak
        # source of randomness. Note that the first test is this + inc.
        while not STOP:
            alpha_wsr += inc  # Increment the min-entropy rate.
            k_2 = np.floor(alpha_wsr * n_2)
            if strong and not qproof:
                m_raz = calc_raz_out_strong(n_1, k_1, n_2, k_2, log2_error,
                                            m_init=m_raz_min, m_tests=1)
                if m_raz >= m_raz_min:
                    print('Found a suitable min-entropy rate:', alpha_wsr)
                    STOP = True
            if strong and qproof:
                m_raz = calc_raz_out_strong_qproof(n_1, k_1, n_2, k_2,
                                                   log2_error,
                                                   m_init=m_raz_min,
                                                   m_tests=1)
                if m_raz >= m_raz_min:
                    print('Found a suitable min-entropy rate:', alpha_wsr)
                    STOP = True
            if not strong and not qproof:
                m_raz = calc_raz_out(n_1, k_1, n_2, k_2,
                                     log2_error,
                                     m_init=m_raz_min,
                                     m_tests=1)
                if m_raz >= m_raz_min:
                    print('Found a suitable min-entropy rate:', alpha_wsr)
                    STOP = True
            if alpha_wsr > 1:
                print('No suitable min-entropy rate found.')
                STOP = True
    else:
        # The QC source is the 'bad' source:
        n_1half = min(x for x in valid_n1half if x >= n_qc)
        n_1 = 2 * n_1half  # Length of the extractor input from the
        # weak source of randomness (in bits).
        # NOTE: We might want to use the maximum n_1half, since this reduces
        # the min-entropy rate required from the weak source of randomness.
        # To do this, simply replace n_1half = ... with n_1half = 74207281.
        n_2 = n_qc
        k_2 = k_qc
        # Compute the mininimum min-entropy rate of the weak source
        # of randomness required to obtain at least m_raz_min bits
        # of output from the extractor.
        alpha_wsr = 0.5  # Initial min-entropy rate assumption of the weak
        # source of randomness. Note that the first test is this + inc.
        while not STOP:
            alpha_wsr += inc  # Increment the min-entropy rate.
            k_1 = np.floor(alpha_wsr * n_1)
            if strong and not qproof:
                m_raz = calc_raz_out_strong(n_1, k_1, n_2, k_2,
                                            log2_error,
                                            m_init=m_raz_min,
                                            m_tests=1)
                if m_raz >= m_raz_min:
                    print('Found a suitable min-entropy rate:', alpha_wsr)
                    STOP = True
            if strong and qproof:
                m_raz = calc_raz_out_strong_qproof(n_1, k_1, n_2, k_2,
                                                   log2_error,
                                                   m_init=m_raz_min,
                                                   m_tests=1)
                if m_raz >= m_raz_min:
                    print('Found a suitable min-entropy rate:', alpha_wsr)
                    STOP = True
            if not strong and not qproof:
                m_raz = calc_raz_out(n_1, k_1, n_2, k_2, log2_error,
                                     m_init=m_raz_min, m_tests=1)
                if m_raz >= m_raz_min:
                    print('Found a suitable min-entropy rate:', alpha_wsr)
                    STOP = True
            if alpha_wsr > 1:
                print('No suitable min-entropy rate found.')
                STOP = True