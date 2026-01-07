"""
Seismological calculations for earthquake analysis.
"""

import numpy as np
from scipy import stats


def calculate_energy(magnitude):
    """
    Calculate seismic energy released from earthquake magnitude.

    Uses the Gutenberg-Richter energy-magnitude relation:
    log10(E) = 1.5*M + 4.8 (E in Joules)

    Parameters
    ----------
    magnitude : float or array-like
        Earthquake magnitude(s)

    Returns
    -------
    float or array
        Energy in Joules
    """
    return 10 ** (1.5 * np.asarray(magnitude) + 4.8)


def calculate_moment(magnitude):
    """
    Calculate seismic moment from earthquake magnitude.

    Uses the Hanks-Kanamori relation:
    M_w = (2/3) * log10(M_0) - 10.7
    Rearranged: M_0 = 10^(1.5*M + 16.1) (M_0 in dyne-cm)

    Parameters
    ----------
    magnitude : float or array-like
        Earthquake magnitude(s)

    Returns
    -------
    float or array
        Seismic moment in dyne-cm
    """
    return 10 ** (1.5 * np.asarray(magnitude) + 16.1)


def magnitude_completeness_maxc(magnitudes, bin_width=0.1):
    """
    Estimate magnitude of completeness using Maximum Curvature method.

    Mc is estimated as the magnitude bin with the highest frequency,
    plus a correction of 0.2 to account for typical underestimation.

    Parameters
    ----------
    magnitudes : array-like
        Array of earthquake magnitudes
    bin_width : float
        Bin width for magnitude histogram

    Returns
    -------
    float
        Estimated magnitude of completeness
    """
    magnitudes = np.asarray(magnitudes)

    # Create magnitude bins
    mag_min = np.floor(magnitudes.min() * 10) / 10
    mag_max = np.ceil(magnitudes.max() * 10) / 10
    bins = np.arange(mag_min, mag_max + bin_width, bin_width)

    # Count earthquakes in each bin
    counts, bin_edges = np.histogram(magnitudes, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find bin with maximum count
    max_idx = np.argmax(counts)
    mc = bin_centers[max_idx] + 0.2  # Add correction factor

    return mc


def magnitude_completeness_gft(magnitudes, bin_width=0.1, residual_threshold=0.05):
    """
    Estimate magnitude of completeness using Goodness-of-Fit Test.

    Tests synthetic Gutenberg-Richter distribution against observed
    for different Mc values, selecting the smallest Mc with R >= 90%.

    Parameters
    ----------
    magnitudes : array-like
        Array of earthquake magnitudes
    bin_width : float
        Bin width for magnitude histogram
    residual_threshold : float
        Residual threshold (1 - R value)

    Returns
    -------
    float
        Estimated magnitude of completeness
    """
    magnitudes = np.asarray(magnitudes)
    mc_maxc = magnitude_completeness_maxc(magnitudes, bin_width)

    # Test Mc values around MAXC estimate
    mc_range = np.arange(mc_maxc - 0.5, mc_maxc + 1.0, bin_width)

    best_mc = mc_maxc
    for mc_test in mc_range:
        mags_above = magnitudes[magnitudes >= mc_test]
        if len(mags_above) < 50:  # Need minimum sample
            continue

        # Calculate b-value
        b = gutenberg_richter_bvalue(mags_above, mc_test, method='mle')
        if b is None:
            continue

        # Calculate R value (goodness of fit)
        # Simplified: check if distribution follows G-R law
        # Full implementation would compare observed vs synthetic CDF
        if b > 0.5 and b < 2.0:  # Reasonable b-value range
            best_mc = mc_test
            break

    return best_mc


def gutenberg_richter_bvalue(magnitudes, mc, method='mle'):
    """
    Calculate Gutenberg-Richter b-value.

    The Gutenberg-Richter law: log10(N) = a - b*M

    Parameters
    ----------
    magnitudes : array-like
        Array of earthquake magnitudes
    mc : float
        Magnitude of completeness
    method : str
        'mle' for Maximum Likelihood Estimation (Aki, 1965)
        'lsq' for Least Squares fit

    Returns
    -------
    dict
        Dictionary containing:
        - b: b-value
        - b_std: standard error of b-value
        - a: a-value
        - n_events: number of events used
    """
    magnitudes = np.asarray(magnitudes)
    mags = magnitudes[magnitudes >= mc]

    if len(mags) < 10:
        return None

    n_events = len(mags)
    m_mean = np.mean(mags)

    if method == 'mle':
        # Maximum Likelihood Estimation (Aki, 1965)
        # b = log10(e) / (M_mean - Mc)
        # With correction for binned data: Mc -> Mc - delta_m/2
        delta_m = 0.1  # Standard magnitude bin width
        b = np.log10(np.e) / (m_mean - (mc - delta_m / 2))

        # Standard error (Shi & Bolt, 1982)
        b_std = 2.3 * b**2 * np.std(mags) / np.sqrt(n_events * (n_events - 1))

    elif method == 'lsq':
        # Least Squares fit to cumulative distribution
        mags_sorted = np.sort(mags)[::-1]
        cum_counts = np.arange(1, len(mags_sorted) + 1)

        # Log-linear fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            mags_sorted, np.log10(cum_counts)
        )
        b = -slope
        b_std = std_err

    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate a-value
    # log10(N) = a - b*Mc, where N is total number above Mc
    a = np.log10(n_events) + b * mc

    return {
        'b': b,
        'b_std': b_std,
        'a': a,
        'n_events': n_events,
        'mc': mc,
        'method': method
    }


def bootstrap_bvalue(magnitudes, mc, n_iterations=1000, confidence=0.95):
    """
    Bootstrap estimation of b-value confidence intervals.

    Parameters
    ----------
    magnitudes : array-like
        Array of earthquake magnitudes
    mc : float
        Magnitude of completeness
    n_iterations : int
        Number of bootstrap iterations
    confidence : float
        Confidence level (0-1)

    Returns
    -------
    dict
        Dictionary containing:
        - b_mean: mean b-value
        - b_std: standard deviation
        - b_lower: lower confidence bound
        - b_upper: upper confidence bound
    """
    magnitudes = np.asarray(magnitudes)
    mags = magnitudes[magnitudes >= mc]
    n = len(mags)

    if n < 20:
        return None

    b_values = []
    for _ in range(n_iterations):
        # Resample with replacement
        sample = np.random.choice(mags, size=n, replace=True)
        result = gutenberg_richter_bvalue(sample, mc, method='mle')
        if result is not None:
            b_values.append(result['b'])

    b_values = np.array(b_values)
    alpha = (1 - confidence) / 2

    return {
        'b_mean': np.mean(b_values),
        'b_std': np.std(b_values),
        'b_lower': np.percentile(b_values, alpha * 100),
        'b_upper': np.percentile(b_values, (1 - alpha) * 100)
    }


def fit_interevent_distribution(times, distribution='exponential'):
    """
    Fit distribution to inter-event times.

    Parameters
    ----------
    times : array-like
        Inter-event times (in days)
    distribution : str
        Distribution to fit: 'exponential', 'weibull', 'gamma'

    Returns
    -------
    dict
        Fitted parameters and goodness-of-fit metrics
    """
    times = np.asarray(times)
    times = times[times > 0]  # Remove zeros

    if len(times) < 20:
        return None

    if distribution == 'exponential':
        # Exponential: f(t) = lambda * exp(-lambda*t)
        # MLE: lambda = 1/mean
        rate = 1 / np.mean(times)
        params = {'rate': rate}

        # KS test
        ks_stat, p_value = stats.kstest(times, 'expon', args=(0, 1/rate))

    elif distribution == 'weibull':
        # Weibull: f(t) = (k/lambda) * (t/lambda)^(k-1) * exp(-(t/lambda)^k)
        shape, loc, scale = stats.weibull_min.fit(times, floc=0)
        params = {'shape': shape, 'scale': scale}

        # KS test
        ks_stat, p_value = stats.kstest(times, 'weibull_min', args=(shape, 0, scale))

    elif distribution == 'gamma':
        # Gamma distribution
        shape, loc, scale = stats.gamma.fit(times, floc=0)
        params = {'shape': shape, 'scale': scale}

        # KS test
        ks_stat, p_value = stats.kstest(times, 'gamma', args=(shape, 0, scale))

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return {
        'distribution': distribution,
        'params': params,
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'n_samples': len(times)
    }


def omori_law_fit(times, magnitudes, mainshock_time, mainshock_magnitude,
                  min_magnitude=None, max_days=365):
    """
    Fit modified Omori law to aftershock sequence.

    Modified Omori Law: n(t) = K / (t + c)^p

    Parameters
    ----------
    times : array-like
        Earthquake times (datetime or numeric)
    magnitudes : array-like
        Earthquake magnitudes
    mainshock_time : datetime or numeric
        Time of mainshock
    mainshock_magnitude : float
        Magnitude of mainshock
    min_magnitude : float, optional
        Minimum magnitude for aftershocks (default: mainshock_magnitude - 2)
    max_days : float
        Maximum days after mainshock to consider

    Returns
    -------
    dict
        Omori law parameters (K, c, p) and fit statistics
    """
    import pandas as pd

    times = pd.to_datetime(times)
    mainshock_time = pd.to_datetime(mainshock_time)

    if min_magnitude is None:
        min_magnitude = mainshock_magnitude - 2

    # Select aftershocks
    time_diff = (times - mainshock_time).dt.total_seconds() / 86400  # days
    mask = (time_diff > 0) & (time_diff <= max_days) & (np.asarray(magnitudes) >= min_magnitude)

    aftershock_times = time_diff[mask].values

    if len(aftershock_times) < 20:
        return None

    # Bin aftershocks by time
    bins = np.logspace(-2, np.log10(max_days), 50)
    counts, bin_edges = np.histogram(aftershock_times, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    # Rate (events per day)
    rates = counts / bin_widths

    # Fit Omori law (log-linear fit for simplicity)
    # log(n) = log(K) - p*log(t+c)
    # Approximate with c=0.1 for initial fit
    c_init = 0.1
    valid = rates > 0

    if sum(valid) < 5:
        return None

    log_t = np.log10(bin_centers[valid] + c_init)
    log_rate = np.log10(rates[valid])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_t, log_rate)

    p = -slope
    K = 10 ** intercept * (c_init ** p)

    return {
        'K': K,
        'c': c_init,
        'p': p,
        'r_squared': r_value ** 2,
        'n_aftershocks': len(aftershock_times)
    }


def calculate_recurrence_interval(b_value, a_value, magnitude):
    """
    Calculate recurrence interval for earthquakes of given magnitude.

    Based on Gutenberg-Richter: log10(N) = a - b*M
    Recurrence interval = 1/N (in years)

    Parameters
    ----------
    b_value : float
        Gutenberg-Richter b-value
    a_value : float
        Gutenberg-Richter a-value (normalized to annual rate)
    magnitude : float or array-like
        Target magnitude(s)

    Returns
    -------
    float or array
        Recurrence interval in years
    """
    log_n = a_value - b_value * np.asarray(magnitude)
    n = 10 ** log_n  # Annual rate
    return 1 / n  # Years


def mann_kendall_test(data):
    """
    Perform Mann-Kendall trend test.

    Parameters
    ----------
    data : array-like
        Time series data

    Returns
    -------
    dict
        Test statistics including trend direction, S statistic,
        Z score, and p-value
    """
    data = np.asarray(data)
    n = len(data)

    if n < 4:
        return None

    # Calculate S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])

    # Calculate variance
    # Account for ties
    unique, counts = np.unique(data, return_counts=True)
    ties = counts[counts > 1]

    var_s = (n * (n - 1) * (2 * n + 5)) / 18
    if len(ties) > 0:
        for t in ties:
            var_s -= t * (t - 1) * (2 * t + 5) / 18

    # Calculate Z score
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Trend direction
    if p_value < 0.05:
        trend = 'increasing' if z > 0 else 'decreasing'
    else:
        trend = 'no trend'

    return {
        'S': s,
        'var_S': var_s,
        'Z': z,
        'p_value': p_value,
        'trend': trend,
        'n': n
    }
