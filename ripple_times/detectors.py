from itertools import chain

import numpy as np
import pandas as pd

from .core import (exclude_close_events, filter_ripple_band,
                   gaussian_smooth, threshold_by_zscore)


def Kay_ripple_detector(time, LFPs, sampling_frequency, minimum_duration=0.015,
                        zscore_threshold=2.0, smoothing_sigma=0.004,
                        close_ripple_threshold=0.0):
    '''Find start and end times of sharp wave ripple events (150-250 Hz)
    based on Kay et al. 2016 [1].

    Parameters
    ----------
    time : array_like, shape (n_time,)
    LFPs : array_like, shape (n_time, n_signals)
        Time series of electric potentials   
    sampling_frequency : float
        Number of samples per second.   
    minimum_duration : float, optional
        Minimum time the z-score has to stay above threshold to be
        considered a ripple. The default is given assuming time is in
        units of seconds.
    zscore_threshold : float, optional
        Number of standard deviations the ripple power must exceed to
        be considered a ripple
    smoothing_sigma : float, optional
        Amount to smooth the time series over time. The default is
        given assuming time is in units of seconds.
    close_ripple_threshold : float, optional
        Exclude ripples that occur within `close_ripple_threshold` of a
        previously detected ripple.

    Returns
    -------
    ripple_times : pandas DataFrame

    References
    ----------
    .. [1] Kay, K., Sosa, M., Chung, J.E., Karlsson, M.P., Larkin, M.C.,
    and Frank, L.M. (2016). A hippocampal network for spatial coding during
    immobility and sleep. Nature 531, 185-190.

    '''
    not_null = np.all(pd.notnull(LFPs), axis=1) 
    LFPs, time = LFPs[not_null], time[not_null]

    filtered_lfps = np.stack(
        [filter_ripple_band(lfp, sampling_frequency) for lfp in LFPs.T])
    combined_filtered_lfps = np.sum(filtered_lfps ** 2, axis=0)
    combined_filtered_lfps = gaussian_smooth(
        combined_filtered_lfps, smoothing_sigma, sampling_frequency)
    combined_filtered_lfps = np.sqrt(combined_filtered_lfps)
    candidate_ripple_times = threshold_by_zscore(
        combined_filtered_lfps, time, minimum_duration, zscore_threshold)
    ripple_times = exclude_close_events(
        candidate_ripple_times, close_ripple_threshold)
    index = pd.Index(np.arange(len(ripple_times)) + 1,
                     name='ripple_number')
    return filtered_lfps, pd.DataFrame(ripple_times, columns=['start_time', 'end_time'],
                        index=index)


