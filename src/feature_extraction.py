'''
Feature Extraction (Feature Engineering) code for Classic Machine Learning Approach.

Otsuka et al., (2024) Methods in Ecology and Evolution
"Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers"

'''

import numpy as np
# import pandas as pd
import polars as pl
from scipy import stats
from sklearn.linear_model import LinearRegression


def calc_static_and_dynamic_components(df, 
                                       sampling_rate=25, 
                                       rolling_window_sec=2):
    '''
    Args:
        df (DataFrame): preprocessed data
        sampling_rate (int): sampling rate of acceleration sensor
        rolling_window_sec (int): window length in seconds for rolling calculation
    Returns:
        df (DataFrame): data frame with static and dynamic components, 
                        as well as pitch, roll, and VeDBA
    '''
    
    # Static Components
    # window_size = int(sampling_rate*rolling_window_sec)
    window_size = sampling_rate*rolling_window_sec
    
    # Pandas
    # df['acc_x_st'] = df['acc_x'].rolling(window=window_size, center=True).mean()
    # df['acc_y_st'] = df['acc_y'].rolling(window=window_size, center=True).mean()
    # df['acc_z_st'] = df['acc_z'].rolling(window=window_size, center=True).mean()
    
    # Polars | much faster than than pandas
    x_st = np.array(df['acc_x'].rolling_mean(
        window_size=window_size, center=True))
    y_st = np.array(df['acc_y'].rolling_mean(
        window_size=window_size, center=True))
    z_st = np.array(df['acc_z'].rolling_mean(
        window_size=window_size, center=True))
    
    # Pitch & Roll
    # https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html
    # https://watako-lab.com/2019/02/15/3axis_acc/
    # https://garchiving.com/angle-from-acceleration/
    # pitch
    pitch_radian = -1 * np.arctan( (x_st) / np.sqrt(y_st**2 + z_st**2) )
    pitch = pitch_radian * (180 / np.pi)
    # roll
    roll_radian = np.arctan( y_st / z_st )
    roll = roll_radian * (180 / np.pi)
    
    # Dynamic Components 
    # dynamic component = (raw signal) - (static component)
    x_dy = np.array(df['acc_x']) - x_st
    y_dy = np.array(df['acc_y']) - y_st
    z_dy = np.array(df['acc_z']) - z_st
    
    # VeDBA (Vectorized Dynamic Body Acceleration)
    VeDBA = np.sqrt( x_dy*x_dy + y_dy*y_dy + z_dy*z_dy )
    
    # add new_columns (in polars's way)
    df = df.with_columns(
        [
            pl.Series(x_st).alias("acc_x_st"),
            pl.Series(y_st).alias("acc_y_st"),
            pl.Series(z_st).alias("acc_z_st"),
            pl.Series(pitch).alias("pitch"),
            pl.Series(roll).alias("roll"),
            pl.Series(x_dy).alias("acc_x_dy"),
            pl.Series(y_dy).alias("acc_y_dy"),
            pl.Series(z_dy).alias("acc_z_dy"),
            pl.Series(VeDBA).alias("VeDBA"),
        ]
    )
    
    return df


def calc_features_for_one_sliding_window(window_tmp):
    ''' Feature Extraction (Calculation)
    Args:
        window_tmp (numpy.ndarray): a window
    Returns:
        feature_list (list): extracted features of this window as a list
    '''
    
    # data in this window
    # datetime, unixtime (column 0, 1)
    x = window_tmp[:, 2].astype(np.float64) # np.array(df['acc_x'])
    y = window_tmp[:, 3].astype(np.float64) # np.array(df['acc_y'])
    z = window_tmp[:, 4].astype(np.float64) # np.array(df['acc_z'])
    # label, label_id (column 5, 6)
    x_st = window_tmp[:, 7].astype(np.float64) # np.array(df['acc_x_st'])
    y_st = window_tmp[:, 8].astype(np.float64) # np.array(df['acc_y_st'])
    z_st = window_tmp[:, 9].astype(np.float64) # np.array(df['acc_z_st'])
    pitch = window_tmp[:, 10].astype(np.float64) # np.array(df['pitch'])
    roll = window_tmp[:, 11].astype(np.float64) # np.array(df['roll'])
    x_dy = window_tmp[:, 12].astype(np.float64) # np.array(df['acc_x_dy'])
    y_dy = window_tmp[:, 13].astype(np.float64) # np.array(df['acc_y_dy'])
    z_dy = window_tmp[:, 14].astype(np.float64) # np.array(df['acc_z_dy'])
    VeDBA = window_tmp[:, 15].astype(np.float64) # np.array(df['VeDBA'])

    # mean
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    
    # variance
    x_var = np.var(x)
    y_var = np.var(y)
    z_var = np.var(z)
    
    # standard deviation
    x_std = np.std(x)
    y_std = np.std(y)
    z_std = np.std(z)
    
    # coefficient of variance
    x_cv = x_std / x_mean * 100
    y_cv = y_std / y_mean * 100
    z_cv = z_std / z_mean * 100
    
    # skewness
    x_skew = stats.skew(x)
    y_skew = stats.skew(y)
    z_skew = stats.skew(z)
    
    # kurtosis
    x_kurtosis = stats.kurtosis(x)
    y_kurtosis = stats.kurtosis(y)
    z_kurtosis = stats.kurtosis(z)
    
    # maximum
    x_max = np.max(x)
    y_max = np.max(y)
    z_max = np.max(z)
    
    # minimum
    x_min = np.min(x)
    y_min = np.min(y)
    z_min = np.min(z)

    # range
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # 25% quantile
    x_q25 = np.quantile(x, q=0.25)
    y_q25 = np.quantile(y, q=0.25)
    z_q25 = np.quantile(z, q=0.25)
    
    # 50% quantile
    x_q50 = np.quantile(x, q=0.50) # np.median(x)
    y_q50 = np.quantile(y, q=0.50) # np.median(y)
    z_q50 = np.quantile(z, q=0.50) # np.median(z)
    
    # 75% quantile
    x_q75 = np.quantile(x, q=0.75)
    y_q75 = np.quantile(y, q=0.75)
    z_q75 = np.quantile(z, q=0.75)

    # magnitude: difference between magnitude and VeDBA -> Raw data or Dynamic components (x, y, z)
    magnitude = np.sqrt(x*x + y*y + z*z)
    mag_mean = np.mean(magnitude)
    mag_var = np.var(magnitude)
    mag_skew = stats.skew(magnitude)
    mag_kurtosis = stats.kurtosis(magnitude)
    mag_rms = np.sqrt(np.mean(magnitude**2))

    # norm
    x_norm = np.sqrt(np.sum(np.square(x)))
    y_norm = np.sqrt(np.sum(np.square(y)))
    z_norm = np.sqrt(np.sum(np.square(z)))

    # autocorrelation time lag of 1 data point
    x_ac = calc_auto_correlation_v2(x, time_lag=1)[1]
    y_ac = calc_auto_correlation_v2(y, time_lag=1)[1]
    z_ac = calc_auto_correlation_v2(z, time_lag=1)[1]

    # trend
    x_trend = calc_trend(x)
    y_trend = calc_trend(y)
    z_trend = calc_trend(z)
    mag_trend = calc_trend(magnitude)

    # covariance
    cov_xy = np.cov(x, y, ddof=0)[0][1] # x.cov(y)
    cov_yz = np.cov(y, z, ddof=0)[0][1] # y.cov(z)
    cov_zx = np.cov(z, x, ddof=0)[0][1] # z.cov(x)

    # correlation
    corr_xy = np.corrcoef(x, y)[0][1] # x.corr(y)
    corr_yz = np.corrcoef(y, z)[0][1] # y.corr(z)
    corr_zx = np.corrcoef(z, x)[0][1] # z.corr(x)

    # diff mean
    diff_xy_mean = np.mean(np.abs(x - y))
    diff_yz_mean = np.mean(np.abs(y - z))
    diff_zx_mean = np.mean(np.abs(z - x))

    # diff std
    diff_xy_std = np.std(np.abs(x - y))
    diff_yz_std = np.std(np.abs(y - z))
    diff_zx_std = np.std(np.abs(z - x))

    # static components mean, variance
    x_st_mean = np.mean(x_st)
    y_st_mean = np.mean(y_st)
    z_st_mean = np.mean(z_st)
    x_st_var = np.var(x_st)
    y_st_var = np.var(y_st)
    z_st_var = np.var(z_st)
    
    # pitch
    pitch_mean = np.mean(pitch)
    pitch_var = np.var(pitch)
    
    # roll
    roll_mean = np.mean(roll)
    roll_var = np.var(roll)

    # mean, variance, max of Dynamic Body Acceleration of x, y, and z 
    x_dy_mean = np.mean(x_dy)
    y_dy_mean = np.mean(y_dy)
    z_dy_mean = np.mean(z_dy)
    x_dy_var = np.var(x_dy)
    y_dy_var = np.var(y_dy)
    z_dy_var = np.var(z_dy)
    x_dy_max = np.max(x_dy)
    y_dy_max = np.max(y_dy)
    z_dy_max = np.max(z_dy)

    # Partial Dynamic Body Acceleration
    x_PDBA = np.abs(x_dy)
    y_PDBA = np.abs(y_dy)
    z_PDBA = np.abs(z_dy)
    x_PDBA_mean = np.mean(x_PDBA)
    y_PDBA_mean = np.mean(y_PDBA)
    z_PDBA_mean = np.mean(z_PDBA)

    # Overall Dynamic Body Acceleration mean, variance, max
    # https://besjournals.onlinelibrary.wiley.com/doi/10.1111/j.1365-2656.2006.01127.x
    ODBA = np.abs(x_dy) + np.abs(y_dy) + np.abs(z_dy)
    ODBA_mean = np.mean(ODBA)
    ODBA_var = np.var(ODBA)
    ODBA_max = np.max(ODBA)

    # VeDBA (Vectorized Dynamic Body Acceleration) mean
    # https://animalbiotelemetry.biomedcentral.com/articles/10.1186/s40317-017-0121-3
    VeDBA_mean = np.mean(VeDBA)
    VeDBA_var = np.var(VeDBA)
    VeDBA_max = np.max(VeDBA)

    # Smoothed VeDBA within a window -> Running mean
    VeDBA_s = np.convolve(VeDBA, np.ones(5)/5, mode='same')
    VeDBA_s_mean = np.mean(VeDBA_s)

    # Ratio VeDBA to PDBA
    x_ratio_VeDBA_to_PDBA_mean = np.mean(VeDBA / x_PDBA)
    y_ratio_VeDBA_to_PDBA_mean = np.mean(VeDBA / y_PDBA)
    z_ratio_VeDBA_to_PDBA_mean = np.mean(VeDBA / z_PDBA)
   
    # difference of continuous points
    (
        x_dcp, x_dcp_mean, x_dcp_std
    ) = calc_difference_of_continuous_points(x)
    (
        y_dcp, y_dcp_mean, y_dcp_std
    ) = calc_difference_of_continuous_points(y)
    (
        z_dcp, z_dcp_mean, z_dcp_std
    ) = calc_difference_of_continuous_points(z)
    
    # main frequency and amplitude
    (
        x_main_freq_1, x_main_amp_1, x_main_psd_1, 
        x_main_freq_2, x_main_amp_2, x_main_psd_2
    ) = calc_main_freq_and_amplitude(x)
    (
        y_main_freq_1, y_main_amp_1, y_main_psd_1, 
        y_main_freq_2, y_main_amp_2, y_main_psd_2
    ) = calc_main_freq_and_amplitude(y)
    (
        z_main_freq_1, z_main_amp_1, z_main_psd_1, 
        z_main_freq_2, z_main_amp_2, z_main_psd_2
    ) = calc_main_freq_and_amplitude(z)
    
    # a list of feature
    feature_list = [
        x_mean, 
        y_mean, 
        z_mean, 
        x_var, 
        y_var, 
        z_var, 
        x_std, 
        y_std, 
        z_std, 
        x_cv, 
        y_cv, 
        z_cv,
        x_skew, 
        y_skew, 
        z_skew, 
        x_kurtosis, 
        y_kurtosis, 
        z_kurtosis, 
        x_max, 
        y_max, 
        z_max, 
        x_min, 
        y_min, 
        z_min, 
        x_range, 
        y_range, 
        z_range,
        x_q25, 
        y_q25, 
        z_q25, 
        x_q50, 
        y_q50, 
        z_q50,
        x_q75, 
        y_q75, 
        z_q75, 
        mag_mean, 
        mag_var, 
        mag_skew, 
        mag_kurtosis, 
        mag_rms, 
        x_norm, 
        y_norm, 
        z_norm,
        x_ac, 
        y_ac, 
        z_ac,
        x_trend, 
        y_trend, 
        z_trend, 
        mag_trend,
        cov_xy, 
        cov_yz, 
        cov_zx, 
        corr_xy,
        corr_yz, 
        corr_zx,
        diff_xy_mean, 
        diff_yz_mean, 
        diff_zx_mean, 
        diff_xy_std, 
        diff_yz_std, 
        diff_zx_std, 
        x_st_mean, 
        y_st_mean, 
        z_st_mean, 
        x_st_var, 
        y_st_var, 
        z_st_var, 
        pitch_mean, 
        pitch_var, 
        roll_mean, 
        roll_var, 
        x_dy_mean, 
        y_dy_mean, 
        z_dy_mean, 
        x_dy_var, 
        y_dy_var, 
        z_dy_var, 
        x_dy_max, 
        y_dy_max, 
        z_dy_max, 
        x_PDBA_mean, 
        y_PDBA_mean, 
        z_PDBA_mean,
        ODBA_mean, 
        ODBA_var, 
        ODBA_max,
        VeDBA_mean, 
        VeDBA_var, 
        VeDBA_max,
        VeDBA_s_mean,
        x_ratio_VeDBA_to_PDBA_mean, 
        y_ratio_VeDBA_to_PDBA_mean,
        z_ratio_VeDBA_to_PDBA_mean,
        x_dcp_mean, 
        y_dcp_mean, 
        z_dcp_mean,
        x_dcp_std, 
        y_dcp_std, 
        z_dcp_std,
        x_main_freq_1, 
        x_main_amp_1, 
        x_main_psd_1, 
        x_main_freq_2, 
        x_main_amp_2, 
        x_main_psd_2,
        y_main_freq_1, 
        y_main_amp_1, 
        y_main_psd_1, 
        y_main_freq_2, 
        y_main_amp_2, 
        y_main_psd_2,
        z_main_freq_1, 
        z_main_amp_1, 
        z_main_psd_1, 
        z_main_freq_2, 
        z_main_amp_2, 
        z_main_psd_2,
    ]
    
    return feature_list


# def calc_auto_correlation(data_array, time_lag=1):
#     '''
#     https://momonoki2017.blogspot.com/2018/03/python7.html
#     '''
#     ac_array = np.empty(time_lag+1)
#     mean = np.mean(data_array)
#     ac_array[0] = np.sum((data_array - mean)**2) / np.sum((data_array - mean)**2) 
#     for i in np.arange(1, time_lag+1): 
#         ac_array[i] = np.sum((data_array[i:] - mean)*(data_array[:-i] - mean)) / np.sum((data_array - mean)**2)
#     return ac_array


def calc_auto_correlation_v2(data_array, time_lag=1):
    '''
    https://momonoki2017.blogspot.com/2018/03/python7.html
    '''
    ac_array = np.empty(time_lag+1)
    mean = np.mean(data_array)
    # print(mean)
    if np.sum((data_array - mean)**2) == 0:
        ac_array = np.ones(time_lag+1)
        print(f"Warning: cannot calculate auto correlation for this inputs, due to devision by zero")
        print(f"np.sum((data_array - mean)**2) = {np.sum((data_array - mean)**2)} -> Returned an array of 1.0")
    else:
        ac_array[0] = np.sum((data_array - mean)**2) / np.sum((data_array - mean)**2) 
        for i in np.arange(1, time_lag+1): 
            ac_array[i] = np.sum((data_array[i:] - mean)*(data_array[:-i] - mean)) / np.sum((data_array - mean)**2)
    
    # clip
    if ac_array.size > data_array.size:
        ac_array = ac_array[:data_array.size]
    
    return ac_array


def calc_trend(value):
    t = np.arange(0, value.shape[0], 1)
    model = LinearRegression().fit(t.reshape(-1, 1), value.reshape(-1, 1))
    trend = model.coef_[0][0]
    return trend


def calc_difference_of_continuous_points(signal):
    dcp = []
    for i in range(0, len(signal)-1):
        dcp.append(np.abs(signal[i+1]-signal[i]))
    dcp_mean = np.mean(dcp)
    dcp_std = np.std(dcp)
    return dcp, dcp_mean, dcp_std


def calc_main_freq_and_amplitude(signal):
    '''
    # https://momonoki2017.blogspot.com/2018/03/pythonfft-2.html
    # https://momonoki2017.blogspot.com/2018/03/pythonfft-1-fft.html
    '''
    SAMPLING_RATE = 25
    WINDOW_LENGTH = 2
    N = SAMPLING_RATE*WINDOW_LENGTH
    t = np.arange(0, N, 1)
    dt = 1 / SAMPLING_RATE
    fq = np.linspace(0, 1.0/dt, N) 
    
    # https://ryo-iijima.com/fftresult/
    F = np.fft.fft(signal) # FFT, Fast Fourier Transform
    F_abs = np.abs(F) # AS, Amplitude Spectrum
    F_abs_amp = F_abs / SAMPLING_RATE * 2 # AS divided by (sampling rate/2)
    F_abs_amp[0] = 0 
    # amplitude at frequency 0 should be 0
    #  (if there is direct current component, the value will be large) # F_abs_amp[0] / 2 
    ps = F_abs_amp*F_abs_amp # Power Spectrum
    psd = ps / SAMPLING_RATE
    # nyquist_frequency = int(SAMPLING_RATE/2)
    idx_1 = np.argsort(F_abs_amp[:np.int(N/2)])[::-1][0] # the index of the largest amplitude
    idx_2 = np.argsort(F_abs_amp[:np.int(N/2)])[::-1][1] # the index of the second largest amplitude 
    main_freq_1, main_freq_2 = fq[idx_1], fq[idx_2]
    main_amp_1, main_amp_2 = F_abs_amp[idx_1], F_abs_amp[idx_2]
    main_psd_1, main_psd_2 = psd[idx_1], psd[idx_2]
    return (main_freq_1, main_amp_1, main_psd_1, 
            main_freq_2, main_amp_2, main_psd_2)
    



