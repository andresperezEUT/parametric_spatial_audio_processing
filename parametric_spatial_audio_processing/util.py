##############################
#
#   util.py
#
#   Created by Andres Perez Lopez
#   25/01/2018
#
#   Some utility methods
##############################

import numpy as np
import parametric_spatial_audio_processing.core
import copy

# BFormat Channel order dictionaries
# TODO: extend to higher orders

acn2fuma = {0: 0, 1: 3, 2: 1, 3: 2}
fuma2acn = {0: 0, 1: 2, 2: 3, 3: 1}

def _validate_bformat_array(array):
    # ndarray, 4 channel, shape [channels,samples]
    # assert type(array) is np.ndarray
    assert np.ndim(array) == 2
    assert np.shape(array)[0] == 4

def convert_bformat_acn_2_fuma(acn_array):
    '''
    Convert a 4-channel BFormat audio stream into FuMa channel ordering
    :param acn_array: np.ndarray [channel,samples]
    :return: a new ndarray
    '''
    _validate_bformat_array(acn_array)

    fuma_array = np.zeros(np.shape(acn_array))

    num_channels = np.shape(acn_array)[0]
    for channel in range(num_channels):
        fuma_channel = acn2fuma[channel]
        fuma_array[channel, :] = acn_array[fuma_channel, :]

    return fuma_array


def convert_bformat_fuma_2_acn(fuma_array):
    '''
    Convert a 4-channel BFormat audio stream into ACN channel ordering
    :param fuma_array: np.ndarray [channel,samples]
    :return: a new ndarray
    '''
    _validate_bformat_array(fuma_array)

    acn_array = np.zeros(np.shape(fuma_array))

    num_channels = np.shape(fuma_array)[0]
    for channel in range(num_channels):
        acn_channel = fuma2acn[channel]
        acn_array[channel, :] = fuma_array[acn_channel, :]

    return acn_array

def convert_bformat_sn3d_2_fuma(sn3d_array):
    '''
    Convert a 4-channel BFormat audio stream into FuMa normalization
    :param sn3d_array: np.ndarray [channel,samples]
    :return: a new ndarray
    '''
    _validate_bformat_array(sn3d_array)

    fuma_array = sn3d_array
    # Channel 0 reduced by sqrt(2)
    fuma_array[0,:] = sn3d_array[0,:] / np.sqrt(2)

    return fuma_array


def convert_bformat_fuma_2_sn3d(fuma_array):
    '''
    Convert a 4-channel BFormat audio stream into FuMa normalization
    :param fuma_array: np.ndarray [channel,samples]
    :return: a new ndarray
    '''
    _validate_bformat_array(fuma_array)

    sn3d_array = fuma_array
    # Channel 0 reduced by sqrt(2)
    sn3d_array[0, :] = fuma_array[0, :] * np.sqrt(2)

    return sn3d_array

def convert_bformat_n3d_2_sn3d(n3d_array):
    '''
    Convert a 4-channel BFormat audio stream into SN3D normalization
    :param n3d_array: np.ndarray [channel,samples]
    :return: a new ndarray
    '''
    _validate_bformat_array(n3d_array)

    sn3d_array = n3d_array
    # Channel 0 remains same
    # Channels 1:3 -> n3d = sqrt(3)*sn3d
    sn3d_array[1:] = n3d_array[1:] / np.sqrt(3)

    return sn3d_array


def cartesian_2_spherical(arg):
    '''
    Convert a cartesian time series array
    into spherical coordinates [azimuth,elevation,radius]
    From Signal or Stft types
    :param ndarray_cartesian: cartesian ndarray
    :return: Spherical coordinates ndarray
    '''

    # Input must be ndarray with:
    #   2D with [channels,samples]
    #   3D with [channels,t,f]
    if type(arg) is not np.ndarray:
        raise TypeError
    if np.ndim(arg) == 2:
        if np.shape(arg)[0] is not 3:
            raise TypeError
        return cartesian_2_spherical_t(arg)
    elif np.ndim(arg) == 3:
        if np.shape(arg)[0] is not 3:
            raise TypeError
        return cartesian_2_spherical_tf(arg)
    else:
        raise TypeError

def cartesian_2_spherical_t(ndarray_cartesian):
    '''
    2D array
    '''
    x = ndarray_cartesian[0,:]
    y = ndarray_cartesian[1,:]
    z = ndarray_cartesian[2,:]

    r = np.sqrt(np.power(x, 2)
                + np.power(y, 2)
                + np.power(z, 2))

    # Substitute all zeros by nans, just to avoid runtimeWarning
    # TODO: probably there's a much more pythonic way to write that...
    for i in range(len(r)):
        if r[i]==0:
            r[i] = np.nan

    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / r)

    # If there's a nan in elevation (because of r), then override it also in azimuth...
    # TODO: probably there's a much more pythonic way to write that...
    for i in range(len(azimuth)):
        if np.isnan(elevation[i]):
            azimuth[i] = np.nan

    return np.asarray([azimuth, elevation, r])

def cartesian_2_spherical_tf(ndarray_cartesian):
    '''
    3D array
    '''
    x = ndarray_cartesian[0,:,:]
    y = ndarray_cartesian[1,:,:]
    z = ndarray_cartesian[2,:,:]

    r = np.sqrt(np.power(x, 2)
                + np.power(y, 2)
                + np.power(z, 2))

    # Substitute all zeros by nans, just to avoid runtimeWarning
    # TODO: probably there's a much more pythonic way to write that...
    for i in range(np.shape(r)[0]):
        for j in range(np.shape(r)[1]):
            if r[i][j]==0:
                r[i][j] = np.nan

    azimuth = np.arctan2(y, x)
    elevation = np.arcsin(z / r)

    # If there's a nan in elevation (because of r), then override it also in azimuth...
    # TODO: probably there's a much more pythonic way to write that...
    for i in range(np.shape(azimuth)[0]):
        for j in range(np.shape(azimuth)[1]):
            if np.isnan(elevation[i][j]):
                azimuth[i][j] = np.nan

    return np.asarray([azimuth, elevation, r])

def herm(ndarray):
    '''
    TODO
    Hermitian Transpose:
    matrix transposition + complex conjugate
    :param ndarray:
    :return:
    '''
    return np.conj(np.transpose(ndarray))

def spatial_covariance_matrix(stft,ch):

    k = stft.get_num_frequency_bins()
    n = stft.get_num_time_bins()

    a = stft.data[ch,:,:]

    c = np.cov(np.dot(a,herm(a)))
    return c


def compute_signal_envelope(signal, windowsize=1024):
    '''
    Computing from W channel!!

    :param signal:
    :param windowsize:
    :return:
    '''

    window = np.zeros(windowsize)
    envelope = np.zeros((1,signal.get_num_frames()))

    # zeropad with windowsize-1 zeros to the left
    zp_array = np.pad(signal.data[0],(windowsize-1,0),'constant')
    for n in range(windowsize,len(zp_array)+1):
        window = zp_array[n-windowsize:n]

        rms = np.sum(np.power(window,2))
        rms = np.sqrt(rms/windowsize)

        envelope[0,n-windowsize] = rms

    envelope_signal = parametric_spatial_audio_processing.Signal(envelope,signal.sample_rate)

    return envelope_signal

def find_contiguous_region(array, windowsize=1024, th=0.01):

    min_consecutive = windowsize
    current_count = 0
    current = False
    last_current = False

    starts = []
    ends = []

    for n in range(len(array)):
        v = array[n]
        if (v>th):
            current_count += 1
            if current_count == min_consecutive:
                current = True
                starts.append(n-min_consecutive+1)

        else:
            if (current == True):
                current = False
                ends.append(n)
            current_count = 0

    assert len(starts) == len(ends)
    num_regions = len(starts)

    return (num_regions, starts, ends)

def segmentate_audio(signal, windowsize=1024, th=0.01):

    # TODO
    '''

    :param signal:
    :param windowsize:
    :param th:
    :return:
    '''
    env = compute_signal_envelope(signal.data[0],windowsize)
    return find_contiguous_region(env.data[0],windowsize,th)


def moving_average(a, n=3) :
    """
    Compute moving average of the signal
    :param a:
    :param n:
    :return:
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n