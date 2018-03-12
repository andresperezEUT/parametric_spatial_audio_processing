##############################
#
#   plot.py
#
#   Created by Andres Perez Lopez
#   25/01/2018
#
#   Utils for plotting waveforms and spectrograms
##############################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_audio_waveform(signal,title=''):

    fig = plt.figure()
    fig.suptitle(title, fontsize=16)

    v_min = np.min(signal.data)
    v_max = np.max(signal.data)

    num_channels = signal.get_num_channels()
    for i in range(num_channels):
        ax = plt.subplot(num_channels, 1, i + 1)
        ax.plot(signal.data[i, :])
        plt.ylim(v_min,v_max)
        plt.xlabel('Time [samples]')
        plt.grid()

def plot_magnitude_spectrogram(stft,title=''):

    fig = plt.figure()
    fig.suptitle(title, fontsize=16)

    v_min = np.min(np.abs(stft.data))
    v_max = np.max(np.abs(stft.data))

    num_channels = stft.get_num_channels()
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)
        plt.pcolormesh(stft.t, stft.f, np.abs(stft.data[i,:,:]),norm=LogNorm(vmin=1e-10, vmax=v_max))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()

def plot_phase_spectrogram(stft,abs=False,title=''):

    fig = plt.figure()
    fig.suptitle(title, fontsize=16)

    num_channels = stft.get_num_channels()
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)
        if not abs:
            plt.pcolormesh(stft.t, stft.f, np.angle(stft.data[i,:,:]))
        else:
            plt.pcolormesh(stft.t, stft.f, np.abs(np.angle(stft.data[i, :, :])))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()



def plot_doa(arg):
    if arg.data.ndim == 2:
        return plot_doa_t(arg)
    elif arg.data.ndim == 3:
        return plot_doa_tf(arg)
    else:
        raise TypeError

def plot_doa_t(signal):
    '''
    Time signal [2,num_samples] with [azimuth,elevation]
    :param Signal:
    :return:
    '''
    fig = plt.figure()
    fig.suptitle("DoA[n]", fontsize=16)

    # Azimuth
    ax = plt.subplot("211")
    ax.plot(signal.data[0, :])
    ax.set_ylim([-np.pi, np.pi])
    plt.ylabel('Azimuth (rad)')
    plt.xlabel('Time [samples]')

    # Elevation
    ax = plt.subplot("212")
    ax.set_title("Elevation")
    ax.plot(signal.data[1, :])
    ax.set_ylim([-np.pi/2, np.pi/2])
    plt.ylabel('Elevation (rad)')
    plt.xlabel('Time [samples]')

def plot_doa_tf(stft):
    '''
    Stft [2,num_f,num_t] with [azimuth,elevation]
    :param stft:
    :return:
    '''
    fig = plt.figure()
    fig.suptitle("DoA[kn]", fontsize=16)

    # Azimuth
    plt.subplot("211")
    plt.pcolormesh(stft.t, stft.f, stft.data[0, :, :], cmap='rainbow', vmin=-np.pi, vmax=np.pi)
    plt.title('Azimuth (rad)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()

    # Elevation
    plt.subplot("212")
    plt.pcolormesh(stft.t, stft.f, stft.data[1, :, :], cmap='magma', vmin=-np.pi/2, vmax=np.pi/2)
    plt.title('Elevation (rad)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()

def plot_diffuseness(arg):
    if arg.data.ndim == 2:
        return plot_diffuseness_t(arg)
    elif arg.data.ndim == 3:
        return plot_diffuseness_tf(arg)
    else:
        raise TypeError

def plot_diffuseness_t(signal):
    # Signal with 1 channel
    fig = plt.figure()
    fig.suptitle("phi[n]", fontsize=16)
    ax = plt.plot(signal.data[0,:])
    plt.xlabel('Time [samples]')

def plot_diffuseness_tf(stft):
    # Stft with 1 channel
    fig = plt.figure()
    fig.suptitle("phi[kn]", fontsize=16)

    v_min = np.min(np.abs(stft.data))
    v_max = np.max(np.abs(stft.data))

    plt.pcolormesh(stft.t, stft.f, 1-stft.data[0], cmap='hot', vmin=v_min, vmax=v_max)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()

def plot_directivity(arg):
    if arg.data.ndim == 2:
        return plot_directivity_t(arg)
    elif arg.data.ndim == 3:
        return plot_directivity_tf(arg)
    else:
        raise TypeError

def plot_directivity_t(signal):
    # Signal with 1 channel
    fig = plt.figure()
    fig.suptitle("delta[n]", fontsize=16)
    ax = plt.plot(signal.data[0,:])
    plt.xlabel('Time [samples]')

def plot_directivity_tf(stft):
    # Stft with 1 channel
    fig = plt.figure()
    fig.suptitle("delta[kn]", fontsize=16)

    v_min = np.min(np.abs(stft.data))
    v_max = np.max(np.abs(stft.data))

    plt.pcolormesh(stft.t, stft.f, stft.data[0], cmap='hot', vmin=v_min, vmax=v_max)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()