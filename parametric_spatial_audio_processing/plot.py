##############################
#
#   plot.py
#
#   Created by Andres Perez Lopez
#   25/01/2018
#
#   Utils for plotting waveforms and spectrograms
##############################
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


def plot_signal(signal,title='',time_axis='samples',y_scale='linear'):

    fig = plt.figure()
    fig.suptitle(title, fontsize=16, y=1)

    v_min = np.min(signal.data)
    v_max = np.max(signal.data)

    # signal_data_log = 20*np.log10(signal.data*signal.data)
    # signal_data_log = signal_data_log[signal_data_log!=-np.inf]
    # print(signal_data_log)
    # v_min_log = np.min(signal_data_log)
    # v_max_log = np.max(signal_data_log)

    # print('limits',v_min,v_max,v_min_log,v_max_log)

    channel_names = ['W','X','Y','Z']
    num_channels = signal.get_num_channels()
    for i in range(num_channels):
        ax = plt.subplot(num_channels, 1, i + 1)

        if y_scale == 'linear':
            ax.plot(signal.data[i, :])
            plt.ylim(v_min,v_max)
        elif y_scale == 'log':
            ax.semilogy(signal.data[i, :])
            # plt.ylim(1e-5,1e0)

        if i<len(channel_names):
            plt.ylabel(channel_names[i])

        if time_axis == 'samples':
            plt.xlabel('Time [samples]')
        elif time_axis == 'seconds':
            plt.xlabel('Time [seconds]')

            # signal_length_samples = float(np.shape(signal.data)[1])
            # signal_length_seconds = signal_length_samples/signal.sample_rate
            # interval_seconds = 0.5
            # interval_samples = interval_seconds * signal.sample_rate
            #
            # ax.set_xticks(np.arange(0,signal_length_samples,interval_samples))
            # ax.set_xticklabels(np.arange(0,signal_length_seconds,interval_seconds))

            signal_length = np.shape(signal.data)[1]
            scale = 1./signal.sample_rate
            ax.set_xticks(np.arange(0,signal_length,signal_length/10))
            ax.set_xticklabels(np.arange(0,signal_length*scale,signal_length/10*scale))

        plt.grid()

def plot_magnitude_spectrogram(stft,title=''):

    fig = plt.figure()
    fig.suptitle(title, fontsize=16, y=1)

    # Compare with values from W channel, which should be probably maximum
    # Also it won't be empty as it might be the other channels in the case of background/max diffuseness
    v_min = np.min(np.abs(stft.data[0]))
    if v_min == 0:
        v_min = 1e-20
    v_max = np.max(np.abs(stft.data[0]))

    ## Preprocess data to add a very small amount in order to avoid pure 0s
    mag_stft = stft.get_magnitude_stft()
    for ch in range(mag_stft.get_num_channels()):
        for t in range(mag_stft.get_num_time_bins()):
            for k in range(mag_stft.get_num_frequency_bins()):
                if mag_stft.data[ch,k,t] == 0:
                    mag_stft.data[ch, k, t] = v_min

    ## Adjust frequency ranges for pcolormesh
    freq_distance = stft.f[1] - stft.f[0]
    dd = freq_distance / 2
    f = np.arange(stft.f[0]-dd, stft.f[-1]+dd+1, freq_distance)

    num_channels = stft.get_num_channels()
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)
        plt.pcolormesh(mag_stft.t, f, mag_stft.data[i,:,:],norm=LogNorm(v_min,v_max))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()

def plot_phase_spectrogram(stft,abs=False,title=''):

    fig = plt.figure()
    fig.suptitle(title, fontsize=16, y=1)

    ## Adjust frequency ranges for pcolormesh
    freq_distance = stft.f[1] - stft.f[0]
    dd = freq_distance / 2
    f = np.arange(stft.f[0]-dd, stft.f[-1]+dd+1, freq_distance)

    num_channels = stft.get_num_channels()
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)
        if not abs:
            plt.pcolormesh(stft.t, f, np.angle(stft.data[i,:,:]))
        else:
            plt.pcolormesh(stft.t, f, np.abs(np.angle(stft.data[i, :, :])))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()



def plot_doa(arg,title=''):
    if arg.data.ndim == 2:
        return plot_doa_t(arg,title)
    elif arg.data.ndim == 3:
        return plot_doa_tf(arg,title)
    else:
        raise TypeError

def plot_doa_t(signal,title=''):
    '''
    Time signal [2,num_samples] with [azimuth,elevation]
    :param Signal:
    :return:
    '''
    fig = plt.figure()
    fig.suptitle(title, fontsize=16, y=1)

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

def plot_doa_tf(stft,title=''):
    '''
    Stft [2,num_f,num_t] with [azimuth,elevation]
    :param stft:
    :return:
    '''
    fig = plt.figure()
    fig.suptitle(title, fontsize=16, y=1)

    ## Adjust frequency ranges for pcolormesh
    freq_distance = stft.f[1] - stft.f[0]
    dd = freq_distance / 2
    f = np.arange(stft.f[0]-dd, stft.f[-1]+dd+1, freq_distance)

    # Azimuth
    plt.subplot("211")
    plt.pcolormesh(stft.t, f, stft.data[0, :, :], cmap='rainbow', vmin=-np.pi, vmax=np.pi)
    plt.title('Azimuth (rad)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()

    # Elevation
    plt.subplot("212")
    plt.pcolormesh(stft.t, f, stft.data[1, :, :], cmap='magma', vmin=-np.pi/2, vmax=np.pi/2)
    plt.title('Elevation (rad)')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()

def plot_diffuseness(arg,title=''):
    if arg.data.ndim == 2:
        return plot_diffuseness_t(arg,title)
    elif arg.data.ndim == 3:
        return plot_diffuseness_tf(arg,title)
    else:
        raise TypeError

def plot_diffuseness_t(signal,title=''):
    # Signal with 1 channel
    fig = plt.figure()
    fig.suptitle(title, fontsize=16, y=1)
    ax = plt.plot(signal.data[0,:])
    plt.xlabel('Time [samples]')

def plot_diffuseness_tf(stft,title=''):

    fig = plt.figure()
    fig.suptitle(title, fontsize=16, y=1)

    v_min = 0.
    v_max = 1.

    ## Adjust frequency ranges for pcolormesh
    freq_distance = stft.f[1] - stft.f[0]
    dd = freq_distance / 2
    f = np.arange(stft.f[0]-dd, stft.f[-1]+dd+1, freq_distance)

    num_channels = stft.get_num_channels()
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)
        plt.pcolormesh(stft.t, f, stft.data[i,:,:], cmap='plasma_r', vmin=v_min, vmax=v_max)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()

def plot_directivity(arg,title=''):

    if arg.data.ndim == 2:
        return plot_directivity_t(arg,title)
    elif arg.data.ndim == 3:
        return plot_directivity_tf(arg,title)
    else:
        raise TypeError

def plot_directivity_t(signal,title=''):
    # Signal with 1 channel
    fig = plt.figure()
    fig.suptitle(title, fontsize=16, y=1)
    ax = plt.plot(signal.data[0,:])
    plt.xlabel('Time [samples]')

def plot_directivity_tf(stft,title=''):
    # Stft with 1 channel
    fig = plt.figure()
    fig.suptitle(title, fontsize=16, y=1)

    v_min = 0.
    v_max = 1.

    ## Adjust frequency ranges for pcolormesh
    freq_distance = stft.f[1] - stft.f[0]
    dd = freq_distance / 2
    f = np.arange(stft.f[0]-dd, stft.f[-1]+dd+1, freq_distance)

    num_channels = stft.get_num_channels()
    for i in range(num_channels):
        plt.subplot(num_channels, 1, i + 1)
        plt.pcolormesh(stft.t, f, stft.data[i, :, :], cmap='plasma', vmin=v_min, vmax=v_max)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar()


def plot_mask(arg,title=''):
    if arg.data.ndim == 2:
        return plot_mask_t(arg,title)
    elif arg.data.ndim == 3:
        return plot_mask_tf(arg,title)
    else:
        raise TypeError

def plot_mask_t(signal):
    raise NotImplementedError

def plot_mask_tf(stft,title=''):
    # Stft with 1 channel
    fig = plt.figure()
    if title:
        fig.suptitle(title, fontsize=16, y=1)

    v_min = 0.
    v_max = 1.

    ## Adjust frequency ranges for pcolormesh
    freq_distance = stft.f[1] - stft.f[0]
    dd = freq_distance / 2
    f = np.arange(stft.f[0]-dd, stft.f[-1]+dd+1, freq_distance)

    plt.pcolormesh(stft.t, f, stft.data[0], cmap='binary', vmin=v_min, vmax=v_max)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()


def plot_doa_2d_histogram(doa,jnd=1,title='',groundtruth=None, tol=None):
    # TODO: 1d version

    fig = plt.figure()
    ax = plt.gca()
    if title:
        fig.suptitle(title, fontsize=16, y=1)

    resolution_azi = 360/jnd
    resolution_ele = 180/jnd

    # First, flatten the data
    flatten_azi = doa.data[0,:,:].flatten()
    flatten_ele = doa.data[1, :, :].flatten()

    # Call the 2dhist method, but with the `nan`s removed
    cmap = plt.cm.get_cmap("plasma")
    # cmap.set_bad("blue")
    h = plt.hist2d(flatten_azi[~np.isnan(flatten_azi)],
                   flatten_ele[~np.isnan(flatten_ele)],
                   bins=[resolution_azi, resolution_ele],
                   range=[[-np.pi,np.pi],[-np.pi/2,np.pi/2]],
                   norm=matplotlib.colors.LogNorm(),
                   cmap=cmap)



    # plot grid
    # TODO
    # v_ticks = np.arange(-np.pi, np.pi, 5*2*np.pi/360)
    # h_ticks = np.arange(-np.pi/2, np.pi*2, 5 * 2 * np.pi / 360)
    # ax.set_xticks(v_ticks)
    # ax.set_yticks(h_ticks)
    plt.grid()
    plt.colorbar()



    # plot groundtruth
    if groundtruth:
        # TODO: CHECK THAT IS A LIST WITH [azi, ele]
        azi = groundtruth[0]
        ele = groundtruth[1]
        plt.scatter(azi, ele, s=30, marker='o')

        if tol:
            # add lines
            x = [azi-tol/2,azi+tol/2,azi+tol/2,azi-tol/2,azi-tol/2]
            y = [ele-tol/2,ele-tol/2,ele+tol/2,ele+tol/2,ele-tol/2]
            # x = [0, 1, 1, 0, 0]
            # y = [0, 0, 1, 1, 0]
            line = Line2D(x, y)
            ax.add_line(line)

    return h[0]

