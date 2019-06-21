from pysofaconventions import *
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

pp = "/Users/andres.perez/source/parametric_spatial_audio_processing"
sys.path.append(pp)
# pp = "/Users/andres.perez/source/parametric_spatial_audio_processing/research_stuff/utils.py"
# sys.path.append(pp)
import parametric_spatial_audio_processing as psa
import copy
import acoustics
import scipy.stats
import scipy.signal
from matplotlib.patches import Ellipse
from scipy import ndimage as ndi
from matplotlib.colors import LogNorm

from utils import *


################################################################################################
################################################################################################


compute_room_model = True
compute_recorded = True
compute_simulated = True

### PARAMETERS

sofa_folder = '/Volumes/Dinge/SOFA/Fundacio_Miro'
# sofa_folder = '/Volumes/Dinge/SOFA/BINCI/Alte_Pinakotheke'

# sofa_files = glob.glob(sofa_folder+'/*.sofa')
sofa_files = [sofa_folder+'/Sala1.sofa']
# sofa_files = [sofa_folder+'/foyer_ambeo.sofa']

## Stft
processing_window_ms = 50
analysis_window_size = 32
window_overlap = analysis_window_size * 3 / 4
fmin = 125
fmax = 8000
fft_factor = 4
fft_size = analysis_window_size * fft_factor

# Energy density smoothing
gaussian_window_length = 21
gaussian_window_shape = 0.5
gaussian_window_std = 2

# Peak peaking
# cwt_widths = np.arange(gaussian_window_length-10, gaussian_window_length+10)

# Threshold
diffuseness_th = 0.75

## Simulation parameters

matlab_root = "/Applications/MATLAB_R2018b.app" # Matlab path
rir_root = '/Users/andres.perez/source/MATLAB/SMIR-Generator-master' # Script path

n_harm = 36
oversampling = 1
high_pass = 1
source_directivity = 'o'  # omni
reflection_order = 2
refl_coeff_ang_dep = 0  # not really sure about this parameter
sphRadius = 0.049  # Ambeo
sphType = 'open'  # Ambeo

## [FLU, FRD, BLD, BRU]
## ACTHUNG!! the second parameter is actually `inclination`, not `elevation`
capsule_positions = [[np.pi / 4, np.pi / 4],
                     [7 * np.pi / 4, 3 * np.pi / 4],
                     [3 * np.pi / 4, 3 * np.pi / 4],
                     [5 * np.pi / 4, np.pi / 4]]


################################################################################################
################################################################################################



def compute_peak_statistics(ir,
                            sample_rate,
                            ambisonics_ordering,
                            ambisonics_normalization,
                            plot=False,
                            plot_title = ''):


    ## Signal
    signal = psa.Signal(ir, int(sample_rate), ambisonics_ordering, ambisonics_normalization)
    if plot:
        psa.plot_signal(signal,title=plot_title+'IR')

    stft = psa.Stft.fromSignal(signal,
                               window_size=analysis_window_size,
                               window_overlap=window_overlap,
                               nfft=fft_size,
                               )
    stft = stft.limit_bands(fmin=fmin, fmax=fmax)

    if plot:
        psa.plot_magnitude_spectrogram(stft,title=plot_title+'IR Magnitude Spectrogram, w='+str(analysis_window_size))

    ### Energy Density
    energy_density_t = psa.compute_energy_density(signal)
    if plot:
        psa.plot_signal(energy_density_t,'Energy Density', y_scale='log')

    # # Smoothed signal
    # L = gaussian_window_length
    # smooth_window = scipy.signal.general_gaussian(L, p=gaussian_window_shape, sig=gaussian_window_std)
    # smoothed_energy_density_t = scipy.signal.fftconvolve(smooth_window, energy_density_t.data[0, :])
    # smoothed_energy_density_t = (np.average(energy_density_t.data[0, :]) / np.average(smoothed_energy_density_t)) * smoothed_energy_density_t
    # smoothed_energy_density_t = np.roll(smoothed_energy_density_t, -((L - 1) / 2))
    # smoothed_energy_density_t = smoothed_energy_density_t[:-(L - 1)]  # same length
    #
    # ### Peak peaking
    #
    # # WAVELET
    # cwt_widths = np.arange(0.5*L,1.5*L) # Find peaks of shape among 2 gaussian window lengths
    # smoothed_peaks = scipy.signal.find_peaks_cwt(smoothed_energy_density_t, widths=cwt_widths)

    # # Fine sample correction of peaks: find local maxima over a gaussian window length
    # corrected_peaks = copy.deepcopy(smoothed_peaks)
    # for peak_idx,peak in enumerate(smoothed_peaks):
    #     local_energy = smoothed_energy_density_t[peak - (L / 2):peak + (L / 2)]
    #     corrected_peaks[peak_idx] = np.argmax(local_energy) + peak - (L / 2)
    #
    # if plot:
    #     plt.figure()
    #     plt.suptitle('Smoothed Energy Density & peaks')
    #     ax = plt.subplot(111)
    #     ax.semilogy(energy_density_t.data[0,:])
    #     ax.semilogy(smoothed_energy_density_t)
    #
    #     # plot peak estimates
    #     for peak in corrected_peaks:
    #         plt.axvline(x=peak, color='g')
    #
    #     # plot time frames
    #     for x in np.arange(0,processing_window_samples,analysis_window_size):
    #         plt.axvline(x=x, color='r', alpha=0.3)
    #
    #     plt.grid()
    #
    # peak_time_bins = []
    # for peak in corrected_peaks:
    #     peak_time_bins.append(find_maximal_time_bin(peak, stft, overlap_factor))


    ## Raw Estimates
    doa = psa.compute_DOA(stft)
    if plot:
        psa.plot_doa(doa,title=plot_title+'DoA estimates, w='+str(analysis_window_size))

    # diffuseness = psa.compute_diffuseness(stft)
    # if plot:
    #     psa.plot_diffuseness(diffuseness,title=plot_title+'Diffuseness, w='+str(analysis_window_size))


    neighborhood_size = 3
    ## DOA variance
    doa_var = copy.deepcopy(doa)
    for n in range(doa.get_num_time_bins()):
        for k in range(doa.get_num_frequency_bins()):
            local_var_azi = 0
            local_var_ele = 0
            local_azi = []
            local_ele = []
            r = int(np.floor(neighborhood_size/2)) # neighborhood radius
            for x in np.arange(n - r, n + r + 1):
                for y in np.arange(k - r, k + r + 1):
                    if x < 0:
                        continue
                    elif x >= doa.get_num_time_bins():
                        continue
                    if y < 0:
                        continue
                    elif y >= doa.get_num_frequency_bins():
                        continue
                    local_azi.append(doa.data[0,y,x])
                    local_ele.append(doa.data[1,y,x])
                    # local_var_azi += np.std(doa.data[0,y,x])
                    # local_var_ele += np.std(doa.data[1,y,x])
            local_var_azi = scipy.stats.circvar(np.array(local_azi))
            local_var_ele = np.var(np.array(local_ele))
            doa_var.data[0,k,n] = local_var_azi
            doa_var.data[1,k,n] = local_var_ele

    ## DOA VAR salience

    neighborhood_size = round_up_to_odd(doa_var.get_num_frequency_bins())
    doa_var_salience = threshold_local(doa_var.data[0,:],block_size=neighborhood_size)
    doa_var_max_salience_mask = copy.deepcopy(doa_var)
    doa_var_min_salience_mask = copy.deepcopy(doa_var)
    for k in range(doa_var.get_num_frequency_bins()):
        for n in range(doa_var.get_num_time_bins()):

            if doa_var.data[0, k, n] > doa_var_salience[k, n]:
                doa_var_max_salience_mask.data[:, k, n] = 1.
            else:
                doa_var_max_salience_mask.data[:, k, n] = np.nan

            if doa_var.data[0, k, n] < doa_var_salience[k, n]:
                doa_var_min_salience_mask.data[:, k, n] = 1.
            else:
                doa_var_min_salience_mask.data[:, k, n] = np.nan


    # MINIMUM VARIANCE DOA
    masked_doa = doa.apply_mask(doa_var_min_salience_mask)
    if plot:
        psa.plot_doa(masked_doa,
                     title=plot_title + 'DOA - Minimum variance Salience Masked, w=' + str(analysis_window_size) + ' N: ' + str(
                         neighborhood_size))

    masked_doa = doa.apply_mask(doa_var_max_salience_mask)
    # if plot:
    #     psa.plot_doa(masked_doa,
    #                  title=plot_title + 'DOA - Maximim variance Salience Masked, w=' + str(analysis_window_size) + ' N: ' + str(
    #                      neighborhood_size))


    # if plot:
        # plt.figure()
        # plt.suptitle('DOA VAR')
        # plt.subplot(211)
        # plt.pcolormesh(doa_var.data[0,:,:])
        # plt.subplot(212)
        # plt.pcolormesh(doa_var.data[1,:,:])
        #
        # psa.plot_mask(doa_var_max_salience_mask,title='MAX SALIENCE')
        # psa.plot_mask(doa_var_min_salience_mask,title='MIN SALIENCE')







    ## Energy density
    energy_density_tf = psa.compute_energy_density(stft)
    # if plot:
    #     psa.plot_magnitude_spectrogram(energy_density_tf,title='Energy Density Spectrogram, w='+str(analysis_window_size))


    # Energy density salience

    neighborhood_size = round_up_to_odd(energy_density_tf.get_num_frequency_bins())
    energy_density_salience = threshold_local(energy_density_tf.data[0,:],block_size=neighborhood_size)
    energy_density_salience_mask = copy.deepcopy(energy_density_tf)
    for k in range(energy_density_tf.get_num_frequency_bins()):
        for n in range(energy_density_tf.get_num_time_bins()):
            if energy_density_tf.data[0, k, n] > energy_density_salience[k, n]:
                energy_density_salience_mask.data[:, k, n] = 1.
            else:
                energy_density_salience_mask.data[:, k, n] = np.nan

    # if plot:
    #     fig = plt.figure()
    #     fig.suptitle('energy salience, w=' + str(analysis_window_size))
    #
    #     x = np.arange(np.shape(energy_density_salience)[0])
    #     y = np.arange(np.shape(energy_density_salience)[1])
    #     plt.pcolormesh(y, x, energy_density_salience, norm=LogNorm())
    #     plt.ylabel('Frequency [Hz]')
    #     plt.xlabel('Time [sec]')
    #     plt.colorbar()

    # if plot:
    #     psa.plot_mask(energy_density_salience_mask, title='Energy Salience Mask'+str(analysis_window_size))

    masked_energy = energy_density_tf.apply_mask(energy_density_salience_mask)
    # if plot:
    #     psa.plot_magnitude_spectrogram(masked_energy, title=plot_title+'Energy - Energy Salience Masked, w='+str(analysis_window_size)+' N: '+str(neighborhood_size))

    masked_doa = doa.apply_mask(energy_density_salience_mask)
    if plot:
        psa.plot_doa(masked_doa, title=plot_title+'DOA - Energy Salience Masked, w='+str(analysis_window_size)+' N: '+str(neighborhood_size))

    masked_doa = doa.apply_mask(energy_density_salience_mask).apply_mask(doa_var_min_salience_mask)
    if plot:
        psa.plot_doa(masked_doa, title=plot_title+'DOA - VAR MIN,  Energy Salience Masked, w='+str(analysis_window_size)+' N: '+str(neighborhood_size))


    # masked_diffuseness = diffuseness.apply_mask(energy_density_salience_mask)
    # if plot:
    #     psa.plot_diffuseness(masked_diffuseness, title=plot_title+'Diffuseness - Energy Salience Masked, w='+str(analysis_window_size)+' N: '+str(neighborhood_size))


    # # Diffuseness density salience
    #
    # neighborhood_size = round_up_to_odd(diffuseness.get_num_frequency_bins())
    # diffuseness_salience = threshold_local(diffuseness.data[0,:],block_size=neighborhood_size)
    # diffuseness_salience_mask = copy.deepcopy(diffuseness)
    # for k in range(diffuseness.get_num_frequency_bins()):
    #     for n in range(diffuseness.get_num_time_bins()):
    #         if diffuseness.data[0, k, n] < diffuseness_salience[k, n]:
    #             diffuseness_salience_mask.data[:, k, n] = 1.
    #         else:
    #             diffuseness_salience_mask.data[:, k, n] = np.nan
    #
    # masked_energy = energy_density_tf.apply_mask(diffuseness_salience_mask)
    # if plot:
    #     psa.plot_magnitude_spectrogram(masked_energy, title=plot_title + 'Energy - Diffuseness Salience Masked, w=' + str(
    #         analysis_window_size) + ' N: ' + str(neighborhood_size))
    #
    # masked_doa = doa.apply_mask(diffuseness_salience_mask)
    # if plot:
    #     psa.plot_doa(masked_doa, title=plot_title + 'DOA - Diffuseness Salience Masked, w=' + str(
    #         analysis_window_size) + ' N: ' + str(neighborhood_size))
    #
    # masked_diffuseness = diffuseness.apply_mask(diffuseness_salience_mask)
    # if plot:
    #     psa.plot_diffuseness(masked_diffuseness, title=plot_title + 'Diffuseness - Diffuseness Salience Masked, w=' + str(
    #         analysis_window_size) + ' N: ' + str(neighborhood_size))

    # #
    # if plot:
    #     psa.plot_mask(diffuseness_salience_mask, title='Diffuseness Salience Mask'+str(neighborhood_size))
    # #
    # masked_dif = diffuseness.apply_mask(diffuseness_salience_mask)
    # if plot:
    #     psa.plot_diffuseness(masked_dif, title='Diffuseness - Salience Masked'+str(neighborhood_size))
    #
    # energy_diffuseness_mask = energy_density_salience_mask.apply_mask(diffuseness_salience_mask)
    #
    # masked_doa = masked_doa.apply_mask(diffuseness_salience_mask)
    # if plot:
    #     psa.plot_doa(masked_doa, title='DOA - Salience Masked - Diffuseness Masked, w='+str(analysis_window_size))
    #
    # fig = plt.figure()
    # fig.suptitle("diffuseness salience, block:"+str(neighborhood_size), fontsize=16)
    # x = np.arange(np.shape(diffuseness_salience)[0])
    # y = np.arange(np.shape(diffuseness_salience)[1])
    # plt.pcolormesh(y,x, diffuseness_salience, cmap='plasma_r',norm=LogNorm())
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.colorbar()



    # diffuseness_energy_mask = diffuseness_mask.apply_mask(energy_density_mask)
    # if plot:
    #     psa.plot_mask(diffuseness_energy_mask, title='Diffuseness + Energy Density Mask')
    #
    # masked_diffuseness = diffuseness.apply_mask(diffuseness_mask)
    # if plot:
    #     psa.plot_diffuseness(masked_diffuseness, title='Diffuseness, diffuseness mask')
    #
    # masked_diffuseness = masked_diffuseness.apply_mask(energy_density_mask)
    # if plot:
    #     psa.plot_diffuseness(masked_diffuseness, title='Diffuseness, energy density maskm diffuseness mask')
    #
    # masked_doa = masked_doa.apply_mask(diffuseness_mask)
    # if plot:
    #     psa.plot_doa(masked_doa,title='DoA estimates, energy density mask, diffuseness mask')


    ### Find horizontal-contiguous bins on doa estimates
    time_bins_with_energy = []
    for n in range(energy_density_salience_mask.get_num_time_bins()):
        if not np.all(np.isnan(energy_density_salience_mask.data[0,:,n])):
            time_bins_with_energy.append(n)

    time_region_starts = []
    time_region_ends = []
    for idx, b in enumerate(time_bins_with_energy):
        if time_bins_with_energy[idx] - time_bins_with_energy[idx - 1] != 1:
            time_region_starts.append(time_bins_with_energy[idx])
            time_region_ends.append(time_bins_with_energy[idx - 1]+1)

    time_region_starts.sort()
    time_region_ends.sort()
    assert len(time_region_starts) == len(time_region_ends)


    # Compute local doa estimates on contiguous time regions
    peak_stats = []
    for idx in range(len(time_region_starts)):
        n_range = range(time_region_starts[idx],time_region_ends[idx])

        local_azi = []
        local_ele = []

        index_of_bins_estimated = []
        for n in n_range:
            # Filter nans
            for k in np.arange(energy_density_salience_mask.get_num_frequency_bins()):

                if not np.isnan(energy_density_salience_mask.data[0, k, n]):
                    local_azi.append(masked_doa.data[0, k, n])
                    local_ele.append(masked_doa.data[1, k, n])
                    index_of_bins_estimated.append(n)

        local_azi = np.asarray(local_azi)
        local_ele = np.asarray(local_ele)
        # local_dif = np.asarray(local_dif)

        local_azi_mean = scipy.stats.circmean(local_azi, high=np.pi, low=-np.pi)
        local_azi_std = scipy.stats.circstd(local_azi, high=np.pi, low=-np.pi)
        local_ele_mean = np.mean(local_ele)
        local_ele_std = np.std(local_ele)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.suptitle(plot_title+'FREQ BIN '+str(idx)+' - bins '+str(n_range))
            # cmap = plt.cm.get_cmap("copper")
            plt.grid()
            plt.xlim(-np.pi,np.pi)
            plt.ylim(-np.pi/2,np.pi/2)

            plt.scatter(local_azi, local_ele, marker='o',)
            plt.scatter(local_azi_mean, local_ele_mean, c='red', s=20, marker='+')
            ax.add_patch(Ellipse(xy=(local_azi_mean,local_ele_mean), width=local_azi_std,height=local_ele_std, alpha=0.5))
            ax.add_patch(Ellipse(xy=(local_azi_mean,local_ele_mean), width=3*local_azi_std,height=3*local_ele_std, alpha=0.1))


        mean_location = np.mean(index_of_bins_estimated)
        time_resolution_ms = float(processing_window_ms) / masked_doa.get_num_time_bins()
        estimated_location_ms = mean_location * time_resolution_ms

        peak_stats.append(
            [estimated_location_ms, [local_azi_mean, local_azi_std], [local_ele_mean, local_ele_std]])

    ### Return peak stats
    return peak_stats




            ### Not histograms but just azi-ele position
    # peak_stats = []

    # time_resolution_ms = float(processing_window_ms)/masked_doa.get_num_time_bins()

    ### Iterate over all time frames
    # for n in np.arange(masked_doa.get_num_time_bins()):
    #     ### must take care of nans
    #     local_azi = []
    #     local_ele = []
    #     # local_dif = []
    #     for k in np.arange(masked_doa.get_num_frequency_bins()):
    #         if not np.isnan(masked_doa.data[0, k, n]):
    #             local_azi.append(masked_doa.data[0, k, n])
    #         if not np.isnan(masked_doa.data[1, k, n]):
    #             local_ele.append(masked_doa.data[1, k, n])
    #         # if not np.isnan(diffuseness_Menergy.data[0, k, n]):
    #         #     local_dif.append(diffuseness_Menergy.data[0, k, n])
    #
    #     ### If we don't have values, just skip this window
    #     # if len(local_azi) == 0 or len(local_ele) == 0 or len(local_dif) == 0:
    #     if len(local_azi) == 0 or len(local_ele) == 0:
    #         continue
    #
    #     local_azi = np.asarray(local_azi)
    #     local_ele = np.asarray(local_ele)
    #     # local_dif = np.asarray(local_dif)
    #
    #     local_azi_mean = scipy.stats.circmean(local_azi, high=np.pi, low=-np.pi)
    #     local_azi_std = scipy.stats.circstd(local_azi, high=np.pi, low=-np.pi)
    #     local_ele_mean = np.mean(local_ele)
    #     local_ele_std = np.std(local_ele)
    #
    #     peak_stats.append([n*time_resolution_ms, [local_azi_mean, local_azi_std], [local_ele_mean, local_ele_std]])
    #
    # ### Return peak stats
    # return peak_stats








    # for idx, n in enumerate(peak_time_bins):
    #
    #     # RAW
    #
    #     local_azi = doa.data[0, :, n]
    #     local_ele = doa.data[1, :, n]
    #     local_dif = diffuseness.data[0, :, n]
    #
    #     local_azi_mean = scipy.stats.circmean(local_azi, high=np.pi, low=-np.pi)
    #     local_azi_std = scipy.stats.circstd(local_azi, high=np.pi, low=-np.pi)
    #     local_ele_mean = np.mean(local_ele)
    #     local_ele_std = np.std(local_ele)
    #
    #     if plot:
    #         fig = plt.figure()
    #         plt.suptitle('FREQ BIN '+str(n) + ' IDX ' + str(idx))
    #         ax = fig.add_subplot(311)
    #         cmap = plt.cm.get_cmap("copper")
    #         plt.grid()
    #         plt.xlim(-np.pi,np.pi)
    #         plt.ylim(-np.pi/2,np.pi/2)
    #
    #         plt.scatter(local_azi, local_ele, marker='o', c=local_dif, cmap=cmap, vmin=0, vmax=1)
    #         plt.scatter(local_azi_mean, local_ele_mean, c='red', s=20, marker='+')
    #         ax.add_patch(Ellipse(xy=(local_azi_mean,local_ele_mean), width=local_azi_std,height=local_ele_std, alpha=0.5))
    #         ax.add_patch(Ellipse(xy=(local_azi_mean,local_ele_mean), width=3*local_azi_std,height=3*local_ele_std, alpha=0.1))
    #
    #     # ENERGY DENSITY MASK
    #
    #     # must take care of nans
    #     local_azi = []
    #     local_ele = []
    #     local_dif = []
    #     for k in range(doa_Menergy.get_num_frequency_bins()):
    #         if not np.isnan(doa_Menergy.data[0, k, n]):
    #             local_azi.append(doa_Menergy.data[0, k, n])
    #         if not np.isnan(doa_Menergy.data[1, k, n]):
    #             local_ele.append(doa_Menergy.data[1, k, n])
    #         if not np.isnan(diffuseness_Menergy.data[0, k, n]):
    #             local_dif.append(diffuseness_Menergy.data[0, k, n])
    #
    #     # If we don't have values, just skip this window
    #     if len(local_azi) == 0 or len(local_ele) == 0 or len(local_dif) == 0:
    #         continue
    #
    #     local_azi = np.asarray(local_azi)
    #     local_ele = np.asarray(local_ele)
    #     local_dif = np.asarray(local_dif)
    #
    #     local_azi_mean = scipy.stats.circmean(local_azi, high=np.pi, low=-np.pi)
    #     local_azi_std = scipy.stats.circstd(local_azi, high=np.pi, low=-np.pi)
    #     local_ele_mean = np.mean(local_ele)
    #     local_ele_std = np.std(local_ele)
    #
    #     if plot:
    #         ax = fig.add_subplot(312)
    #         cmap = plt.cm.get_cmap("copper")
    #         plt.grid()
    #         plt.xlim(-np.pi, np.pi)
    #         plt.ylim(-np.pi / 2, np.pi / 2)
    #         ax.scatter(local_azi, local_ele, marker='o', c=local_dif, cmap=cmap, vmin=0, vmax=1)
    #         ax.scatter(local_azi_mean, local_ele_mean, c='red', s=20, marker='+')
    #         ax.add_patch(Ellipse(xy=(local_azi_mean,local_ele_mean), width=local_azi_std,height=local_ele_std, alpha=0.5))
    #         ax.add_patch(Ellipse(xy=(local_azi_mean,local_ele_mean), width=3*local_azi_std,height=3*local_ele_std, alpha=0.1))
    #
    #
    #
    #     # DIFFUSENESS MASK
    #
    #     # must take care of nans
    #     local_azi = []
    #     local_ele = []
    #     local_dif = []
    #     for k in range(doa_Mdif.get_num_frequency_bins()):
    #         if not np.isnan(doa_Mdif.data[0, k, n]):
    #             local_azi.append(doa_Mdif.data[0, k, n])
    #         if not np.isnan(doa_Mdif.data[1, k, n]):
    #             local_ele.append(doa_Mdif.data[1, k, n])
    #         if not np.isnan(diffuseness_Mdif.data[0, k, n]):
    #             local_dif.append(diffuseness_Mdif.data[0, k, n])
    #
    #     # If we don't have values, just skip this window
    #     if len(local_azi) == 0 or len(local_ele) == 0 or len(local_dif) == 0:
    #         continue
    #
    #     local_azi = np.asarray(local_azi)
    #     local_ele = np.asarray(local_ele)
    #     local_dif = np.asarray(local_dif)
    #
    #     local_azi_mean = scipy.stats.circmean(local_azi, high=np.pi, low=-np.pi)
    #     local_azi_std = scipy.stats.circstd(local_azi, high=np.pi, low=-np.pi)
    #     local_ele_mean = np.mean(local_ele)
    #     local_ele_std = np.std(local_ele)
    #
    #     if plot:
    #         ax = fig.add_subplot(313)
    #         cmap = plt.cm.get_cmap("copper")
    #         plt.grid()
    #         plt.xlim(-np.pi, np.pi)
    #         plt.ylim(-np.pi / 2, np.pi / 2)
    #         ax.scatter(local_azi, local_ele, marker='o', c=local_dif, cmap=cmap, vmin=0, vmax=1)
    #         ax.scatter(local_azi_mean, local_ele_mean, c='red', s=20, marker='+')
    #         ax.add_patch(Ellipse(xy=(local_azi_mean,local_ele_mean), width=local_azi_std,height=local_ele_std, alpha=0.5))
    #         ax.add_patch(Ellipse(xy=(local_azi_mean,local_ele_mean), width=3*local_azi_std,height=3*local_ele_std, alpha=0.1))
    #
    #     #
    #     # # NOW find the contiguous vertical regions and compute a different centroid for each
    #     # # numeration in python indexing style
    #     # region_starts = []
    #     # region_ends = []
    #     # for k in range(diffuseness_energy_mask.get_num_frequency_bins()):
    #     #     val = diffuseness_energy_mask.data[0, k, n]
    #     #     previous_val = diffuseness_energy_mask.data[0, k - 1, n] if k > 0 else np.nan
    #     #
    #     #     if val == 1 and np.isnan(previous_val):
    #     #         # start here
    #     #         region_starts.append(k)
    #     #
    #     #     if np.isnan(val) and previous_val == 1:
    #     #         # ends here
    #     #         region_ends.append(k)
    #     #
    #     # # if last bin was active, close it here
    #     # if diffuseness_energy_mask.data[0, -1, n] == 1:
    #     #     region_ends.append(diffuseness_energy_mask.get_num_frequency_bins())
    #     #
    #     # assert len(region_starts) == len(region_ends)
    #     #
    #     # if plot:
    #     #     ax = fig.add_subplot(313)
    #     #     cmap = plt.cm.get_cmap("copper")
    #     #     plt.grid()
    #     #     plt.xlim(-np.pi, np.pi)
    #     #     plt.ylim(-np.pi / 2, np.pi / 2)
    #     #
    #     # # ITERATE OVER DIFFERENT REGIONS
    #     # for idx in range(len(region_starts)):
    #     #     start_k = region_starts[idx]
    #     #     end_k = region_ends[idx]
    #     #
    #     #     region_length_th = np.floor(diffuseness_energy_mask.get_num_frequency_bins() / 5)
    #     #     region_length = end_k - start_k
    #     #     if region_length >= region_length_th:
    #     #         local_azi = doa.data[0, start_k:end_k, n]
    #     #         local_ele = doa.data[1, start_k:end_k, n]
    #     #         local_dif = diffuseness.data[0, start_k:end_k, n]
    #     #
    #     #         local_azi_mean = scipy.stats.circmean(local_azi, high=np.pi, low=-np.pi)
    #     #         local_azi_std = scipy.stats.circstd(local_azi, high=np.pi, low=-np.pi)
    #     #         local_ele_mean = np.mean(local_ele)
    #     #         local_ele_std = np.std(local_ele)
    #     #
    #     #         if plot:
    #     #             plt.scatter(local_azi, local_ele, marker='o', c=local_dif, cmap=cmap, vmin=0, vmax=1)
    #     #             plt.scatter(local_azi_mean, local_ele_mean, c='red', s=20, marker='+')
    #     #             ax.add_patch(Ellipse(xy=(local_azi_mean, local_ele_mean), width=local_azi_std, height=local_ele_std,
    #     #                                  alpha=0.5))
    #     #             ax.add_patch(
    #     #             Ellipse(xy=(local_azi_mean, local_ele_mean), width=3 * local_azi_std, height=3 * local_ele_std,
    #     #                     alpha=0.1))
    #     #
    #     #         # Keep this values
    #     #         peak_stats.append([n, [local_azi_mean, local_azi_std], [local_ele_mean, local_ele_std]])
    #
    #     # return peak_stats



def plot_peak_statistics(peak_stats,title=''):
    fig = plt.figure()
    plt.suptitle(title)

    x = [peak[0] for peak in peak_stats]

    ax = fig.add_subplot(211)
    plt.grid()
    # azi
    mean = [peak[1][0] for peak in peak_stats]
    std = [peak[1][1] for peak in peak_stats]
    ax.errorbar(x, mean, yerr=std, fmt='o')
    plt.ylim(-np.pi, np.pi)
    plt.xlim(0,processing_window_ms)
    ax.set_title('Azimuth estimates')
    for xpos in x:
        plt.axvline(x=xpos, color='k', alpha=0.3)

    if peak_modelled:
        for peak in peak_modelled:
            t = peak[0]
            azi = peak[1][0]
            plt.scatter(t,azi,color='r')
            plt.axvline(x=t, color='r', alpha=0.3)


    # ele
    ax = fig.add_subplot(212)
    plt.grid()
    mean = [peak[2][0] for peak in peak_stats]
    std = [peak[2][1] for peak in peak_stats]
    ax.errorbar(x, mean, yerr=std, fmt='o')
    plt.ylim(-np.pi / 2, np.pi / 2)
    plt.xlim(0,processing_window_ms)
    ax.set_title('Elevation estimates')
    for x in x:
        plt.axvline(x=x, color='k', alpha=0.3)

    if peak_modelled:
        for peak in peak_modelled:
            t = peak[0]
            ele = peak[1][1]
            plt.scatter(t,ele,color='r')
            plt.axvline(x=t, color='r', alpha=0.3)







################################################################################################
################################################################################################


# ANALIZE EACH OF THE SOFA FILES
for sofa_file_path in sofa_files:

    # Print name and check that it's valid
    sofa_file = SOFAAmbisonicsDRIR(sofa_file_path, 'r')
    print(sofa_file_path)
    print('File is a valid AmbisonicsDRIR: ', sofa_file.isValid())

    # Get ambisonics order
    print('Ambisonics Order: ', sofa_file.getGlobalAttributeValue('AmbisonicsOrder'))
    print('Dimensions:')
    sofa_file.printSOFADimensions()
    print('Variables')
    sofa_file.printSOFAVariables()

    # Get dimensions
    Mdim = sofa_file.getDimensionSize('M')
    Rdim = sofa_file.getDimensionSize('R')
    Edim = sofa_file.getDimensionSize('E')
    Ndim = sofa_file.getDimensionSize('N')
    print('\n')

    # Load IR data into memory
    recorded_data = sofa_file.getDataIR()
    recorded_sample_rate = sofa_file.getSamplingRate()
    if type(recorded_sample_rate) is np.ndarray:
        recorded_sample_rate = int(recorded_sample_rate[0])
    recorded_ambisonics_ordering = str(sofa_file.getDataIRChannelOrdering())
    recorded_ambisonics_normalization = str(sofa_file.getDataIRNormalization())

    ## Set this parameter now that we know the recording's sample rate
    processing_window_samples = float(
        recorded_sample_rate) * processing_window_ms / 1000  # TODO! COMPUTE mixing point FOR THAT!!

    # just take it from first W measurement
    # recorded_t60 = acoustics.room.t60_impulse(recorded_data[0, 0, 0, :], recorded_sample_rate, np.array([1000]), rt='t30')[0]
    ## TODO!!!!
    recorded_t60 = 2.0

    print('recorded_t60', recorded_t60)


    #### ROOM GEOMETRICAL MODEL

    wall0 = np.array([0, 0, 0])
    wall1 = np.array([7.7, 7.7, 4])
    sound_speed = 343.


    # Iterate over the IR data
    for m in range(Mdim):
        print('Measurement (M):', m)

        listener_position = sofa_file.getListenerPositionValues()[m]
        listener_coordinates = sofa_file.getListenerPositionInfo()[-1]
        if listener_coordinates == 'spherical':
            listener_position = np.asarray(spherical_to_cartesian_degree(listener_position))
        # listener_view = sofa_file.getListenerViewValues()[m]
        print('LISTENER POSITION: ',listener_position)

        source_position = sofa_file.getSourcePositionValues()[m]
        source_coordinates = sofa_file.getSourcePositionInfo()[-1]
        if source_coordinates == 'spherical':
            source_position = np.asarray(spherical_to_cartesian_degree(source_position))
        # source_view = sofa_file.getSourceViewValues()[m]
        print('SOURCE POSITION: ', source_position)


        # for e in range(Edim):
        for e in range(3,4):
            print('\t\tEmitter (E):', e)

            emitter_position = sofa_file.getEmitterPositionValues()[e, :, m]
            emitter_coordinates = sofa_file.getEmitterPositionInfo()[-1]
            if emitter_coordinates == 'spherical':
                emitter_position = np.asarray(spherical_to_cartesian_degree(emitter_position))
            emitter_position += source_position  # source translation
            print('EMITTER POSITION: ', emitter_position)



            ################################################################################################
            ## ROOM MODEL

            if compute_room_model:
                peak_modelled=[]

                image_positions = compute_image_source_positions(wall0,wall1,emitter_position,order=reflection_order)
                for pos in image_positions:
                    v = pos - listener_position
                    delay = np.linalg.norm(v) / (sound_speed / 1000 ) # ms
                    peak_modelled.append([delay, cartesian_to_spherical_radian(v)])
                    # print('-->', delay, cartesian_to_spherical_degree(v), cartesian_to_spherical_radian(v))


                plt.figure()
                plt.suptitle('MODELLED EARLY REFLECTIONS FOR EMITTER:'+str(e))
                ax=plt.subplot(211)
                plt.grid()
                ax.set_title('Azimuth estimates')
                for peak in peak_modelled:
                    delay = peak[0]
                    val = peak[1][0]
                    plt.scatter(delay,val,color='b')
                    plt.xlim(0,processing_window_ms)
                    plt.ylim(-np.pi,np.pi)
                    plt.axvline(x=delay, color='k', alpha=0.3)
                ax=plt.subplot(212)
                ax.set_title('Elevation estimates')
                plt.grid()
                for peak in peak_modelled:
                    delay = peak[0]
                    val = peak[1][1]
                    plt.scatter(delay, val, color='b')
                    plt.xlim(0, processing_window_ms)
                    plt.ylim(-np.pi/2, np.pi/2)
                    plt.axvline(x=delay, color='k', alpha=0.3)



            ################################################################################################
            # RECORDED IRS
            if compute_recorded:
                print('Processing recorded IR...')

                ### Get data from SOFA file
                recorded_ir = recorded_data[m,:4,e,:int(processing_window_samples)]

                ### Compute the stuff
                recorded_peak_stats = compute_peak_statistics(recorded_ir,
                                                              recorded_sample_rate,
                                                              recorded_ambisonics_ordering,
                                                              recorded_ambisonics_normalization,
                                                              plot=True,
                                                              plot_title = 'Recorded - ')

                plot_peak_statistics(recorded_peak_stats, title='ESTIMATED SIMULATED PEAKS FOR w=' + str(analysis_window_size))


                print('Processing recorded IR finished!')

            ################################################################################################
            # SIMULATED IRS

            if compute_simulated:
                print('Processing simulated IR...')

                ## Load matlab interface
                try:
                    import matlab_wrapper
                except ImportError:
                    print('matlab_wrapper not available')

                ## Start Matlab Session
                matlab = matlab_wrapper.MatlabSession(matlab_root=matlab_root)

                ## Add script into the matlab path
                matlab.eval('addpath ' + rir_root)

                ## Pass parameters to matlab
                matlab.eval('clear')
                matlab.put('c', float(sound_speed))
                matlab.put('procFs', float(recorded_sample_rate))
                matlab.put('sphLocation', listener_position)  # Listener position
                matlab.put('s',emitter_position) # Source position
                matlab.put('L', wall1-wall0) # Room dimensions
                matlab.put('beta',float(recorded_t60))
                matlab.put('sphType',sphType)
                matlab.put('sphRadius',float(sphRadius))
                matlab.put('mic',capsule_positions)
                matlab.put('N_harm', float(n_harm))
                matlab.put('nsample', float(processing_window_samples)) # IR length
                matlab.put('K', float(oversampling))
                matlab.put('order',float(reflection_order))
                matlab.put('refl_coeff_ang_dep',float(refl_coeff_ang_dep))
                matlab.put('HP',float(high_pass))
                matlab.put('src_type',source_directivity)

                ## Evaluate expression
                matlab.eval('run_smir_generator_from_python')

                ## Get the values back into python
                simulated_ir = matlab.get('S')
                simulated_ambisonics_ordering = 'acn'
                simulated_ambisonics_normalization = 'sn3d'

                ## Analize IR
                simulated_peak_stats = compute_peak_statistics(simulated_ir,
                                                               recorded_sample_rate,
                                                               simulated_ambisonics_ordering,
                                                               simulated_ambisonics_normalization,
                                                               plot=True,
                                                               plot_title='Simulated - ')

                plot_peak_statistics(simulated_peak_stats, title='ESTIMATED SIMULATED PEAKS FOR w=' + str(analysis_window_size))

                ## Close matlab
                del matlab

                print('Processing simulated IR finished!')

plt.show()



################################################################################################

pass