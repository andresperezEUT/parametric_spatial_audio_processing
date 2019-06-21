import numpy as np
import soundfile as sf
import sys
import matplotlib.pyplot as plt
import scipy.stats
import scipy.signal as sg
from matplotlib.colors import LogNorm

pp = "/Users/andres.perez/source/parametric_spatial_audio_processing"
sys.path.append(pp)
import parametric_spatial_audio_processing as psa

import mir_eval
################################################################################################

## Parameters
processing_window_ms = 50
analysis_window_size = 512
window_overlap = analysis_window_size // 2
fmin = 125
fmax = 8000
fft_factor = 1
fft_size = analysis_window_size * fft_factor

## Open the file to analyze

# file_path = '/Volumes/Dinge/ambiscaper/background/background_anechoic_pad/background_anechoic_pad.wav'
file_path = '/Volumes/Dinge/ambiscaper/testing/len2/len2.wav'
data, sr = sf.read(file_path)

## Check diffuseness

signal = psa.Signal(data.T, sr, 'acn', 'sn3d')
psa.plot_signal(signal,title='waveform')

stft = psa.Stft.fromSignal(signal,
                           window_size=analysis_window_size,
                           window_overlap=window_overlap,
                           nfft=fft_size)
psa.plot_magnitude_spectrogram(stft,title='magnitude spectrogram')

diffuseness_stft = psa.compute_diffuseness(stft)
directivity_stft = psa.compute_directivity(stft)
psa.plot_diffuseness(diffuseness_stft,title='diffuseness')


## Apply diffuseness mask to separate background and foreground

background_stft = stft.apply_mask(diffuseness_stft)
foreground_stft = stft.apply_mask(directivity_stft)

psa.plot_magnitude_spectrogram(background_stft,title='background magnitude spectrogram')
psa.plot_magnitude_spectrogram(foreground_stft,title='foreground magnitude spectrogram')


## Find DOA on the foreground

doa = psa.compute_DOA(stft)
psa.plot_doa(doa,title='doa')

doa_foreground = psa.compute_DOA(foreground_stft)
psa.plot_doa(doa_foreground,title='doa_foreground')

doa_background = psa.compute_DOA(background_stft)
psa.plot_doa(doa_background,title='doa_background')

## Just get the most directive values

directivity_mask = directivity_stft.compute_mask(th=0.95)
psa.plot_mask(directivity_mask,title='directivity mask')

masked_doa_foreground = doa_foreground.apply_mask(directivity_mask)
psa.plot_doa(masked_doa_foreground,title='masked_doa_foreground')

## Compute DOA mean from the masked spectrogram in order to filter less energetic bins
local_azi = []
local_ele = []
for k in doa_foreground.range('k'):
    for n in doa_foreground.range('n'):
        if not np.any(np.isnan(masked_doa_foreground.data[:,k,n])):
            local_azi.append(masked_doa_foreground.data[0,k,n])
            local_ele.append(masked_doa_foreground.data[1,k,n])

mean_azi = scipy.stats.circmean(local_azi)
mean_ele = np.mean(local_azi)

print('azi, ele',str(mean_azi),str(mean_ele))


## Steer a beam on stft to the recovered direction
# I think it's supercardioid, but let's check it

# ambisonic gains
w = 1
x = np.cos(mean_azi) * np.cos(mean_ele)
y = np.sin(mean_azi) * np.cos(mean_ele)
z = np.sin(mean_ele)
ambi_gains = [w,x,y,z]

plt.figure()
plt.suptitle('beamformer')
azi_values = np.arange(0,2*np.pi,2*np.pi/360)
ele_values = np.arange(-np.pi/2,np.pi/2,np.pi/180)

plt.subplot(5,1,1,projection='polar')
plt.polar(azi_values,np.ones(np.size(azi_values)))
plt.subplot(5,1,2,projection='polar')
plt.polar(azi_values,np.abs(np.cos(azi_values)*np.cos(mean_ele))*x)
plt.subplot(5,1,3,projection='polar')
plt.polar(azi_values,np.abs(np.sin(azi_values) * np.cos(mean_ele))*y)
plt.subplot(5,1,4,projection='polar')
plt.polar(ele_values,np.abs(np.sin(ele_values))*z)

# plt.subplot(5,1,5,projection='polar')
# plt.polar(x_values,w + x_values*x + x_values*y + x_values*z)

# Steer in the foreground stft
k = foreground_stft.get_num_frequency_bins()
n = foreground_stft.get_num_time_bins()

beamed_foreground = np.ndarray((k,n),dtype='complex128')
for ch in foreground_stft.range('ch'):
    for k in foreground_stft.range('k'):
        for n in foreground_stft.range('n'):
            beamed_foreground[k,n] += foreground_stft.data[ch,k,n] * ambi_gains[ch]

plt.figure()
plt.suptitle('beamed foreground stft')
plt.pcolormesh(np.abs(beamed_foreground),norm=LogNorm())

t, beamed_foreground_signal = sg.istft(beamed_foreground,
                                        fs=stft.sample_rate,
                                        window='hann',
                                        nperseg=analysis_window_size,
                                        noverlap=analysis_window_size//2,
                                        nfft=analysis_window_size)
beamed_foreground_signal = beamed_foreground_signal[:signal.get_num_frames()]

plt.figure()
plt.suptitle('beamed foreground signal')
plt.plot(beamed_foreground_signal)


## Inverse FFT to recover signals

resynth = psa.Signal.fromStft(stft,analysis_window_size,window_overlap,fft_size)
background_t = psa.Signal.fromStft(background_stft,analysis_window_size,window_overlap,fft_size)
foreground_t = psa.Signal.fromStft(foreground_stft,analysis_window_size,window_overlap,fft_size)

psa.plot_signal(resynth,title='resynth')
psa.plot_signal(background_t,title='background')
psa.plot_signal(foreground_t,title='foreground')


#################### #################### #################### ####################
########### Compare separated source with original

# fg0_path = '/Volumes/Dinge/ambiscaper/background/background_anechoic_pad/source/fg0.wav'
fg0_path = '/Volumes/Dinge/ambiscaper/testing/len2/source/fg0.wav'
fg0, sr = sf.read(fg0_path)

plt.figure()
plt.subplot(311)
plt.plot(beamed_foreground_signal)
plt.subplot(312)
plt.plot(fg0)
plt.subplot(313)
plt.plot(np.power(beamed_foreground_signal,2)-np.power(fg0,2))

### SEPARATION STATISTICS

fg0 = fg0[:,np.newaxis].T
beamed_foreground_signal = beamed_foreground_signal[:,np.newaxis].T

mir_eval.separation.validate(fg0,beamed_foreground_signal)

# sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(fg0,beamed_foreground_signal)
sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources_framewise(fg0,beamed_foreground_signal)

print('++++++++')
print('Framewise evaluation')
print('sdr:',sdr)
print('sir:',sir)
print('sar:',sar)
print('perm:',perm)
print('++++++++')


####################

# Now compare background separation
bg0_path = '/Volumes/Dinge/ambiscaper/testing/len2/source/bg0.wav'
bg0, sr = sf.read(bg0_path)

plt.figure()
plt.subplot(311)
plt.plot(background_t.data[0,:])
plt.subplot(312)
plt.plot(bg0)
plt.subplot(313)
plt.plot(np.abs(np.power(background_t.data[0,:],2)-np.power(bg0,2)))

### SEPARATION STATISTICS

bg0 = bg0[:,np.newaxis].T
background_t = background_t.data[0,np.newaxis]

mir_eval.separation.validate(bg0,background_t)

# sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(fg0,beamed_foreground_signal)
sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources_framewise(bg0,background_t)

print('++++++++')
print('Framewise evaluation')
print('sdr:',sdr)
print('sir:',sir)
print('sar:',sar)
print('perm:',perm)
print('++++++++')





#################### #################### ####################
## Write results to file

output_path = file_path = '/Volumes/Dinge/ambiscaper/background/background_anechoic/resynthesis/'
sf.write(output_path+'background.wav',background_t.T,sr)
sf.write(output_path+'foreground.wav',psa.convert_bformat_fuma_2_acn(foreground_t.data).T,sr)
sf.write(output_path+'resynth.wav',psa.convert_bformat_fuma_2_acn(resynth.data).T,sr)
sf.write(output_path+'beamed_foreground_signal.wav',beamed_foreground_signal.T,sr)


plt.show()
pass


