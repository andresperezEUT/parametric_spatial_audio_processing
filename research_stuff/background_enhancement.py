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

## Open open the noise background

# bg_path = '/Volumes/Dinge/ambiscaper/background/noise2/noise2.wav'
# bg_path = '/Volumes/Dinge/ambiscaper/background/isf/isf_acn_sn3d.wav'
bg_path = '/Volumes/Dinge/ambiscaper/background/isf/isf_bformat_plugin_filter.wav'
# bg_path = '/Volumes/Dinge/ambiscaper/background/isf/isf.wav'
bg,sr = sf.read(bg_path)
bg = bg.T * 0.1
bg_signal = psa.Signal(bg, sr, 'acn', 'sn3d')
bg_stft = psa.Stft.fromSignal(bg_signal,
                             window_size=analysis_window_size,
                              window_overlap=window_overlap,
                              nfft=fft_size)
# psa.plot_signal(bg_signal,title='groundtruth background')
# psa.plot_magnitude_spectrogram(bg_stft,title='groundtruth  background')
# psa.plot_diffuseness(psa.compute_diffuseness(bg_stft),'groundtruth bg diffuseness')

# Open the foreground
fg_path = '/Volumes/Dinge/ambiscaper/background/foreground/foreground.wav'
fg,sr = sf.read(fg_path)
fg = fg.T * 5
fg_signal = psa.Signal(fg, sr, 'acn', 'sn3d')
fg_stft = psa.Stft.fromSignal(fg_signal,
                              window_size=analysis_window_size,
                              window_overlap=window_overlap,
                              nfft=fft_size)
# psa.plot_signal(fg_signal,title='groundtruth foreground')
# psa.plot_magnitude_spectrogram(fg_stft,title='groundtruth foreground')
# psa.plot_diffuseness(psa.compute_diffuseness(fg_stft),'groundtruth fg diffuseness')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
  # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


### First approach

s = bg[:,:240000] + fg[:,:240000]
b = psa.Signal(s, sr, 'acn', 'sn3d')
psa.plot_signal(b,title='input')

B = psa.Stft.fromSignal(b,
                        window_size=analysis_window_size,
                        window_overlap=window_overlap,
                        nfft=fft_size)

psa.plot_magnitude_spectrogram(B,title='B')

diffuseness_B = psa.compute_diffuseness(B)
directivity_B = psa.compute_directivity(B)
psa.plot_diffuseness(diffuseness_B,'diffuseness_B')

# directivity_B_mask = directivity_B.compute_mask(th=0.95)
# psa.plot_mask(directivity_B_mask,title='directivity_B_mask 0.95')
#
# diffuseness_B_mask = diffuseness_B.compute_mask(th=0.95)
# psa.plot_mask(diffuseness_B_mask,title='diffuseness mask 0.95')


B_dir = B.apply_mask(directivity_B)
psa.plot_magnitude_spectrogram(B_dir,title='B_dir')

B_dif = B.apply_mask(diffuseness_B)
psa.plot_magnitude_spectrogram(B_dif,title='B_dif')

b_dif = psa.Signal.fromStft(B_dif,
                            window_size=analysis_window_size,
                            window_overlap=window_overlap,
                            nfft=fft_size)

# background_t = psa.Signal.fromStft(background_stft,analysis_window_size,window_overlap,fft_size)
# foreground_t = psa.Signal.fromStft(foreground_stft,analysis_window_size,window_overlap,fft_size)

psa.plot_signal(bg_signal,title='groundtruth background')
psa.plot_signal(b_dif,title='estimated background')



#### Second approach


## Compute DOA on the foreground signal
# doa_B = psa.compute_DOA(B)
# psa.plot_doa(doa_B,title='doa_B')
#
# ## Filter by energy density
# energy_density_B = psa.compute_energy_density(B)
# psa.plot_magnitude_spectrogram(energy_density_B,title='enegy_density_B')






# output_path = '/Volumes/Dinge/ambiscaper/background/resynth/'
#
# sf.write(output_path+'input.wav',s.T,sr)
# sf.write(output_path+'foreground.wav',psa.convert_bformat_fuma_2_acn(foreground_t.data).T,sr)
# sf.write(output_path+'background.wav',psa.convert_bformat_fuma_2_acn(background_t.data).T,sr)


mir_eval.separation.validate(b.data,b_dif.data)

# sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(fg0,beamed_foreground_signal)
sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources_framewise(b.data,b_dif.data)

print('++++++++')
print('Framewise evaluation')
print('sdr:',sdr)
print('sir:',sir)
print('sar:',sar)
print('perm:',perm)
print('++++++++')


####
# np.mean(np.power(b.data[0,:]),2)
# np.mean(np.power(b_dif.data - b.data[0,:]),2)




plt.show()
pass


