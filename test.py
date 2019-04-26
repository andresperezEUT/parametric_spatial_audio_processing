

## TEST
from matplotlib.colors import LogNorm

import parametric_spatial_audio_processing as psa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

audio_path = "/Volumes/Dinge/ambiscaper/database/soundscape0/soundscape0.wav"
data, sr = sf.read(audio_path)


# n_samples = 20
# sr = 48000
# data = np.zeros((n_samples,4))
# data[0,:] = 0.5
# data[1,:] = 0.25
# data[2,:] = 0.125

signal = psa.Signal(data[:30000].T, sr, ordering='acn',norm='sn3d')

signal_limit = signal.limit_envelope(th=0.005)
print(np.shape(signal_limit.data))
print(np.shape(signal.data))
print(signal_limit.data)


for i in range(np.shape(signal_limit.data)[1]):
    print(signal_limit.data[:,i])
psa.plot_signal(signal,'Signal Waveform')
psa.plot_signal(signal_limit,'limit')
plt.show()
#
# print(signal.data)
#
# doa_n = psa.compute_DOA(signal)
# print(doa_n.data)


# stft = psa.Stft(signal)

stft = psa.Stft.fromSignal(signal)
directivity = psa.compute_directivity(stft)


doa = psa.compute_DOA(stft)
h = psa.plot_doa_2d_histogram(doa)
print(h)
# masked1 = stft.apply_softmask(directivity,0.9)



# dpd = psa.compute_DPD_test(stft,1,1,2.)


# psa.plot_magnitude_spectrogram(stft)
# psa.plot_mask(directivity,"directivity")
# psa.plot_mask(masked1,"masked1")
plt.show()

# input('Press ENTER to exit')





# print(doa_kn.data)


# # time-frequency
# stft = psa.Stft(signal)
# # psa.plot_magnitude_spectrogram(stft,'Signal STFT Spectrogram')
#
#
# p_n = psa.compute_sound_pressure(signal)
# # psa.plot_signal(p_n)
#
# p_kn = psa.compute_sound_pressure(stft)
# # psa.plot_magnitude_spectrogram(p_kn)
#
# u_n = psa.compute_particle_velocity(signal)
# # psa.plot_signal(u_n)
#
# u_kn = psa.compute_particle_velocity(stft)
# # psa.plot_magnitude_spectrogram(u_kn)
#
# # for f in range(p_kn.get_num_frequency_bins()):
# #     for n in range(p_kn.get_num_time_bins()):
# #         print(p_kn.data[0,f,n])
#
# i_n = psa.compute_intensity_vector(signal)
# # psa.plot_signal(i_n)
#
# i_kn = psa.compute_intensity_vector(stft)
# # psa.plot_magnitude_spectrogram(i_kn)
#
# e_n = psa.compute_energy_density(signal)
# # psa.plot_signal(e_n)
#
# e_kn = psa.compute_energy_density(stft)
#
# # print('e')
# # print(e_n.data[-1,-1])
# # print(e_kn.data[-1,-1,-1])
#
# # for f in range(e_kn.get_num_frequency_bins()):
# #     for n in range(e_kn.get_num_time_bins()):
# #         print(i_kn.data[0,f,n],i_kn.data[1,f,n],i_kn.data[2,f,n])
#
# # psa.plot_magnitude_spectrogram(e_kn)
#
#
#
# doa_n = psa.compute_DOA(signal)
# # psa.plot_doa(doa_n)
#
#
# # psa.plot_signal(doa_n)
#
#
#
# doa_kn = psa.compute_DOA(stft)
# # psa.plot_doa(doa_kn)
# # plt.figure(10)
# # psa.plot_magnitude_spectrogram(doa_kn)
#
# # print(doa_kn.data[0,0,0])
#
# f_n = psa.compute_diffuseness(signal)
# psa.plot_diffuseness(f_n)
# # plt.figure(plt_idx)
# # plt_idx+=1
# # psa.plot_signal(f_n)
#
# f_kn = psa.compute_diffuseness(stft)
#
# # for f in range(f_kn.get_num_frequency_bins()):
# #     for n in range(f_kn.get_num_time_bins()):
# #         print(f_kn.data[0,f,n])
# psa.plot_diffuseness(f_kn)
# # print(np.shape(f_kn.data))
# # print(f_kn.data)
#
# plt.show()
#
# plt.figure()
# plt.subplot(4, 1, 1)
# c0 = psa.spatial_covariance_matrix(stft,0)
# plt.pcolormesh(np.abs(c0),norm=LogNorm(vmin=np.abs(c0).min(), vmax=np.abs(c0).max()))

# c1 = psa.spatial_covariance_matrix(stft,1)
# plt.subplot(4, 1, 2)
# plt.pcolormesh(np.abs(c1))
#
# c2 = psa.spatial_covariance_matrix(stft,2)
# plt.subplot(4, 1,3)
# plt.pcolormesh(np.abs(c2))
#
# c3 = psa.spatial_covariance_matrix(stft,3)
# plt.subplot(4, 1,4)
# plt.pcolormesh(np.abs(c3))


# plt.show()

# SCM...

# def sign(z):
    # return z/np.abs(z)

# s = psa.Stft(np.arange(10),
#              np.arange(10),
#              np.sqrt(np.abs(stft.data))*sign(stft.data),
#              # stft.data[:,:10,:10],
#              stft.sample_rate)

# plt.figure()
# psa.plot_magnitude_spectrogram(s)
# plt.show()

# databin = stft.data[:,0]
#
#
# M = stft.get_num_channels()
# # print('M',M)
# K = stft.get_num_frequency_bins()
# # print('K',K)
# N = stft.get_num_time_bins()
# # print('N',N)

# we compute the correlation of each time-frequency element
# across the ambisonics channels
# each correlation will be a matrix of M by M
# Then we rearrange the data in form of a "matrix of matrices"
# with shape [M*K, M*N]

# scm = np.ndarray((3+2+1,K,N),dtype=complex)
#
# for k in range(K):
#     for n in range(N):
#         a = stft.data[:,k,n,np.newaxis] # add dimension
#         a = np.sqrt(np.abs(a))*sign(a)
#         # print('a',a)
#
#         cov = np.outer(a,np.transpose(np.conjugate(a)))
#
#
#         # print('cov',cov)
#         # each one of these elements will be copied at appropiate location in scm matrox
#
#         # 01
#         scm[0, k, n] = (a[0]-a[1])[0]
#         # 02
#         scm[1, k, n] = (a[0]-a[2])[0]
#         # 03
#         scm[2, k, n] = (a[0]-a[3])[0]
#         # 12
#         scm[3, k, n] = (a[1]-a[2])[0]
#         # 13
#         scm[4, k, n] = (a[1]-a[3])[0]
#         # 23
#         scm[5, k, n] = (a[2]-a[3])[0]
#         # # 01
        # scm[0, k, n] = cov[0,1]
        # # 02
        # scm[1, k, n] = cov[0,2]
        # # 03
        # scm[2, k, n] = cov[0,3]
        # # 12
        # scm[3, k, n] = cov[1,2]
        # # 13
        # scm[4, k, n] = cov[1,3]
        # # 23
        # scm[5, k, n] = cov[2,3]

        # for r in range(M):
            # for c in range(M):
                # scm[(M*r)+k,(M*c)+n] = cov[r,c]



                # plt.pcolormesh(np.abs(cov))
                # plt.colorbar()
                # plt.show()
                # print(cov[r,c])


# print(scm)
#
# vmin = np.min(np.abs(scm))
# vmax = np.max(np.abs(scm))
#
# for i in range(6):
#     plt.subplot(6, 1, i + 1)
#     plt.pcolormesh(np.angle(scm[i,:,:]),norm=LogNorm(vmin=1e-10, vmax=vmax))
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.colorbar()
# plt.show()
#
# # plt.figure()
# sns.heatmap(np.abs(scm),linewidths=.5)
# plt.show()

# plt.figure()
# sns.heatmap(np.angle(cov))
# plt.show()

# print(scm)
# plt.pcolormesh(np.abs(scm[:2*K,:2*N]),norm=LogNorm(vmin=1e-100, vmax=1e-15))
# plt.colorbar()
# plt.show()

#
# # stft has dims [M,K,N]
# # we want to reduce to [M*K,N]
# reshaped_stft = stft.data.reshape(-1,stft.data.shape[-1])
# print('reshaped',reshaped_stft.shape)
#
# # hermitian transpose
# # reshaped_stft_H = np.conj(np.transpose(reshaped_stft))
# # reshaped_stft_H = np.transpose(np.conj(stft.data.reshape(-1,stft.data.shape[1])))
# # print('reshaped_H',reshaped_stft_H.shape)
#
# # scm
# scm = np.outer(stft.data,stft.data)
# print('scm',scm.shape)
#
# plt.pcolormesh(np.abs(scm),norm=LogNorm(vmin=1e-10, vmax=1))
# plt.show()

# from parametric_spatial_audio_processing.util import compute_signal_envelope
# from parametric_spatial_audio_processing.util import find_contiguous_region
#
# a = np.zeros(16)
# a[3:15] = 1.0
# a[7]=0.0
# q = find_contiguous_region(a)



# print(q)
#

# c = compute_signal_envelope(signal.data[0],windowsize=1024)
# print(c)
# q = find_contiguous_region(c,1024,0.01)
#
# z = psa.segmentate_audio(signal,windowsize=1024,th=0.01)
#
#
# print(q)
#
# plt.figure()
# plt.plot(c)
# plt.grid()
# plt.show()
# #
# print(c)
#
# plt.figure()
# plt.plot(signal.data[0],'b',c,'r--')
# plt.show()
#
