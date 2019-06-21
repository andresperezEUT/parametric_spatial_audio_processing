import parametric_spatial_audio_processing as psa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import time


len = 1. # s
audio_path = "/Volumes/Dinge/DCASE2019/foa_dev/split1_ir0_ov1_1.wav"
data, sr = sf.read(audio_path)
data = data[:int(len*sr)]
signal = psa.Signal(data.T, sr, ordering='acn',norm='sn3d')
stft = psa.Stft.fromSignal(signal)

r=5

t_a = []
t_b = []

for i in range(1000):

    start = time.time()
    a = stft.compute_ita_re(r)
    end = time.time()
    # print('a', end - start)
    t_a.append(end-start)

    start = time.time()
    b = stft.compute_ksi_re(r)
    end = time.time()
    # print('b', end - start)
    t_b.append(end - start)

    # psa.plot_directivity(a)
    # psa.plot_directivity(b)
    # plt.show()

    # assert np.allclose(a.data, b.data)

print(np.mean(t_a), np.mean(t_b))

