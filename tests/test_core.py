
import pytest
import numpy as np
import soundfile as sf
from parametric_spatial_audio_processing.core import Signal, compute_DOA, compute_energy_density, compute_diffuseness
from parametric_spatial_audio_processing.core import Stft
from parametric_spatial_audio_processing.core import compute_sound_pressure
from parametric_spatial_audio_processing.core import compute_particle_velocity
from parametric_spatial_audio_processing.core import compute_intensity_vector
from parametric_spatial_audio_processing.core import c, p0, s
import scipy.signal as sg

def test_signal_init():

    # Data must be an audio numpy array with shape [samples,channels]
    # sample_rate must be an string
    # ordering must be a str from allowed_ordering_strings
    # norm must be a str from allowed_norm_strings

    # Incorrect data format
    incorrect_audio_data = [
        [[1.,2.,3.],[4.,5.,6.,]],
        None,
        3.5,
    ]
    sr = 48000
    for data in incorrect_audio_data:
        pytest.raises(TypeError,Signal,data,sr)

    # Incorrect sample rate
    audio_data = np.zeros([3,100])
    incorrect_sr = [
        '48000',
        48000.0,
        None,
        48000+3j
    ]
    for sr in incorrect_sr:
        pytest.raises(TypeError, Signal, audio_data, sr)

    # Invalid ordering string
    audio_data = np.zeros([3,100])
    sr = 48000
    incorrect_ordering = [
        'FuMa',
        'ACN',
        'fakeOrdering',
        123,
    ]
    for order in incorrect_ordering:
        pytest.raises(TypeError, Signal, audio_data, sr, order, 'fuma')

    # Invalid norm string
    audio_data = np.zeros([3,100])
    sr = 48000
    incorrect_norm = [
        'sn4d',
        'SN3D',
        'fakeNorm',
        123,
    ]
    for norm in incorrect_norm:
        pytest.raises(TypeError, Signal, audio_data, sr, 'fuma', norm)


    # Test correct instanciation
    audio_data = np.array([[1.,1.,1.,1.]]).T
    sr = 48000
    signal = Signal(audio_data,sr)
    # Signal internally transposes the data to shape [channels,samples]
    assert np.allclose(signal.data,audio_data)
    assert signal.sample_rate == sr

    # Test correct scaling
    # FuMa reduces W channel by sqrt(2)
    audio_data_fuma = np.array([[1./np.sqrt(2),1.0,1.0,1.0]]).T
    sr = 48000
    signal = Signal(audio_data_fuma, sr, norm='fuma')
    assert np.allclose(signal.data, np.ones([4,1]))
    # FuMa reduces W channel by sqrt(2)
    audio_data_n3d = np.array([[1,np.sqrt(3),np.sqrt(3),np.sqrt(3)]]).T
    sr = 48000
    signal = Signal(audio_data_n3d, sr, norm='n3d')
    assert np.allclose(signal.data, np.ones([4,1]))


    # Test correct ordering
    audio_data_acn = np.array([[0.,1.,2.,3.]]).T
    sr = 48000
    signal = Signal(audio_data_acn, sr, ordering='acn')
    # Acn -> fuma: [W Y Z X] -> [W X Y Z]
    assert np.allclose(signal.data, np.array([[0.,3.,1.,2.]]).T)


def test_signal_num_channels():

    audio_data_acn = np.zeros([4,1])
    sr = 48000
    signal = Signal(audio_data_acn, sr)
    assert signal.get_num_channels() == 4


def test_stft_init():

    # Two possibilities for data input:
    # 1.) Argument is a Signal instance
    #       In this case a new Stft will be created
    # 2.) Argument is *[t f, data, sample_rate]
    #       Then a new Stft will be created with the given atrributes


    # Test incorrect Stft num args
    incorrect_args = [
        [1,2],
        [1,2,3],
        [1,2,3,4,5],
    ]
    for arg in incorrect_args:
        pytest.raises(TypeError,Stft,*arg)


    # Test incorrect signal
    incorrect_signal_type = [
        np.zeros([4,1]),
        [1.,2.,3.,4.],
    ]
    for s in incorrect_signal_type:
        pytest.raises(TypeError,Stft,s)


    # Test correct stft instanciation from signal
    audio_data_acn = np.zeros([4, 1000])
    sr = 48000
    signal = Signal(audio_data_acn, sr)
    stft = Stft(signal)
    f, t, Zxx = sg.stft(signal.data, signal.sample_rate)

    assert np.allclose(f,stft.f)
    assert np.allclose(t,stft.t)
    assert np.allclose(Zxx,stft.data)
    assert sr == stft.sample_rate


    # Test incorrect Stft instanciation
    audio_data_acn = np.zeros([4, 1000])
    sr = 48000
    signal = Signal(audio_data_acn, sr)
    stft = Stft(signal)

    # Incorrect types / shapes
    incorrect_type_args = [
        # t
        [[1,2,3], stft.f, stft.data, stft.sample_rate],
        [np.ones([2,2]), stft.f, stft.data, stft.sample_rate],
        # f
        [stft.t, [1, 2, 3], stft.data, stft.sample_rate],
        [stft.t, np.ones([2, 2]), stft.data, stft.sample_rate],
        # data
        [stft.t, stft.f, [1,2,3], stft.sample_rate],
        [stft.t, stft.f, np.ones([2]), stft.sample_rate],
        # sr
        [stft.t, stft.f, stft.data, 48000.0],
        [stft.t, stft.f,stft.data, '48000'],
    ]
    for args in incorrect_type_args:
        pytest.raises(TypeError,Stft,*args)

    # Incorrect lengths
    incorrect_length_args = [
        [stft.t[1:], stft.f, stft.data, stft.sample_rate],
        [stft.t, stft.f[1:], stft.data, stft.sample_rate],
    ]
    for args in incorrect_length_args:
        pytest.raises(AssertionError,Stft,*args)


    # Test correct stft instanciation from stft
    audio_data_acn = np.zeros([4, 1000])
    sr = 48000
    signal = Signal(audio_data_acn, sr)
    stft1 = Stft(signal)
    stft2 = Stft(stft1.t,
                 stft1.f,
                 stft1.data,
                 stft1.sample_rate)

    assert np.allclose(stft1.t, stft2.t)
    assert np.allclose(stft1.f, stft2.f)
    assert np.allclose(stft1.data, stft2.data)
    assert np.allclose(stft1.sample_rate, stft2.sample_rate)


def test_stft_num_channels():

    audio_data_acn = np.zeros([4,1000])
    sr = 48000
    signal = Signal(audio_data_acn, sr)
    stft = Stft(signal)
    assert stft.get_num_channels() == 4


def test_compute_sound_pressure():

    # incorrect input types
    pytest.raises(TypeError,compute_sound_pressure,12345)
    pytest.raises(TypeError,compute_sound_pressure,np.zeros([1,2]))
    pytest.raises(TypeError,compute_sound_pressure,[[1,2],[3,4]])

    # Valid signal
    audio_data = np.ones([4, 1000])
    sr = 48000
    signal = Signal(audio_data,sr)
    p_n = compute_sound_pressure(signal)

    # sound pressure is W channel
    assert np.allclose(p_n.data,signal.data[[0],:])
    assert np.allclose(p_n.data,np.ones([1,1000]))

    # Valid stft
    audio_data = np.ones([4, 1000])
    sr = 48000
    stft = Stft(Signal(audio_data,sr))
    p_kn = compute_sound_pressure(stft)

    # sound pressure is W channel
    assert np.allclose(p_kn.data,stft.data[[0],:,:])


def test_compute_particle_velocity():

    # incorrect input types
    pytest.raises(TypeError,compute_particle_velocity,12345)
    pytest.raises(TypeError,compute_particle_velocity,np.zeros([1,2]))
    pytest.raises(TypeError,compute_particle_velocity,[[1,2],[3,4]])

    # Valid signal
    audio_data = np.ones([4,1000])
    sr = 48000
    signal = Signal(audio_data,sr)
    u_n = compute_particle_velocity(signal)

    # particle velocity are [1:] channels scaled by -1/(s*p0*c)
    scale = -1.0/(s*p0*c)
    assert np.allclose(u_n.data,signal.data[1:,:]*scale)
    assert np.allclose(u_n.data,np.ones([3,1000])*scale)

    # Valid stft
    audio_data = np.ones([4,1000])
    sr = 48000
    stft = Stft(Signal(audio_data,sr))
    u_kn = compute_particle_velocity(stft)

    # particle velocity are [1:] channels scaled by -1/(s*p0*c)
    scale = -1.0/(s*p0*c)
    assert np.allclose(u_kn.data,stft.data[1:,:,:]*scale)


def test_compute_intensity_vector():

    # incorrect input types
    pytest.raises(TypeError, compute_intensity_vector, 12345)
    pytest.raises(TypeError, compute_intensity_vector, np.zeros([1, 2]))
    pytest.raises(TypeError, compute_intensity_vector, [[1, 2], [3, 4]])

    # Valid signal
    audio_data = np.ones([4,10])
    sr = 48000
    signal = Signal(audio_data,sr)
    i_n = compute_intensity_vector(signal)

    # sound intensity is w*[x,y,z]
    p_n = compute_sound_pressure(signal)
    u_n = compute_particle_velocity(signal)

    groundtruth = np.zeros(shape=np.shape(u_n.data))
    for ch in range(u_n.get_num_channels()):
        groundtruth[ch,:] = 0.5 * p_n.data * u_n.data[[ch],:]

    assert np.allclose(i_n.data,groundtruth)

    # Valid stft
    audio_data = np.ones([4,1000])
    sr = 48000
    stft = Stft(Signal(audio_data,sr))
    i_kn = compute_intensity_vector(stft)

    # sound intensity is w*[x,y,z]
    p_kn = compute_sound_pressure(stft)
    u_kn = compute_particle_velocity(stft)

    groundtruth = np.zeros(np.shape(u_kn.data))
    for ch in range(u_kn.get_num_channels()):
        groundtruth[ch,:,:] = 0.5 * np.real(np.conjugate(p_kn.data) * u_kn.data[ch,:,:])

    assert np.allclose(groundtruth,i_kn.data)


def test_compute_DOA():
    # incorrect input types
    pytest.raises(TypeError, compute_DOA, 12345)
    pytest.raises(TypeError, compute_DOA, np.zeros([1, 2]))
    pytest.raises(TypeError, compute_DOA, [[1, 2], [3, 4]])

    # Valid signal at front
    # c[1,0,0] -> sph[0,0]
    audio_data = np.zeros([4, 10])
    audio_data[0,:] = 1.0
    audio_data[1,:] = 1.0
    sr = 48000
    signal = Signal(audio_data, sr)
    doa = compute_DOA(signal)
    groundtruth = np.zeros(shape=(2,10))

    assert np.allclose(doa.data,groundtruth)

    # Valid signal at left
    # c[0,1,0] -> sph[pi/2,0]
    audio_data = np.zeros([4, 10])
    audio_data[0,:] = 1.0
    audio_data[2,:] = 1.0
    sr = 48000
    signal = Signal(audio_data, sr)
    doa = compute_DOA(signal)
    groundtruth = np.zeros(shape=(2,10))
    groundtruth[0,:] = np.pi/2
    assert np.allclose(doa.data,groundtruth)

    # Valid signal below
    # c[0,0,-1] -> sph[0,-pi/2]
    audio_data = np.zeros([4, 10])
    audio_data[0,:] = 1.0
    audio_data[3,:] = -1.0
    sr = 48000
    signal = Signal(audio_data, sr)
    doa = compute_DOA(signal)
    groundtruth = np.zeros(shape=(2,10))
    groundtruth[1,:] = -np.pi/2
    # Azimuth value is not valid at +-pi/2
    assert np.allclose(doa.data[1],groundtruth[1])


    # Valid stft at front
    # c[1,0,0] -> sph[0,0]
    audio_data = np.zeros(shape=(4,1000))
    audio_data[0,:] = 1.0
    audio_data[1,:] = 1.0
    sr = 48000
    stft = Stft(Signal(audio_data,sr))
    # Manually put ones in all spectrum
    newdata = np.zeros(shape=(stft.get_num_channels(),
                              stft.get_num_frequency_bins(),
                              stft.get_num_time_bins()))
    newdata[0,:,:] = 1.0
    newdata[1,:,:] = 1.0
    stft.data = newdata
    doa = compute_DOA(stft)


    groundtruth = np.zeros(shape=(2,
                                  stft.get_num_frequency_bins(),
                                  stft.get_num_time_bins()))
    assert np.allclose(doa.data,groundtruth)



def test_compute_energy_density():

    # incorrect input types
    pytest.raises(TypeError, compute_energy_density, 12345)
    pytest.raises(TypeError, compute_energy_density, np.zeros([1, 2]))
    pytest.raises(TypeError, compute_energy_density, [[1, 2], [3, 4]])

    # Valid signal
    audio_data = np.ones([4, 10])
    sr = 48000
    signal = Signal(audio_data, sr)
    e_n = compute_energy_density(signal)

    groundtruth = np.zeros(shape=(1,10))
    u_n = compute_particle_velocity(signal)
    p_n = compute_sound_pressure(signal)
    groundtruth = ((p0/4.)*pow(np.linalg.norm(u_n.data,axis=0),2)) \
                  + ((1./(4*p0*pow(c,2)))*pow(np.abs(p_n.data),2))

    assert np.allclose(e_n.data,groundtruth)

    # Valid stft
    audio_data = np.ones([4, 1000])
    sr = 48000
    stft = Stft(Signal(audio_data, sr))
    e_kn = compute_energy_density(stft)

    # sound intensity is w*[x,y,z]
    p_kn = compute_sound_pressure(stft)
    u_kn = compute_particle_velocity(stft)

    groundtruth = ((p0 / 4.) * pow(np.linalg.norm(u_kn.data,axis=0), 2)) \
                  + ((1. / (4 * p0 * pow(c, 2))) * pow(np.abs(p_kn.data), 2))
    assert np.allclose(groundtruth, e_kn.data)


def test_compute_diffuseness():

    # incorrect input types
    pytest.raises(TypeError, compute_diffuseness, 12345)
    pytest.raises(TypeError, compute_diffuseness, np.zeros([1, 2]))
    pytest.raises(TypeError, compute_diffuseness, [[1, 2], [3, 4]])

    # Valid signal
    audio_data = np.ones([4, 10])
    sr = 48000
    signal = Signal(audio_data, sr)
    f_n = compute_diffuseness(signal)

    # Compute against equation 13 (Politis and Pulkki)
    num = 2 * np.linalg.norm(signal.data[0] * signal.data[1:],axis=0)
    den = np.power(np.linalg.norm(signal.data[1:],axis=0), 2) + np.power(np.abs(signal.data[0]), 2)
    groundtruth = 1 - (num / den)
    assert np.allclose(f_n.data,groundtruth)

    # Valid stft
    audio_data = np.ones([4, 1000])
    sr = 48000
    stft = Stft(Signal(audio_data, sr))
    f_kn = compute_diffuseness(stft)

    # Compute against equation 13 (Politis and Pulkki)
    num = 2 * np.linalg.norm(np.real(np.conjugate(stft.data[0]) * stft.data[1:]),axis=0)
    den = np.power(np.linalg.norm(stft.data[1:],axis=0), 2) + np.power(np.abs(stft.data[0]), 2)
    groundtruth = 1 - (num / den)
    assert np.allclose(f_kn.data, groundtruth)