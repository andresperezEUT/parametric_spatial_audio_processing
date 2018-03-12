##############################
#
#   core.py
#
#   Created by Andres Perez Lopez
#   25/01/2018
#
#   Main parametric spatial audio processing methods
#
#   We will consider Ambisonics in SN3D normalization
#   (1st order channels normalized to 1)
#   and FuMa ordering.
#   See for example [Politis,2016] or [Pulkki,2018] Chapter 5.
#
#   References:
#   [Politis,2016] "ACOUSTIC INTENSITY, ENERGY-DENSITY AND
#       DIFFUSENESS ESTIMATION IN A DIRECTIONALLY-CONSTRAINED
#       REGION (Politis and Pulkki, 2016)
#   [Pulkki,2018] " PARAMETRIC TIME-FREQUENCY DOMAIN SPATIAL AUDIO
#       (Pulkki et al, 2018)
#
##############################

'''
TODO
find transposition for 3D matrices for implementing norm form
'''

import numpy as np
import scipy.signal as sg
from parametric_spatial_audio_processing.util import convert_bformat_acn_2_fuma, convert_bformat_sn3d_2_fuma, \
    cartesian_2_spherical, convert_bformat_fuma_2_sn3d, convert_bformat_n3d_2_sn3d, herm

# Values taken from https://en.wikipedia.org/wiki/Acoustic_impedance
# at 25 celsius degrees
c = 346.13  # m/s
p0 = 1.1839 # kg/m3
s = 1  # default microphone sensitivity

# If vsa is very small, and we have a lot of zeros in the signal,
# then it's very possible that pow(norm(x|y|z),2) will yield 0.0
# (for example in the energy density, or in the doa angle computation ),
# which will cause NaN warnings when divided by zero
vsa = 1e-50 # very small amount



class Signal:
    '''
    Custom class for holding an audio signal
    in form of multichannel time-series.
    Internally it keeps the audio in sn3d, FuMa ordering
    For the moment only BFormat (4 channel) compatible

    Parameters
    _____________
    :data: 2D numpy array with shape [channels,samples]
    :sample_rate: audio sample rate as int
    :ordering: optional str for specifying channel order of :data:
        allowed values: ['fuma','acn']
        default to FuMa
    ordering: optional str for specifying normalization of :data:
        allowed values: ['sn3d','n3d','fuma','maxn']
        default to sn3d
    '''

    allowed_ordering_strings = ['fuma','acn']
    allowed_norm_strings = ['sn3d','n3d','fuma','maxn']

    def __init__(self, data, sample_rate, ordering='fuma', norm='sn3d'):

        # Validate data types
        if not isinstance(data,np.ndarray):
            raise TypeError

        if not isinstance(sample_rate,int):
            raise TypeError

        if not isinstance(ordering,str):
            raise TypeError
        elif ordering not in self.allowed_ordering_strings:
            raise TypeError('Allowed ordering strings: ' + self.allowed_ordering_strings)

        if not isinstance(norm,str):
            raise TypeError
        elif norm not in self.allowed_norm_strings:
            raise TypeError('Allowed norm strings: ' + self.allowed_norm_strings)

        self.sample_rate = sample_rate

        self.data = data

        # Instanciation, formatting
        if ordering is 'acn':
            self.data = convert_bformat_acn_2_fuma(self.data)

        if norm is 'n3d':
            self.data = convert_bformat_n3d_2_sn3d(self.data)
        elif norm is 'fuma':
            self.data = convert_bformat_fuma_2_sn3d(self.data)

        # Add a small noise floor to avoid 0s (only on sinthetic)
        noise = np.random.rand(self.data.shape[0],self.data.shape[1])*2-1
        self.data += noise*vsa

    def get_num_channels(self):
        return np.shape(self.data)[0]

    def get_num_frames(self):
        return np.shape(self.data)[1]


class Stft:

    @classmethod
    def fromSignal(self,signal,window_size=256 ,window_overlap=128):

        # Validate data types
        if not isinstance(signal, Signal):
            raise TypeError

        # Instanciate
        f, t, Zxx = sg.stft(signal.data,
                            fs=signal.sample_rate,
                            nperseg=window_size,
                            noverlap=window_overlap)
        # self.t = t
        # self.f = f
        # self.data = Zxx
        # self.sample_rate = signal.sample_rate

        # # Add a small amout to the samples to avoid 0s (only on sinthetic)
        # for ch in range(self.get_num_channels()):
        #     for f in range(self.get_num_frequency_bins()):
        #         for n in range(self.get_num_time_bins()):
        #             if np.abs(self.data[ch, f, n]) == 0:
        #                 self.data[ch, f, n] = complex(vsa, vsa)

        return Stft(t,f,Zxx,signal.sample_rate)


    def __init__(self, *args):

        # Input is Signal
        if isinstance(args[0],Signal):
            signal = args[0]
            window_size = 256
            window_overlap = 128

            # Validate data types
            if not isinstance(signal, Signal):
                raise TypeError

            # Instanciate
            f, t, Zxx =  sg.stft(signal.data,
                                 fs=signal.sample_rate,
                                 nperseg=window_size,
                                 noverlap=window_overlap)
            self.t = t
            self.f = f
            self.data = Zxx
            self.sample_rate = signal.sample_rate

            # Add a small amout to the samples to avoid 0s (only on sinthetic)
            for ch in range(self.get_num_channels()):
                for f in range(self.get_num_frequency_bins()):
                    for n in range(self.get_num_time_bins()):
                        if np.abs(self.data[ch,f,n]) == 0:
                            self.data[ch, f, n] = complex(vsa,vsa)

        # Input is Stft attributes

        # TODO
        # elif type(args[0]) == Stft:

        elif len(args)==4:
            t = args[0]
            f = args[1]
            data = args[2]
            sample_rate = args[3]

            # Validata data types and shapes
            if not isinstance(t,np.ndarray) or np.ndim(t) != 1:
                raise TypeError

            if not isinstance(f, np.ndarray) or np.ndim(f) != 1:
                raise TypeError

            if not isinstance(data, np.ndarray) or np.ndim(data) != 3:
                raise TypeError

            if not isinstance(sample_rate, int):
                raise TypeError

            # Validate data shape consistency:
            # t     [ch, num_t]
            # f     [ch, num_f]
            # data  [ch, num_f, num_t]
            num_t = np.size(t)
            assert num_t == np.shape(data)[-1]
            num_f = np.size(f)
            assert num_f == np.shape(data)[-2]

            # Instanciate
            self.t = t
            self.f = f
            self.data = data
            self.sample_rate = sample_rate

        # Other number of args
        else:
            raise TypeError('Incorrect number of arguments')


    def get_num_channels(self):
        return np.shape(self.data)[0]

    def get_num_frequency_bins(self):
        return np.shape(self.data)[1]

    def get_num_time_bins(self):
        return np.shape(self.data)[2]


def compute_sound_pressure(arg):
    '''
    Compute the sound pressure scalar value p(n)
    from a BFormat signal

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    1 channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_sound_pressure_t(arg)
    elif isinstance(arg,Stft):
        return compute_sound_pressure_tf(arg)
    else:
        raise TypeError

def compute_sound_pressure_t(signal):
    scale = 1.0 / s
    data = scale * signal.data[[0], :]
    return Signal(data, signal.sample_rate)

def compute_sound_pressure_tf(stft):
    scale = 1.0 / s
    data = scale * stft.data[[0],:,:]
    return Stft(stft.t, stft.f, data, stft.sample_rate)

def compute_sound_pressure_norm(arg):
    '''
    Compute the normalized sound pressure scalar value p(n)
    from a BFormat signal

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    1 channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_sound_pressure_norm_t(arg)
    elif isinstance(arg,Stft):
        return compute_sound_pressure_norm_tf(arg)
    else:
        raise TypeError

def compute_sound_pressure_norm_t(signal):
    return Signal(signal.data[[0],:], signal.sample_rate)

def compute_sound_pressure_norm_tf(stft):
    return Stft(stft.t, stft.f, stft.data[[0],:], stft.sample_rate)


def compute_particle_velocity(arg):
    '''
    Compute the particle velocity vector u(n) from a
    BFormat signal assuming an input plane wave

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    3-channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_particle_velocity_t(arg)
    elif isinstance(arg,Stft):
        return compute_particle_velocity_tf(arg)
    else:
        raise TypeError

def compute_particle_velocity_t(signal):
    scale = -1.0 / (s * p0 * c)
    data = scale * signal.data[1:,:]
    return Signal(data, signal.sample_rate)


def compute_particle_velocity_tf(stft):
    scale = -1.0 / (s * p0 * c)
    data = scale * stft.data[1:,:,:]
    return Stft(stft.t, stft.f, data, stft.sample_rate)

def compute_particle_velocity_norm(arg):
    '''
    Compute the normalized particle velocity vector u(n) from a
    BFormat signal assuming an input plane wave

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    3-channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_particle_velocity_norm_t(arg)
    elif isinstance(arg,Stft):
        return compute_particle_velocity_norm_tf(arg)
    else:
        raise TypeError

def compute_particle_velocity_norm_t(signal):
    return Signal(signal.data[1:], signal.sample_rate)

def compute_particle_velocity_norm_tf(stft):
    return Stft(stft.t, stft.f, stft.data[1:], stft.sample_rate)

def compute_intensity_vector(arg):
    '''
    Compute the active intensity vector i(n)
    from a BFormat signal

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    3-channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_intensity_vector_t(arg)
    elif isinstance(arg,Stft):
        return compute_intensity_vector_tf(arg)
    else:
        raise TypeError

def compute_intensity_vector_t(signal):
    p_n = compute_sound_pressure(signal)
    u_n = compute_particle_velocity(signal)

    # Multiply each u_n array by the p_n array
    data = 0.5 * p_n.data * u_n.data
    return Signal(data,signal.sample_rate)

def compute_intensity_vector_tf(stft):
    p_kn = compute_sound_pressure(stft)
    u_kn = compute_particle_velocity(stft)

    # Multiply each u_kn matrix by the p_kn_c matrix
    data = 0.5 * np.real(u_kn.data * np.conjugate(p_kn.data))
    return Stft(stft.t,stft.f,data,stft.sample_rate)

def compute_intensity_vector_norm(arg):
    '''
    Compute the active normalized intensity vector i(n)
    from a BFormat signal

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    3-channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_intensity_vector_norm_t(arg)
    elif isinstance(arg,Stft):
        return compute_intensity_vector_norm_tf(arg)
    else:
        raise TypeError

def compute_intensity_vector_norm_t(signal):
    p = compute_sound_pressure_norm_t(signal).data
    v = compute_particle_velocity_norm_t(signal).data
    vH = herm(v)
    data = -1 * np.real(p*vH.T)
    return Signal(data,signal.sample_rate)

def compute_intensity_vector_norm_tf(stft):
    p = compute_sound_pressure_norm_tf(stft).data
    v = compute_particle_velocity_norm_tf(stft).data
    vH = herm(v)
    data = -1 * np.real(p*vH)
    return Stft(stft.t,stft.f,data,stft.sample_rate)

def compute_DOA(arg):
    '''
    Compute the Direction of Arrival angles
    from a BFormat signal

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    3-channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_DOA_t(arg)
        # return compute_DOA_norm_t(arg)
    elif isinstance(arg,Stft):
        return compute_DOA_tf(arg)
        # return compute_DOA_norm_tf(arg)
    else:
        raise TypeError

def compute_DOA_t(signal):
    i_n = compute_intensity_vector(signal)
    angle = cartesian_2_spherical(-1*i_n.data)
    return Signal(angle[:-1],signal.sample_rate)

def compute_DOA_tf(stft):
    i_kn = compute_intensity_vector(stft)
    angle = cartesian_2_spherical(-1 * i_kn.data)
    return Stft(stft.t,stft.f,angle[:-1],stft.sample_rate)

def compute_DOA_norm_t(signal):
    i = compute_intensity_vector_norm_t(signal).data
    angle = cartesian_2_spherical( -1 * ( i / np.linalg.norm(i,axis=0)))
    return Signal(angle[:-1],signal.sample_rate)

def compute_DOA_norm_tf(stft):
    i = compute_intensity_vector_norm_tf(stft).data
    angle = cartesian_2_spherical(-1 * (i / np.linalg.norm(i, axis=0)))
    return Stft(stft.t,stft.f,angle[:-1],stft.sample_rate)

def compute_energy_density(arg):
    '''
    Compute the energy density vector e(n)
    from a BFormat signal

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    3-channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_energy_density_t(arg)
    elif isinstance(arg,Stft):
        return compute_energy_density_tf(arg)
    else:
        raise TypeError

def compute_energy_density_t(signal):
    p_n = compute_sound_pressure(signal)
    u_n = compute_particle_velocity(signal)

    s1 = np.power(np.linalg.norm(u_n.data,axis=0), 2)
    s2 = np.power(abs(p_n.data), 2)

    data = ((p0/4.)*s1) + ((1./(4*p0*np.power(c,2)))*s2)
    return Signal(data,signal.sample_rate)

def compute_energy_density_tf(stft):
    p_kn = compute_sound_pressure(stft)
    u_kn = compute_particle_velocity(stft)

    s1 = np.power(np.linalg.norm(u_kn.data,axis=0), 2)
    s2 = np.power(abs(p_kn.data), 2)

    data = ((p0/4.)*s1) + ((1./(4*p0*np.power(c,2)))*s2)
    return Stft(stft.t,stft.f,data,stft.sample_rate)


def compute_energy_density_norm(arg):
    '''
    Compute the normalized energy density vector e(n)
    from a BFormat signal

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    3-channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_energy_density_norm_t(arg)
    elif isinstance(arg,Stft):
        return compute_energy_density_norm_tf(arg)
    else:
        raise TypeError

def compute_energy_density_norm_t(signal):
    p = compute_sound_pressure_norm_t(signal).data
    v = compute_particle_velocity_norm_t(signal).data
    vH = herm(v)
    data = 0.5 * ( (np.power(np.abs(p),2)) + (v*vH) )
    return Signal(data,signal.sample_rate)

def compute_energy_density_norm_tf(stft):
    p = compute_sound_pressure_norm_tf(stft).data
    v = compute_particle_velocity_norm_tf(stft).data
    vH = herm(v)
    data = 0.5 * ( (np.power(np.abs(p),2)) + (v*vH) )
    return Stft(stft.t,stft.f,data,stft.sample_rate)

def compute_diffuseness(arg):
    '''
    Compute the diffuseness f(n)
    from a BFormat signal

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    3-channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_diffuseness_t(arg)
        # return compute_diffuseness_norm_t(arg)
    elif isinstance(arg,Stft):
        return compute_diffuseness_tf(arg)
        # return compute_diffuseness_norm_tf(arg)
    else:
        raise TypeError

def compute_diffuseness_t(signal):
    i_n = compute_intensity_vector(signal)
    e_n = compute_energy_density(signal)

    data = 1 - (np.linalg.norm(i_n.data,axis=0)/(c*e_n.data))
    return Signal(data,signal.sample_rate)

def compute_diffuseness_tf(stft):
    i_kn = compute_intensity_vector(stft)
    e_kn = compute_energy_density(stft)

    data = 1 - (np.linalg.norm(i_kn.data,axis=0) / (c * e_kn.data))
    return Stft(stft.t,stft.f,data,stft.sample_rate)

def compute_directivity(arg):
    '''
    Compute the directivity delta(n) -- inverse of the diffuseness
    from a BFormat signal

    Parameters
    :arg: Instance of Signal or STFT

    Returns:
    3-channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_directivity_t(arg)
        # return compute_diffuseness_norm_t(arg)
    elif isinstance(arg,Stft):
        return compute_directivity_tf(arg)
        # return compute_diffuseness_norm_tf(arg)
    else:
        raise TypeError

def compute_directivity_t(signal):
    i_n = compute_intensity_vector(signal)
    e_n = compute_energy_density(signal)

    data = np.linalg.norm(i_n.data,axis=0)/(c*e_n.data)
    return Signal(data,signal.sample_rate)

def compute_directivity_tf(stft):
    i_kn = compute_intensity_vector(stft)
    e_kn = compute_energy_density(stft)

    data = np.linalg.norm(i_kn.data,axis=0) / (c * e_kn.data)
    return Stft(stft.t,stft.f,data,stft.sample_rate)


def compute_diffuseness_norm_t(signal):
    i = compute_intensity_vector_norm_t(signal).data
    e = compute_energy_density_norm_t(signal).data
    data = 1 - ( np.linalg.norm(i,axis=0) / e )
    return Signal(data,signal.sample_rate)

def compute_diffuseness_norm_tf(stft):
    i = compute_intensity_vector_norm_tf(stft).data
    e = compute_energy_density_norm_tf(stft).data
    data = 1 - ( np.linalg.norm(i,axis=0) / e )
    return Stft(stft.t,stft.f,data,stft.sample_rate)

