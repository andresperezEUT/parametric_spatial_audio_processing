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
from numbers import Number
import copy
from warnings import warn
# import pyaudio
import scipy.stats
from scipy import ndimage as ndi

import pyaudio

import numpy as np
import scipy.signal as sg
from parametric_spatial_audio_processing.util import convert_bformat_acn_2_fuma, convert_bformat_sn3d_2_fuma, \
    cartesian_2_spherical, convert_bformat_fuma_2_sn3d, convert_bformat_n3d_2_sn3d, herm, compute_signal_envelope, \
    moving_average

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

        # Expand dims in the mono case
        if np.ndim(data)==1:
            self.data = np.expand_dims(data,0)
        else:
            self.data = data

        # Instanciation, formatting
        if ordering == 'acn':
            self.data = convert_bformat_acn_2_fuma(self.data)

        if norm == 'n3d':
            self.data = convert_bformat_n3d_2_sn3d(self.data)
        elif norm == 'fuma':
            self.data = convert_bformat_fuma_2_sn3d(self.data)

        # Add a small noise floor to avoid 0s (only on sinthetic)
        # noise = np.random.rand(self.data.shape[0],self.data.shape[1])*2-1
        # self.data += noise*vsa

    def __getitem__(self, key):
        return Stft(np.expand_dims(self.data[key], 0), self.sample_rate)

    def get_num_channels(self):
        return np.shape(self.data)[0]

    def get_num_frames(self):
        return np.shape(self.data)[1]

    def limit_envelope(self,th=0.05):
        """
        Returns a new signal with only envelope values bigger than th

        :param fmin:
        :param fmax:
        :return:
        """

        envelope_signal = compute_signal_envelope(self)


        limited_data_list = []
        # limited_data = np.empty(np.shape(self.data))
        # limited_data[:] = np.nan


        for i, v in enumerate(envelope_signal.data[0]):
            if v >= th:
                # limited_data[:,i] = self.data[:,i]
                limited_data_list.append(self.data[:,i].tolist())

        return Signal(np.asarray(limited_data_list).T, self.sample_rate)

    def smooth(self,window_size=8):
        '''
        Computes moving average of the signal.
        Preserves the original length by means of rignt zeropadding

        :param window_size: Number of samples of the averaging window
        :return: A new signal instance
        '''

        averaged_signal = copy.deepcopy(self)
        for ch in range(self.get_num_channels()):
            averaged_signal.data[ch,:-(window_size-1)] = moving_average(averaged_signal.data[ch,:],window_size)
            averaged_signal.data[ch, self.get_num_frames()-window_size:] = np.zeros(window_size)
        return averaged_signal


    def apply_softmask(self, mask, th):

        # TODO error checks

        # Create a new signal with the same self.fata dimensions
        masked_data = np.ndarray(np.shape(self.data))

        # Evaluate each sample with mask, and copy the value if condition is True,
        # and place a numpy NaN if False
        for n in range(self.get_num_frames()):
            # masks are usually single-channel
            if mask.data[:, n] >= th:
                # print(self.data[m, k, n])
                masked_data[:, n] = self.data[:, n]
            else:
                masked_data[:, n] = np.nan

        return Signal(masked_data, self.sample_rate)

    @classmethod
    def fromStft(cls, stft, window_size=256, window_overlap=128, nfft=256, window='hann'):
        """
        TODO
        basically compute IFFT
        :param signal:
        :param window_size:
        :param window_overlap:
        :param nfft:
        :return:
        """
        # Validate data types
        if not isinstance(stft, Stft):
            raise TypeError

        # Check COLA
        if window_overlap is None:
            window_overlap = window_size//2
        cola = sg.check_COLA(window=window,
                             nperseg=window_size,
                             noverlap=window_overlap)
        if not cola:
            warn('Current STFT parameters do not meet the COLA criterium!')

        # Instanciate
        t, x = sg.istft(stft.data,
                        fs=stft.sample_rate,
                        window=window,
                        nperseg=window_size,
                        noverlap=window_overlap,
                        nfft=nfft)

        # With overlap we will have some extra samples, so remove them
        x = x[:,:-(window_size-window_overlap)//2]
        # Stft internal representation is in fuma ordering
        return Signal(x, stft.sample_rate, ordering='fuma', norm='sn3d')


    def play(self,channel=None,start_time=None,end_time=None):
        '''
        PLAY SOUND
        TODO
        :return:
        '''

        first_frame = 0
        last_frame = np.shape(self.data)[1]

        start_frame = first_frame if start_time is None else start_time*self.sample_rate
        end_frame = last_frame if end_time is None else end_time*self.sample_rate

        if start_frame >= end_frame:
            warn('Start time cannot be greater than end time')
            start_frame = 0

        if start_frame < 0:
            warn('Start time must be at least 0')
            start_frame = 0

        if start_frame > last_frame:
            warn('Start time cannot be greater than duration')
            start_frame = 0

        if end_frame < 0:
            warn('End time must be at least 0')
            end_frame = last_frame

        if end_frame > last_frame:
            warn('End time cannot be greater than duration')
            end_frame = last_frame

        if channel is None:
            data = self.data[:, start_frame:end_frame]
        else:
            data = self.data[channel, start_frame:end_frame]


        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.sample_rate,
                        output=True)

        stream.write(data.astype(np.float32).tobytes())
        stream.stop_stream()
        stream.close()
        p.terminate()


class Stft:

    @classmethod
    def fromSignal(cls, signal, window_size=256, window_overlap=128, nfft=256, window='hann'):

        # Validate data types
        if not isinstance(signal, Signal):
            raise TypeError

        # Check COLA
        if window_overlap is None:
            window_overlap = window_size//2
        cola = sg.check_COLA(window=window,
                             nperseg=window_size,
                             noverlap=window_overlap)
        if not cola:
            warn('Current STFT parameters do not meet the COLA criterium!')

        # Instanciate
        f, t, Zxx = sg.stft(signal.data,
                            fs=signal.sample_rate,
                            window=window,
                            nperseg=window_size,
                            noverlap=window_overlap,
                            nfft=nfft)

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
            self.sample_rate = signal.sample_rate

            # if the signal is mono, expand dims, so we can treat the default multichannel
            if np.ndim(Zxx) == 2:
                self.data = np.expand_dims(Zxx,0)
            else:
                self.data = Zxx


            # # Add a small amout to the samples to avoid 0s (only on sinthetic)
            # for ch in range(self.get_num_channels()):
            #     for f in range(self.get_num_frequency_bins()):
            #         for n in range(self.get_num_time_bins()):
            #             if np.abs(self.data[ch,f,n]) == 0:
            #                 self.data[ch, f, n] = complex(vsa,vsa)

        # Input is Stft attributes

        # TODO
        # elif type(args[0]) == Stft:

        elif len(args)==4:
            t = args[0]
            f = args[1]
            data = args[2]
            sample_rate = args[3]

            # Valid data types and shapes
            if not isinstance(t,np.ndarray) or np.ndim(t) != 1:
                raise TypeError

            if not isinstance(f, np.ndarray) or np.ndim(f) != 1:
                raise TypeError

            if not isinstance(data, np.ndarray):
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
            self.sample_rate = sample_rate
            # Expand dims in case of mono
            if np.ndim(data) == 2:
                self.data = np.expand_dims(data,0)
            else:
                self.data = data

        # Other number of args
        else:
            raise TypeError('Incorrect number of arguments')

    def __getitem__(self, key):
        return Stft(self.t,self.f,np.expand_dims(self.data[key],0),self.sample_rate)

    def __setitem__(self, key, value):
            self.data[key] = value
        # return Stft(self.t,self.f,np.expand_dims(self.data[key],0),self.sample_rate)

    def __add__(self,arg):
        if isinstance(arg,Stft):
            return Stft(self.t, self.f, self.data + arg.data, self.sample_rate)
        else:
            return Stft(self.t, self.f, self.data + arg, self.sample_rate)

    def __sub__(self,arg):
        if isinstance(arg,Stft):
            return Stft(self.t, self.f, self.data - arg.data, self.sample_rate)
        else:
            return Stft(self.t, self.f, self.data - arg, self.sample_rate)

    def __mul__(self,arg):
        if isinstance(arg,Stft):
            return Stft(self.t, self.f, self.data * arg.data, self.sample_rate)
        else:
            return Stft(self.t, self.f, self.data * arg, self.sample_rate)

    def sqrt(self):
        return Stft(self.t, self.f, np.sqrt(self.data), self.sample_rate)



    def get_num_channels(self):
        return np.shape(self.data)[0]

    def get_num_frequency_bins(self):
        return np.shape(self.data)[1]

    def get_num_time_bins(self):
        return np.shape(self.data)[2]


    def limit_bands(self,fmin=125,fmax=8000):
        """
        Returns a new stft which is bandlimited between fmin and fmax

        :param fmin:
        :param fmax:
        :return:
        """

        if fmin >= fmax:
            raise ValueError('fmin cannot be equal or bigger than fmax')

        bandlimited_stft = copy.deepcopy(self)

        count_deleted = 0
        for i, k in enumerate(self.f):
            if k < fmin or k > fmax:
                bandlimited_stft.f = np.delete(bandlimited_stft.f, [i - count_deleted], axis=0)
                bandlimited_stft.data = np.delete(bandlimited_stft.data, [i - count_deleted], axis=1)
                count_deleted = count_deleted + 1

        return bandlimited_stft



    def apply_binary_mask(self,mask):
        '''
        Returns a new stft containing only the values of self with evaluate
        positive to the provided binary mask.

        :param mask: An stft with the same dimensions

        :return: A new stft instance containing the values of self when True,
            and np.nan when False
        '''

        # Both stft and mask must have same dimensions
        if not (self.t == mask.t).all():
            raise ValueError("Dimension not matching: t")

        if not (self.f == mask.f).all():
            raise ValueError("Dimension not matching: f")

        # Num channels might be different... but mask must be single channel!
        if not (mask.get_num_channels() == 1):
            raise ValueError("Mask must be single channel!")

        if self.sample_rate != mask.sample_rate:
            raise ValueError("Dimension not matching: sr")

        # Create a new matrix with the same self.fata dimensions
        masked_data = np.ndarray(np.shape(self.data))

        # Evaluate each TF bin with mask, and copy the value if condition is True,
        # and place a None if False
        for k in range(self.get_num_frequency_bins()):
            for n in range(self.get_num_time_bins()):

                if mask.data[:,k,n]:
                    masked_data[:,k,n] = self.data[:,k,n]
                else:
                    masked_data[:, k, n] = np.nan
                    # print(masked_data[:,k,n])

        return Stft(self.t,self.f,masked_data,self.sample_rate)



    def apply_softmask(self, mask, th):
        '''
        Returns a new stft containing only the TF values of self with evaluate
        positive to be equal or greater than the provided softmask .

        :param mask: An stft with the same dimensions
        :param th: Float threshold between 0 and 1

        :return: A new stft instance containing the values of self when True,
            and np.nan when False
        '''

        # Both stft and mask must have same dimensions
        if not (self.t == mask.t).all():
            raise ValueError("Dimension not matching: t")

        if not (self.f == mask.f).all():
            raise ValueError("Dimension not matching: f")

        # Num channels might be different... but mask must be single channel!
        if not (mask.get_num_channels() == 1):
            raise ValueError("Mask must be single channel!")

        if self.sample_rate != mask.sample_rate:
            raise ValueError("Dimension not matching: sr")

        # Create a new matrix with the same self.fata dimensions
        masked_data = np.ndarray(np.shape(self.data))

        # Evaluate each TF bin with mask, and copy the value if condition is True,
        # and place a numpy NaN if False
        for k in range(self.get_num_frequency_bins()):
            for n in range(self.get_num_time_bins()):

                # masks are usually single-channel
                if mask.data[:, k, n] >= th:
                    # print(self.data[m, k, n])
                    masked_data[:, k, n] = self.data[:, k, n]
                else:
                    masked_data[:, k, n] = np.nan

        return Stft(self.t, self.f, masked_data, self.sample_rate)


    def compute_mask(self,th,th_type='absolute'):
        """
        TODO
        :param th:
        :return:
        """
        #TODO ERRORCHECK

        # Create a new matrix with the same self.fata dimensions
        masked_data = np.ndarray(np.shape(self.data))

        if th_type == 'absolute':
            pass

        if th_type == 'percentile':
            # Relative th on the percentile of the data itself
            th = np.percentile(self.data,th)
            pass

        # Evaluate TFbin>=th, and return 1 if True, np.nan if False
        for k in range(self.get_num_frequency_bins()):
            for n in range(self.get_num_time_bins()):

                # masks are usually single-channel
                if self.data[:, k, n] >= th:
                    # print(self.data[m, k, n])
                    masked_data[:, k, n] = 1
                else:
                    masked_data[:, k, n] = np.nan


        return Stft(self.t, self.f, masked_data, self.sample_rate)


    def apply_mask(self,mask):
        """
        TODO
        :param mask:
        :return:
        """
        # TODO ERRORCHECK

        # mask and self must have same same number of channels, or mask should be mono

        # Both stft and mask must have same dimensions
        if not (self.t == mask.t).all():
            raise ValueError("Dimension not matching: t")

        if not (self.f == mask.f).all():
            raise ValueError("Dimension not matching: f")

        # Num channels might be different only if mask is mono!
        if not (mask.get_num_channels() == 1):
            if not (mask.get_num_channels() == self.get_num_channels()):
                raise ValueError("Dimension not matching: m")

        if self.sample_rate != mask.sample_rate:
            raise ValueError("Dimension not matching: sr")

        # Create a new matrix with the same self.data dimensions
        masked_data = self.get_copy()
        # masked_data = np.ndarray(np.shape(self.data))

        if mask.get_num_channels() == 1:
            # Mono mask: apply to all channels
            for k in self.range('k'):
                for n in self.range('n'):
                    masked_data.data[:,k,n] = self.data[:,k,n]*mask.data[:,k,n]
        else:
            # Multichannel mask
            for ch in self.range('ch'):
                for k in self.range('k'):
                    for n in self.range('n'):
                        masked_data.data[ch,k,n] = self.data[ch,k,n]*mask.data[ch,k,n]

        return masked_data

    def get_complementary_mask(self):
        """
        TODO
        :param mask:
        :return:
        """
        # TODO ERRORCHECK


        # Create a new matrix with the same self.data dimensions
        masked_data = self.get_copy()
        # masked_data = np.ndarray(np.shape(self.data))

        for ch in self.range('ch'):
            for k in self.range('k'):
                for n in self.range('n'):
                    masked_data.data[ch,k,n] = 1 - self.data[ch,k,n]

        return masked_data


    def get_copy(self):
        """
        TODO
        """
        return copy.deepcopy(self)

    def range(self,dim):
        '''
        TODO
        :param dim:
        :return:
        '''
        if dim == 'ch':
            return range(self.get_num_channels())
        elif dim == 'k':
            return range(self.get_num_frequency_bins())
        elif dim == 'n':
            return range(self.get_num_time_bins())
        else:
            raise TypeError('unknown dim identifyer')

    def get_magnitude_stft(self):

        # mag_stft = self.get_copy()
        # mag_stft.data = mag_stft.data.astype('float64')


        mag_stft = np.zeros(np.shape(self.data),dtype='float64')

        for ch in self.range('ch'):
            for k in self.range('k'):
                for n in self.range('n'):
                    mag_stft[ch,k,n] = np.abs(self.data[ch,k,n])

        return Stft(self.t, self.f, mag_stft, self.sample_rate)

    def get_phase_stft(self):

        # phase_stft = self.get_copy()
        # phase_stft.data = phase_stft.data.astype('float64')

        phase_stft = np.zeros(np.shape(self.data), dtype='float64')

        for ch in self.range('ch'):
            for k in self.range('k'):
                for n in self.range('n'):
                    phase_stft[ch,k,n] = np.angle(self.data[ch,k,n])

        return Stft(self.t, self.f, phase_stft, self.sample_rate)


    def compute_doa_statistics(self):
        '''
        DISCARD FIRST NANS

        :return:
        '''
        azi = []
        ele = []
        for k in self.range('k'):
            for n in self.range('n'):
                if not np.isnan(self.data[0, k, n]):
                    azi.append(self.data[0, k, n])
                    ele.append(self.data[1, k, n])

        mean_azi = scipy.stats.circmean(azi, high=2 * np.pi, low=0)
        std_azi = scipy.stats.circstd(azi, high=2 * np.pi, low=0)
        mean_ele = np.mean(ele)
        std_ele = np.std(ele)

        return (mean_azi,mean_ele,std_azi,std_ele)



    def compute_threshold_otsu(self,nbins=256):
        """Return threshold value based on Otsu's method.
        Parameters
        ----------
        image : (N, M) ndarray
            Grayscale input image.
        nbins : int, optional
            Number of bins used to calculate histogram. This value is ignored for
            integer arrays.
        Returns
        -------
        threshold : float
            Upper threshold value. All pixels with an intensity higher than
            this value are assumed to be foreground.
        Raises
        ------
        ValueError
             If `image` only contains a single grayscale value.
        References
        ----------
        .. [1] Wikipedia, https://en.wikipedia.org/wiki/Otsu's_Method
        Examples
        --------
        >>> from skimage.data import camera
        >>> image = camera()
        >>> thresh = threshold_otsu(image)
        >>> binary = image <= thresh
        Notes
        -----
        The input image must be grayscale.
        """
        if len(image.shape) > 2 and image.shape[-1] in (3, 4):
            msg = "threshold_otsu is expected to work correctly only for " \
                  "grayscale images; image shape {0} looks like an RGB image"
            warn(msg.format(image.shape))

        # Check if the image is multi-colored or not
        if image.min() == image.max():
            raise ValueError("threshold_otsu is expected to work with images "
                             "having more than one color. The input image seems "
                             "to have just one color {0}.".format(image.min()))

        hist, bin_centers = histogram(image.ravel(), nbins)
        hist = hist.astype(float)

        # class probabilities for all possible thresholds
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        # class means for all possible thresholds
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

        # Clip ends to align class 1 and class 2 variables:
        # The last value of `weight1`/`mean1` should pair with zero values in
        # `weight2`/`mean2`, which do not exist.
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        threshold = bin_centers[:-1][idx]
        return threshold


    def compute_threshold_local(self, block_size, method='gaussian', offset=0,
                        mode='reflect', param=None, cval=0):
        """Compute a threshold mask image based on local pixel neighborhood.
        Also known as adaptive or dynamic thresholding. The threshold value is
        the weighted mean for the local neighborhood of a pixel subtracted by a
        constant. Alternatively the threshold can be determined dynamically by a
        given function, using the 'generic' method.
        Parameters
        ----------
        image : (N, M) ndarray
            Input image.
        block_size : int
            Odd size of pixel neighborhood which is used to calculate the
            threshold value (e.g. 3, 5, 7, ..., 21, ...).
        method : {'generic', 'gaussian', 'mean', 'median'}, optional
            Method used to determine adaptive threshold for local neighbourhood in
            weighted mean image.
            * 'generic': use custom function (see `param` parameter)
            * 'gaussian': apply gaussian filter (see `param` parameter for custom\
                          sigma value)
            * 'mean': apply arithmetic mean filter
            * 'median': apply median rank filter
            By default the 'gaussian' method is used.
        offset : float, optional
            Constant subtracted from weighted mean of neighborhood to calculate
            the local threshold value. Default offset is 0.
        mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
            The mode parameter determines how the array borders are handled, where
            cval is the value when mode is equal to 'constant'.
            Default is 'reflect'.
        param : {int, function}, optional
            Either specify sigma for 'gaussian' method or function object for
            'generic' method. This functions takes the flat array of local
            neighbourhood as a single argument and returns the calculated
            threshold for the centre pixel.
        cval : float, optional
            Value to fill past edges of input if mode is 'constant'.
        Returns
        -------
        threshold : (N, M) ndarray
            Threshold image. All pixels in the input image higher than the
            corresponding pixel in the threshold image are considered foreground.
        References
        ----------
        .. [1] https://docs.opencv.org/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#adaptivethreshold
        Examples
        --------
        """
        if block_size % 2 == 0:
            raise ValueError("The kwarg ``block_size`` must be odd! Given "
                             "``block_size`` {0} is even.".format(block_size))
        # assert_nD(image, 2)
        thresh_image = np.zeros(self.data.shape, 'double')
        if method == 'generic':
            ndi.generic_filter(self.data, param, block_size,
                               output=thresh_image, mode=mode, cval=cval)
        elif method == 'gaussian':
            if param is None:
                # automatically determine sigma which covers > 99% of distribution
                sigma = (block_size - 1) / 6.0
            else:
                sigma = param
            ndi.gaussian_filter(v, sigma, output=thresh_image, mode=mode,
                                cval=cval)
        elif method == 'mean':
            mask = 1. / block_size * np.ones((block_size,))
            # separation of filters to speedup convolution
            ndi.convolve1d(self.data, mask, axis=0, output=thresh_image, mode=mode,
                           cval=cval)
            ndi.convolve1d(thresh_image, mask, axis=1, output=thresh_image,
                           mode=mode, cval=cval)
        elif method == 'median':
            ndi.median_filter(self.data, block_size, output=thresh_image, mode=mode,
                              cval=cval)
        else:
            raise ValueError("Invalid method specified. Please use `generic`, "
                             "`gaussian`, `mean`, or `median`.")

        return thresh_image - offset


    def compute_msc(self,dt=10):
        """
        Magnitud square coherence among all channels
        :param dt:
        :return:
        """
        K = self.get_num_frequency_bins()
        N = self.get_num_time_bins()
        M = self.get_num_channels()

        # num_output_channels = scipy.math.factorial(M-1)
        num_output_channels = M - 1

        msc_matrix = np.zeros((num_output_channels, K, N))

        out_idx = 0
        for idx0 in range(1):
            for idx1 in range(1, M):
                if idx1>idx0:
                    for k in range(K):
                        for n in range(dt/2, N - dt/2):
                            x1 = self.data[idx0, k, n:n + dt]
                            x2 = herm(self.data[idx1, k, n:n + dt])

                            num = float(np.power(np.abs(np.mean(x1 * x2)), 2))
                            den = float(np.mean(np.power(np.abs(x1), 2)) * np.mean(np.power(np.abs(x2), 2)))
                            if den == 0:
                                den = 1e-10
                            msc_matrix[out_idx, k, n] = num / den

                        # Borders: copy neighbor values
                        for n in range(0, dt / 2):
                            msc_matrix[out_idx, k, n] =  msc_matrix[out_idx, k, dt/2]

                        for n in range(N - dt / 2, N):
                            msc_matrix[out_idx, k, n] = msc_matrix[out_idx, k, N - (dt/2) - 1]


                    out_idx += 1

        return Stft(self.t, self.f, msc_matrix, self.sample_rate)

    # def compute_msc_lambda(self, l=0.7):
    #     """
    #     Magnitud square coherence among all channels
    #     :param dt:
    #     :return:
    #     """
    #     K = self.get_num_frequency_bins()
    #     N = self.get_num_time_bins()
    #     M = self.get_num_channels()
    #
    #     # num_output_channels = scipy.math.factorial(M-1)
    #     num_output_channels = M - 1
    #
    #     msc_matrix = np.zeros((num_output_channels, K, N))
    #
    #     out_idx = 0
    #     for idx0 in range(1):
    #         for idx1 in range(1, M):
    #             if idx1>idx0:
    #                 for k in range(K):
    #                     for n in range(N):
    #                         x1 = self.data[idx0, k, n]
    #                         x2 = herm(self.data[idx1, k, n])
    #                         num = float(np.power(np.abs(x1 * x2), 2))
    #                         den = float(np.power(np.abs(x1), 2) * np.power(np.abs(x2), 2))
    #                         if den == 0:
    #                             den = 1e-10
    #                         current = num / den
    #                         last = msc_matrix[out_idx,k,n-1] if n>0 else 1
    #                         msc_matrix[out_idx, k, n] = (l * last) + ((1 - l) * current)
    #                 out_idx += 1
    #
    #     return Stft(self.t, self.f, msc_matrix, self.sample_rate)

    def compute_msc_re(self,r=5):
        """
        Magnitud square coherence among all channels
        :param dt:
        :return:
        """
        K = self.get_num_frequency_bins()
        N = self.get_num_time_bins()
        M = self.get_num_channels()

        # num_output_channels = scipy.math.factorial(M-1)
        num_output_channels = M - 1

        msc_matrix = np.zeros((num_output_channels, K, N))

        out_idx = 0
        for idx0 in range(1):
            for idx1 in range(1, M):
                if idx1>idx0:
                    for k in range(K):
                        for n in range(r, N - r):
                            x1 = self.data[idx0, k, n-r:n+r+1]
                            x2 = herm(self.data[idx1,k, n-r:n+r+1])

                            num = float(np.power(np.abs(np.mean(np.real(x1 * x2))), 2))
                            den = float(np.mean(np.power(np.abs(x1), 2)) * np.mean(np.power(np.abs(x2), 2)))
                            if den == 0:
                                den = 1e-10
                            msc_matrix[out_idx, k, n] = num / den

                        # Borders: copy neighbor values
                        for n in range(0, r):
                            msc_matrix[out_idx, k, n] =  msc_matrix[out_idx, k, r]

                        for n in range(N - r, N):
                            msc_matrix[out_idx, k, n] = msc_matrix[out_idx, k, N - r - 1]


                    out_idx += 1

        return Stft(self.t, self.f, msc_matrix, self.sample_rate)

    # def compute_msc_re_lambda(self, l=0.7):
    #     """
    #     Magnitud square coherence among all channels
    #     :param dt:
    #     :return:
    #     """
    #     K = self.get_num_frequency_bins()
    #     N = self.get_num_time_bins()
    #     M = self.get_num_channels()
    #
    #     # num_output_channels = scipy.math.factorial(M-1)
    #     num_output_channels = M - 1
    #
    #     msc_matrix = np.zeros((num_output_channels, K, N))
    #
    #     out_idx = 0
    #     for idx0 in range(1):
    #         for idx1 in range(1, M):
    #             if idx1>idx0:
    #                 for k in range(K):
    #                     for n in range(N):
    #                         x1 = self.data[idx0, k, n]
    #                         x2 = herm(self.data[idx1, k, n])
    #                         num = float(np.power(np.abs(np.real(x1 * x2)), 2))
    #                         den = float(np.power(np.abs(x1), 2) * np.power(np.abs(x2), 2))
    #                         if den == 0:
    #                             den = 1e-10
    #                         current = num / den
    #                         last = msc_matrix[out_idx,k,n-1] if n>0 else 1
    #                         msc_matrix[out_idx, k, n] = (l * last) + ((1 - l) * current)
    #                 out_idx += 1
    #
    #     return Stft(self.t, self.f, msc_matrix, self.sample_rate)


    # Magnitude squared weight
    def compute_msw(self,r=5):
        """
        Magnitud square weight for each tf bin
        :param dt:
        :return:
        """
        K = self.get_num_frequency_bins()
        N = self.get_num_time_bins()
        M = self.get_num_channels()

        num_output_channels = M - 1

        msw_matrix = np.zeros((num_output_channels, K, N))

        for ch in range(num_output_channels):
            for k in range(K):
                for n in range(r, N - r):

                    num = np.mean(np.power(np.abs(self.data[ch+1, k, n-r:n+r+1]), 2))
                    den = np.mean(np.power(np.linalg.norm(self.data[1:,k, n-r:n+r+1], axis=0), 2))

                    if den == 0:
                        den = 1e-10

                    msw_matrix[ch, k, n] = num / den

                # Borders: copy neighbor values
                for n in range(0, r):
                    msw_matrix[ch, k, n] =  msw_matrix[ch, k, r]

                for n in range(N - r, N):
                    msw_matrix[ch, k, n] = msw_matrix[ch, k,  N - r - 1]


        return Stft(self.t, self.f, msw_matrix, self.sample_rate)

    # def compute_mw(self,dt=10):
    #     """
    #     Magnitud weight for each tf bin
    #     :param dt:
    #     :return:
    #     """
    #     K = self.get_num_frequency_bins()
    #     N = self.get_num_time_bins()
    #     M = self.get_num_channels()
    #
    #     num_output_channels = M - 1
    #
    #     msw_matrix = np.zeros((num_output_channels, K, N))
    #
    #     for ch in range(num_output_channels):
    #         for k in range(K):
    #             for n in range(dt/2, N - dt/2):
    #
    #                 num = np.mean(np.abs(self.data[ch+1, k, n:n + dt]))
    #                 den = np.mean(np.linalg.norm(self.data[1:, k, n:n + dt], axis=0))
    #
    #                 if den == 0:
    #                     den = 1e-10
    #
    #                 msw_matrix[ch, k, n] = num / den
    #
    #             # Borders: copy neighbor values
    #             for n in range(0, dt / 2):
    #                 msw_matrix[ch, k, n] =  msw_matrix[ch, k, dt/2]
    #
    #             for n in range(N - dt / 2, N):
    #                 msw_matrix[ch, k, n] = msw_matrix[ch, k, N - (dt/2) - 1]
    #
    #
    #     return Stft(self.t, self.f, msw_matrix, self.sample_rate)

    # def compute_ksi(self, dt=10):
    #
    #     K = self.get_num_frequency_bins()
    #     N = self.get_num_time_bins()
    #     ksi = np.zeros((1, K, N))
    #     z = p0 * c
    #
    #     # Time average computed from both past and future samples
    #     for k in range(K):
    #         for n in range(dt / 2, N - dt / 2):
    #
    #             W = self.data[0, k, n-r:n+r+1]
    #             X = self.data[1:, k, n-r:n+r+1]
    #
    #             P = W
    #             U = -X/z
    #
    #             num = np.linalg.norm(np.mean(np.real(np.conjugate(P) * U), axis=1))
    #             den = np.sqrt( np.power(np.linalg.norm(U,axis=0), 2)) * np.mean(np.power(abs(P), 2) )
    #
    #             # p = np.conjugate(self.data[0,k,n:n + dt])
    #             # v = self.data[1:,k,n:n + dt]
    #             # pv = p*v
    #             # re_pv = np.real(pv)
    #             # num = np.linalg.norm(np.mean(re_pv,axis=1))
    #             #
    #             # a = np.mean(np.power(abs(p), 2))
    #             # b = np.mean(np.power(np.linalg.norm(v,axis=0), 2))
    #             # den = np.sqrt(a*b)
    #
    #             ksi[0, k, n] = float(num)/den
    #
    #         # Borders: copy neighbor values
    #         for n in range(0, dt / 2):
    #             ksi[0, k, n] = ksi[0, k, dt / 2]
    #
    #         for n in range(N - dt / 2, N):
    #             ksi[0, k, n] = ksi[0, k, N - (dt / 2) - 1]
    #
    #     return Stft(self.t, self.f, ksi, self.sample_rate)
    #
    # def compute_ksi2(self, dt=10):
    #
    #     K = self.get_num_frequency_bins()
    #     N = self.get_num_time_bins()
    #     ksi = np.zeros((1, K, N))
    #
    #     # Time average computed from both past and future samples
    #     for k in range(K):
    #         for n in range(dt / 2, N - dt / 2):
    #         # for n in range(N):
    #
    #             p = np.conjugate(self.data[0,k,n:n + dt])
    #             v = self.data[1:,k,n:n + dt]
    #             pv = p*v
    #             re_pv =pv
    #             num = np.linalg.norm(np.mean(re_pv,axis=1))
    #
    #             a = np.mean(np.power(abs(p), 2))
    #             b = np.mean(np.power(np.linalg.norm(v,axis=0), 2))
    #             den = np.sqrt(a*b)
    #
    #             ksi[0, k, n] = np.power(float(num)/den,2)
    #
    #         # Borders: copy neighbor values
    #         for n in range(0, dt / 2):
    #             ksi[0, k, n] = ksi[0, k, dt / 2]
    #
    #         for n in range(N - dt / 2, N):
    #             ksi[0, k, n] = ksi[0, k, N - (dt / 2) - 1]
    #
    #     return Stft(self.t, self.f, ksi, self.sample_rate)

    # def compute_ksi2_re(self, r=10):
    #
    #     K = self.get_num_frequency_bins()
    #     N = self.get_num_time_bins()
    #     ksi = np.zeros((1, K, N))
    #
    #     # Time average computed from both past and future samples
    #     for k in range(K):
    #         for n in range(dt / 2, N - dt / 2):
    #             # for n in range(N):
    #
    #             p = np.conjugate(self.data[0, k, n-(dt/2):n + (dt/2)])
    #             v = self.data[1:, k, n-(dt/2):n + (dt/2)]
    #             pv = p * v
    #             re_pv = np.real(pv)
    #             num = np.linalg.norm(np.mean(re_pv, axis=1))
    #
    #             a = np.mean(np.power(abs(p), 2))
    #             b = np.mean(np.power(np.linalg.norm(v, axis=0), 2))
    #             den = np.sqrt(a * b)
    #
    #             ksi[0, k, n] = np.power(float(num) / den, 2)
    #
    #         # Borders: copy neighbor values
    #         for n in range(0, dt / 2):
    #             ksi[0, k, n] = ksi[0, k, dt / 2]
    #
    #         for n in range(N - dt / 2, N):
    #             ksi[0, k, n] = ksi[0, k, N - (dt / 2) - 1]
    #
    #     return Stft(self.t, self.f, ksi, self.sample_rate)

    def compute_ksi(self, r=5):

        K = self.get_num_frequency_bins()
        N = self.get_num_time_bins()
        ksi = np.zeros((1, K, N))
        z = p0 * c

        # Time average computed from both past and future samples
        for k in range(K):
            for n in range(r, N - r):

                W = self.data[0, k, n - r:n + r + 1]
                X = self.data[1:, k, n - r:n + r + 1]

                P = W
                U = -X / z

                num = np.linalg.norm(np.mean(np.conjugate(P) * U, axis=1))
                den = np.sqrt(np.mean(np.power(np.linalg.norm(U, axis=0), 2)) * np.mean(np.power(abs(P), 2)))

                ksi[0, k, n] = float(num) / den

            # Borders: copy neighbor values
            for n in range(0, r):
                ksi[0, k, n] = ksi[0, k, r]

            for n in range(N - r, N):
                ksi[0, k, n] = ksi[0, k, N - r - 1]

        return Stft(self.t, self.f, ksi, self.sample_rate)

    def compute_ksi_re(self, r=5):

        K = self.get_num_frequency_bins()
        N = self.get_num_time_bins()
        ksi = np.zeros((1, K, N))
        z = p0 * c

        # Time average computed from both past and future samples
        for k in range(K):
            for n in range(r, N - r):

                W = self.data[0, k, n - r:n + r + 1]
                X = self.data[1:, k, n - r:n + r + 1]

                P = W
                U = -X / z

                num = np.linalg.norm(np.mean(np.real(np.conjugate(P) * U), axis=1))
                den = np.sqrt( np.mean(np.power(np.linalg.norm(U, axis=0), 2)) * np.mean(np.power(abs(P), 2)) )

                ksi[0, k, n] = float(num) / den

            # Borders: copy neighbor values
            for n in range(0, r):
                ksi[0, k, n] = ksi[0, k, r]

            for n in range(N - r, N):
                ksi[0, k, n] = ksi[0, k, N - r - 1]

        return Stft(self.t, self.f, ksi, self.sample_rate)

    def compute_ksi_re_squared(self, r=5):

        K = self.get_num_frequency_bins()
        N = self.get_num_time_bins()
        ksi = np.zeros((1, K, N))
        z = p0 * c

        # Time average computed from both past and future samples
        for k in range(K):
            for n in range(r, N - r):

                W = self.data[0, k, n - r:n + r + 1]
                X = self.data[1:, k, n - r:n + r + 1]

                P = W
                U = -X / z

                num = np.power(np.linalg.norm(np.mean(np.real(np.conjugate(P) * U), axis=1)),2)
                den = np.mean(np.power(np.linalg.norm(U, axis=0), 2)) * np.mean(np.power(abs(P), 2))

                ksi[0, k, n] = float(num) / den

            # Borders: copy neighbor values
            for n in range(0, r):
                ksi[0, k, n] = ksi[0, k, r]

            for n in range(N - r, N):
                ksi[0, k, n] = ksi[0, k, N - r - 1]

        return Stft(self.t, self.f, ksi, self.sample_rate)


    # def compute_ksi2_re(self, r=5):
    #
    #     K = self.get_num_frequency_bins()
    #     N = self.get_num_time_bins()
    #     ksi = np.zeros((1, K, N))
    #
    #     # Time average computed from both past and future samples
    #     for k in range(K):
    #         for n in range(r, N - r):
    #             # for n in range(N):
    #
    #             p = np.conjugate(self.data[0, k, n-r:n+r+1])
    #             v = self.data[1:, k, n-r:n+r+1]
    #             pv = p * v
    #             re_pv = np.real(pv)
    #             num = np.linalg.norm(np.mean(re_pv, axis=1))
    #
    #             a = np.mean(np.power(abs(p), 2))
    #             b = np.mean(np.power(np.linalg.norm(v, axis=0), 2))
    #             den = np.sqrt(a * b)
    #
    #             ksi[0, k, n] = np.power(float(num) / den, 2)
    #
    #         # Borders: copy neighbor values
    #         for n in range(0, r):
    #             ksi[0, k, n] = ksi[0, k, r]
    #
    #         for n in range(N - r, N):
    #             ksi[0, k, n] = ksi[0, k, N - r - 1]
    #
    #     return Stft(self.t, self.f, ksi, self.sample_rate)

    # def compute_ksi2_re_lambda(self, l=0.7):
    #
    #     K = self.get_num_frequency_bins()
    #     N = self.get_num_time_bins()
    #     ksi = np.zeros((1, K, N))
    #
    #     num_matrix = np.zeros((3, K, N))
    #     den_a_matrix = np.zeros((1, K, N))
    #     den_b_matrix = np.zeros((1, K, N))
    #
    #
    #     for k in range(K):
    #         for n in range(N):
    #             # Num
    #             p = np.conjugate(self.data[0,k,n])
    #             v = self.data[1:,k,n]
    #
    #             pv = p*v
    #             re_pv = np.real(pv)
    #             num_matrix[:, k, n] = re_pv
    #
    #             a = np.power(abs(p), 2)
    #             den_a_matrix[0, k, n] = a
    #
    #             b = np.power(np.linalg.norm(v,axis=0), 2)
    #             den_b_matrix[0, k, n] = b
    #
    #             last_num = num_matrix[:, k, n-1] if n>0 else 0.
    #             last_a = den_a_matrix[0, k, n-1] = a if n>0 else 1.
    #             last_b = den_a_matrix[0, k, n-1] = b if n>0 else 1.
    #
    #             current_num = (l * last_num) + ((1 - l) * re_pv)
    #             current_a = (l * last_a) + ((1 - l) * a)
    #             current_b = (l * last_b) + ((1 - l) * b)
    #
    #             ksi[0, k, n] = np.power(np.linalg.norm(current_num),2) / (current_a * current_b)
    #
    #             # num = np.linalg.norm(re_pv)
    #             #
    #             # # Den
    #             # current_a = np.mean(np.power(abs(p), 2))
    #             # current_b = np.mean(np.power(np.linalg.norm(v,axis=0), 2))
    #             # den = np.sqrt(a*b)
    #             # # if den == 0:
    #             # #     den = 1e-10
    #             #
    #             # current = np.power(float(num)/den,2)
    #             # last = ksi[0, k, n - 1] if n > 0 else 0.
    #             # ksi[0, k, n] = (l * last) + ((1 - l) * current)
    #
    #     return Stft(self.t, self.f, ksi, self.sample_rate)



    def compute_ita(self, dt=10):

        K = self.get_num_frequency_bins()
        N = self.get_num_time_bins()
        ita = np.zeros((1, K, N))

        # Time average computed from both past and future samples
        for k in range(K):
            for n in range(dt / 2, N - dt / 2):
            # for n in range(N):

                z = p0*c

                p = np.conjugate(self.data[0,k,n:n + dt])
                v = self.data[1:,k,n:n + dt]*(1./z)
                pv = p*v
                re_pv = pv
                num = 2*z*np.linalg.norm(np.mean(re_pv,axis=1))

                a = np.mean(np.power(abs(p), 2))
                b = np.mean(np.power(np.linalg.norm(v,axis=0), 2))
                den = (np.power(z,2)*b) + a

                ita[0, k, n] = float(num)/den

            # Borders: copy neighbor values
            for n in range(0, dt / 2):
                ita[0, k, n] = ita[0, k, dt / 2]

            for n in range(N - dt / 2, N):
                ita[0, k, n] = ita[0, k, N - (dt / 2) - 1]

        return Stft(self.t, self.f, ita, self.sample_rate)


    # def compute_ita_re(self, dt=10):
    #
    #     K = self.get_num_frequency_bins()
    #     N = self.get_num_time_bins()
    #     ita = np.zeros((1, K, N))
    #
    #     # Time average computed from both past and future samples
    #     for k in range(K):
    #         for n in range(dt / 2, N - dt / 2):
    #         # for n in range(N):
    #
    #             z = p0*c
    #
    #             p = np.conjugate(self.data[0,k,n:n + dt])
    #             v = self.data[1:,k,n:n + dt]*(1./z)
    #             pv = p*v
    #             re_pv = np.real(pv)
    #             num = 2*z*np.linalg.norm(np.mean(re_pv,axis=1))
    #
    #             a = np.mean(np.power(abs(p), 2))
    #             b = np.mean(np.power(np.linalg.norm(v,axis=0), 2))
    #             den = (np.power(z,2)*b) + a
    #
    #             ita[0, k, n] = float(num)/den
    #
    #         # Borders: copy neighbor values
    #         for n in range(0, dt / 2):
    #             ita[0, k, n] = ita[0, k, dt / 2]
    #
    #         for n in range(N - dt / 2, N):
    #             ita[0, k, n] = ita[0, k, N - (dt / 2) - 1]
    #
    #     return Stft(self.t, self.f, ita, self.sample_rate)

    def compute_ita(self, r=5):

        K = self.get_num_frequency_bins()
        N = self.get_num_time_bins()
        ita = np.zeros((1, K, N))

        # Time average computed from both past and future samples
        for k in range(K):
            for n in range(r, N - r):

                W = self.data[0, k, n-r:n+r+1]
                X = self.data[1:, k, n-r:n+r+1]

                num = 2*np.linalg.norm(np.mean(np.conjugate(W)*X,axis=1))
                den = np.mean(np.power(abs(W), 2) + np.power(np.linalg.norm(X,axis=0), 2))

                ita[0, k, n] = float(num)/den

            # Borders: copy neighbor values
            for n in range(0, r):
                ita[0, k, n] = ita[0, k, r]

            for n in range(N - r, N):
                ita[0, k, n] = ita[0, k, N - r - 1]

        return Stft(self.t, self.f, ita, self.sample_rate)

    def compute_ita_re(self, r=5):

        K = self.get_num_frequency_bins()
        N = self.get_num_time_bins()
        ita = np.zeros((1, K, N))

        # Time average computed from both past and future samples
        for k in range(K):
            for n in range(r, N - r):

                W = self.data[0, k, n-r:n+r+1]
                X = self.data[1:, k, n-r:n+r+1]

                num = 2*np.linalg.norm(np.mean(np.real(np.conjugate(W)*X),axis=1))
                den = np.mean(np.power(abs(W), 2) + np.power(np.linalg.norm(X,axis=0), 2))

                ita[0, k, n] = float(num)/den

            # Borders: copy neighbor values
            for n in range(0, r):
                ita[0, k, n] = ita[0, k, r]

            for n in range(N - r, N):
                ita[0, k, n] = ita[0, k, N - r - 1]

        return Stft(self.t, self.f, ita, self.sample_rate)

    def compute_ita_re_squared(self, r=5):

        K = self.get_num_frequency_bins()
        N = self.get_num_time_bins()
        ita = np.zeros((1, K, N))

        # Time average computed from both past and future samples
        for k in range(K):
            for n in range(r, N - r):

                W = self.data[0, k, n-r:n+r+1]
                X = self.data[1:, k, n-r:n+r+1]

                num = np.power(2*np.linalg.norm(np.mean(np.real(np.conjugate(W)*X),axis=1)),2)
                den = np.power(np.mean(np.power(abs(W), 2) + np.power(np.linalg.norm(X,axis=0), 2)),2)

                ita[0, k, n] = float(num)/den

            # Borders: copy neighbor values
            for n in range(0, r):
                ita[0, k, n] = ita[0, k, r]

            for n in range(N - r, N):
                ita[0, k, n] = ita[0, k, N - r - 1]

        return Stft(self.t, self.f, ita, self.sample_rate)


    # def compute_ita2_re(self, r=5):
    #
    #     K = self.get_num_frequency_bins()
    #     N = self.get_num_time_bins()
    #     ita = np.zeros((1, K, N))
    #
    #     # Time average computed from both past and future samples
    #     for k in range(K):
    #         for n in range(r, N - r):
    #
    #             W = self.data[0, k, n-r:n+r+1]
    #             X = self.data[1:, k, n-r:n+r+1]
    #
    #             num = 2*np.linalg.norm(np.mean(np.real(np.conjugate(W)*X),axis=1))
    #             den = np.mean(np.power(abs(W), 2) + np.power(np.linalg.norm(X,axis=0), 2))
    #
    #             ita[0, k, n] = np.power(float(num) / den, 2)
    #
    #         # Borders: copy neighbor values
    #         for n in range(0, r):
    #             ita[0, k, n] = ita[0, k, r]
    #
    #         for n in range(N - r, N):
    #             ita[0, k, n] = ita[0, k, N - r - 1]
    #
    #     return Stft(self.t, self.f, ita, self.sample_rate)





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
    data = np.real(u_kn.data * np.conjugate(p_kn.data))
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

    data = ((p0/2.)*s1) + ((1./(2*p0*np.power(c,2)))*s2)
    return Signal(data,signal.sample_rate)

def compute_energy_density_tf(stft):
    p_kn = compute_sound_pressure(stft)
    u_kn = compute_particle_velocity(stft)

    s1 = np.power(np.linalg.norm(u_kn.data,axis=0), 2)
    s2 = np.power(abs(p_kn.data), 2)

    data = ((p0/2.)*s1) + ((1./(2*p0*np.power(c,2)))*s2)
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

def compute_diffuseness(arg,dt=10):
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
        return compute_diffuseness_t(arg,dt)
        # return compute_diffuseness_norm_t(arg)
    elif isinstance(arg,Stft):
        return compute_diffuseness_tf(arg,dt)
        # return compute_diffuseness_norm_tf(arg)
    else:
        raise TypeError

def compute_diffuseness_t(signal,dt=10):
    i_n = compute_intensity_vector(signal)
    e_n = compute_energy_density(signal)

    data = 1 - (np.linalg.norm(i_n.data,axis=0)/(c*e_n.data))
    return Signal(data,signal.sample_rate)

def compute_diffuseness_tf(stft,dt=10):
    # i_kn = compute_intensity_vector(stft)
    # e_kn = compute_energy_density(stft)
    #
    # K = stft.get_num_frequency_bins()
    # N = stft.get_num_time_bins()
    # dif = np.zeros((1, K, N))
    #
    # for k in range(K):
    #     for n in range(dt / 2, N - dt / 2):
    #         num = np.linalg.norm(np.mean(i_kn.data[:, k, n:n + dt], axis=1))
    #         den = c * np.mean(e_kn.data[:,k,n:n+dt])
    #         dif[0,k,n] = 1 - (num/den)
    #
    #     # Borders: copy neighbor values
    #     for n in range(0, dt/2):
    #         dif[0, k, n] = dif[0, k, dt/2]
    #
    #     for n in range(N - dt / 2, N):
    #         dif[0, k, n] = dif[0, k, N - (dt/2) - 1]
    #
    # return Stft(stft.t, stft.f, dif, stft.sample_rate)

    i_kn = compute_intensity_vector(stft)
    e_kn = compute_energy_density(stft)

    K = stft.get_num_frequency_bins()
    N = stft.get_num_time_bins()
    dif = np.zeros((1, K, N))

    for k in range(K):
        for n in range(dt / 2, N - dt / 2):
            num = np.linalg.norm(np.mean(i_kn.data[:, k, n:n + dt], axis=1))
            den = c * np.mean(e_kn.data[:,k,n:n+dt])
            dif[0,k,n] = 1 - (num/den)

        # Borders: copy neighbor values
        for n in range(0, dt/2):
            dif[0, k, n] = dif[0, k, dt/2]

        for n in range(N - dt / 2, N):
            dif[0, k, n] = dif[0, k, N - (dt/2) - 1]

    return Stft(stft.t, stft.f, dif, stft.sample_rate)


    # i_kn = compute_intensity_vector(stft)
    # e_kn = compute_energy_density(stft)
    #
    # K = stft.get_num_frequency_bins()
    # N = stft.get_num_time_bins()
    # dif = np.zeros((1, K, N))
    #
    # for k in range(K):
    #     for n in range(N):
    #         num = np.linalg.norm(i_kn.data[:, k, n])
    #         den = c * e_kn.data[:,k,n]
    #         current_dif = 1 - (num/den)
    #         last_dif = dif[0,k,n-1] if n>0 else 1
    #         dif[0,k,n] = (l*last_dif) + ((1-l)*current_dif)
    #
    # return Stft(stft.t, stft.f, dif, stft.sample_rate)

    # It also works like this...

    # i_kn = compute_intensity_vector(stft)
    # e_kn = compute_energy_density(stft)
    #
    # K = stft.get_num_frequency_bins()
    # N = stft.get_num_time_bins()
    # dif = np.zeros((1, K, N))
    #
    # for k in range(K):
    #     for n in range(N):
    #         num = 2*np.linalg.norm(np.real(stft.data[1:,k,n]*np.conj(stft.data[0,k,n])))
    #         den = np.power(np.linalg.norm(stft.data[1:,k,n]),2)+np.power(np.abs(stft.data[0,k,n]),2)
    #         current_dif = 1 - (num/den)
    #         last_dif = dif[0,k,n-1] if n>0 else 1
    #         dif[0,k,n] = (l*last_dif) + ((1-l)*current_dif)
    #
    # return Stft(stft.t, stft.f, dif, stft.sample_rate)






def compute_directivity(arg,dt=10):
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
        return compute_directivity_t(arg,dt)
        # return compute_diffuseness_norm_t(arg)
    elif isinstance(arg,Stft):
        return compute_directivity_tf(arg,dt)
        # return compute_diffuseness_norm_tf(arg)
    else:
        raise TypeError

def compute_directivity_t(signal,dt=10):
    i_n = compute_intensity_vector(signal)
    e_n = compute_energy_density(signal)

    data = np.linalg.norm(i_n.data,axis=0)/(c*e_n.data)
    return Signal(data,signal.sample_rate)

def compute_directivity_tf(stft,dt=10):
    i_kn = compute_intensity_vector(stft)
    e_kn = compute_energy_density(stft)

    K = stft.get_num_frequency_bins()
    N = stft.get_num_time_bins()
    dir = np.zeros((1, K, N))

    # Time average computed from both past and future samples
    for k in range(K):
        for n in range(dt/2, N - dt/2):
            num = np.linalg.norm(np.mean(i_kn.data[:, k, n:n + dt], axis=1))
            den = c * np.mean(e_kn.data[:,k,n:n+dt])
            dir[0,k,n] = num/den

        # Borders: copy neighbor values
        for n in range(0, dt/2):
            dir[0, k, n] = dir[0, k, dt/2]

        for n in range(N - dt / 2, N):
            dir[0, k, n] = dir[0, k, N - (dt/2) - 1]

    return Stft(stft.t, stft.f, dir, stft.sample_rate)


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



def compute_DPD_test(arg, J_k, J_n, erank_th):
    '''
    Compute the Direct Patch Dominance test, as defined in
        Nadiri, O., & Rafaely, B. (2014).
        Localization of multiple speakers under high reverberation
        using a spherical microphone array and the direct-path dominance test.
        https://doi.org/10.1109/TASLP.2014.2337846

    For each spectrogram TF bin, compute the Spatial Correlation Matrix (SCM)
    across the Ambisonics Channels dimmension, with time and frequency smoothing (averaging)
    The SCM will have an effective rank corresponding to the number
    of linearly independent basis (uncorrelated signals).
    We therefore estimate the erank(SCM(k,n))=1 through the singular value ratio
    after SVD decomposition. More precisely, when s1 is significatively bigger than s2.

    Parameters
    :arg: Instance of Signal or STFT
    :J_k: Number of bins in the frequency neighborhood
    :J_n: Number of bins in the time neighborhood
    :erank_th: Threshold of the ratio s1/s2 for which to consider a positive DPD.
        Typical values might be around 10.

    Returns:
    1-channel Signal or STFT

    Raises:
    TypeError if arg is not valid
    '''
    if isinstance(arg,Signal):
        return compute_DPD_test_t(arg, J_k, J_n, erank_th)

    elif isinstance(arg,Stft):
        return compute_DPD_test_tf(arg, J_k, J_n, erank_th)

    else:
        raise TypeError

def compute_DPD_test_t(signal, J_k, J_n, erank_th):

    raise NotImplementedError

def compute_DPD_test_tf(stft, J_k, J_n, erank_th):

    if not isinstance(J_k, int):
        raise TypeError
    elif not J_k >= 0:
        raise ValueError

    if not isinstance(J_n, int):
        raise TypeError
    elif not J_n >= 0:
        raise ValueError

    if not isinstance(erank_th, Number):
        raise TypeError
    elif not erank_th >= 1:
        raise ValueError


    K = stft.get_num_frequency_bins()
    N = stft.get_num_time_bins()
    M = stft.get_num_channels()

    dpd_mask = np.zeros((K, N))             # Result matrix
    J = ((2 * J_k) + 1) * ((2 * J_n) + 1)   # neighborhood total size in TF bins

    # Iterate over TF bins
    for k in range(K):
        for n in range(N):

            outOfRange = False
            R_a = np.zeros((M, M))  # Matrix to hold SCM values for each TF bin

            # Iterate over each neighborhood element
            for j_k in range(-J_k, J_k + 1):
                for j_n in range(-J_n, J_n + 1):

                    # Index of the current neighbor TF bin
                    f = k + j_k
                    t = n + j_n

                    # Ensure that TF bin exists
                    if f >= 0 and f <= (K - 1) and t >= 0 and t <= (N - 1):
                        # compute outer product of each cross-spatial vector
                        # and accumulate into R_a
                        a = stft.data[:, f, t]
                        R_a = R_a + np.outer(a, herm(a))
                    # If not, stop the current loop and evaluate mask to 0
                    else:
                        outOfRange = True
                        break

            # Once we have all summed SCMs for a given TF neighborhood,
            # scale among number of bins and compute SVD decomposition
            if not outOfRange:
                R_a = R_a / J
                s = np.linalg.svd(R_a)[1]
                if (s[1] / s[2]) >= erank_th:
                    dpd_mask[k, n] = 1

    # Return rank_Ra in form of stft
    return Stft(stft.t,
                stft.f,
                np.expand_dims(dpd_mask,axis=0),    # just to force 3D
                stft.sample_rate)
