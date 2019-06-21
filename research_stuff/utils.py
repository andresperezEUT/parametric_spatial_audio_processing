import numpy as np
from matplotlib.patches import Ellipse
from scipy import ndimage as ndi
from matplotlib.colors import LogNorm


def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

def degree_to_radian(degree):
    return degree * 2 * np.pi / 360.

def radian_to_degree(rad):
    return rad * 360 / (2 * np.pi)

def cartesian_to_spherical_degree(cartesian_list): # degree
    x = cartesian_list[0]
    y = cartesian_list[1]
    z = cartesian_list[2]
    r = np.sqrt((x*x)+(y*y)+(z*z))
    azimuth = np.arctan2(y,x)
    elevation = np.arcsin((z/r)) if r!=0 else 0
    return [radian_to_degree(azimuth),radian_to_degree(elevation),r]

def cartesian_to_spherical_radian(cartesian_list): # radian
    x = cartesian_list[0]
    y = cartesian_list[1]
    z = cartesian_list[2]
    r = np.sqrt((x*x)+(y*y)+(z*z))
    azimuth = np.arctan2(y,x)
    elevation = np.arcsin((z/r)) if r!=0 else 0
    return [azimuth,elevation,r]

def spherical_to_cartesian_degree(spherical_list): # degree
    azi = degree_to_radian(spherical_list[0])
    ele = degree_to_radian(spherical_list[1])
    r = spherical_list[2]
    x = r * np.cos(ele) * np.cos(azi)
    y = r * np.cos(ele) * np.sin(azi)
    z = r * np.sin(ele)
    return [x,y,z]

def spherical_to_cartesian_radian(spherical_list): # radian
    azi = spherical_list[0]
    ele = spherical_list[1]
    r = spherical_list[2]
    x = r * np.cos(ele) * np.cos(azi)
    y = r * np.cos(ele) * np.sin(azi)
    z = r * np.sin(ele)
    return [x,y,z]

def compute_image_source_positions(wall0, wall1, emitter, order=1):
    positions = []
    room_size = [wall1[0] - wall0[0], wall1[1] - wall0[1], wall1[2] - wall0[2]]

    def compute_position(x, idx, room_size):
        if x % 2 == 0:  # even: transposed position
            xpos = emitter[idx] + (x * room_size[idx])
        else:  # odd: symmetric transposed positions
            xpos = (wall1[idx] - emitter[idx]) + (x * room_size[idx])
        return xpos

    for x in range(-order, order + 1):
        for y in range(-order, order + 1):
            for z in range(-order, order + 1):
                if np.linalg.norm([x, y, z]) <= order:

                    current_position = []
                    for idx, v in enumerate([x, y, z]):
                        p = compute_position(v, idx, room_size)
                        current_position.append(p)
                    positions.append(np.asarray(current_position))

    return np.asarray(positions)

def find_maximal_time_bin(peak_sample, stft, overlap_factor):
    # Pass time bins to samples
    stft_time_bins = stft.t * stft.sample_rate
    # Find the index of the closest time bin
    next_bin_index = np.searchsorted(stft_time_bins, peak_sample)
    # Now find in which analysis window the energy is maximum
    # TODO: implement this for overlapping::window_index = next_bin_index - (overlap_factor / 2)
    window_index = next_bin_index - 1
    # Correct out of bounds
    if window_index < 0:
        window_index = 0
    elif window_index > len(stft_time_bins):
        window_index = len(stft_time_bins)
    return window_index

def threshold_local(image, block_size, method='gaussian', offset=0,
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
    >>> from skimage.data import camera
    >>> image = camera()[:50, :50]
    >>> binary_image1 = image > threshold_local(image, 15, 'mean')
    >>> func = lambda arr: arr.mean()
    >>> binary_image2 = image > threshold_local(image, 15, 'generic',
    ...                                         param=func)
    """
    if block_size % 2 == 0:
        raise ValueError("The kwarg ``block_size`` must be odd! Given "
                         "``block_size`` {0} is even.".format(block_size))
    # assert_nD(image, 2)
    thresh_image = np.zeros(image.shape, 'double')
    if method == 'generic':
        ndi.generic_filter(image, param, block_size,
                           output=thresh_image, mode=mode, cval=cval)
    elif method == 'gaussian':
        if param is None:
            # automatically determine sigma which covers > 99% of distribution
            sigma = (block_size - 1) / 6.0
        else:
            sigma = param
        ndi.gaussian_filter(image, sigma, output=thresh_image, mode=mode,
                            cval=cval)
    elif method == 'mean':
        mask = 1. / block_size * np.ones((block_size,))
        # separation of filters to speedup convolution
        ndi.convolve1d(image, mask, axis=0, output=thresh_image, mode=mode,
                       cval=cval)
        ndi.convolve1d(thresh_image, mask, axis=1, output=thresh_image,
                       mode=mode, cval=cval)
    elif method == 'median':
        ndi.median_filter(image, block_size, output=thresh_image, mode=mode,
                          cval=cval)
    else:
        raise ValueError("Invalid method specified. Please use `generic`, "
                         "`gaussian`, `mean`, or `median`.")

    return thresh_image - offset

def histogram(image, nbins=256):
    """Return histogram of image.
    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.
    The histogram is computed on the flattened image: for color images, the
    function should be used separately on each channel to obtain a histogram
    for each color channel.
    Parameters
    ----------
    image : array
        Input image.
    nbins : int
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    See Also
    --------
    cumulative_distribution
    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.camera())
    >>> np.histogram(image, bins=2)
    (array([107432, 154712]), array([ 0. ,  0.5,  1. ]))
    >>> exposure.histogram(image, nbins=2)
    (array([107432, 154712]), array([ 0.25,  0.75]))
    """
    sh = image.shape
    if len(sh) == 3 and sh[-1] < 4:
        warn("This might be a color image. The histogram will be "
             "computed on the flattened image. You can instead "
             "apply this function to each color channel.")

    # For integer types, histogramming with bincount is more efficient.
    if np.issubdtype(image.dtype, np.integer):
        offset = 0
        image_min = np.min(image)
        if image_min < 0:
            offset = image_min
            image_range = np.max(image).astype(np.int64) - image_min
            # get smallest dtype that can hold both minimum and offset maximum
            offset_dtype = np.promote_types(np.min_scalar_type(image_range),
                                            np.min_scalar_type(image_min))
            if image.dtype != offset_dtype:
                # prevent overflow errors when offsetting
                image = image.astype(offset_dtype)
            image = image - offset
        hist = np.bincount(image.ravel())
        bin_centers = np.arange(len(hist)) + offset

        # clip histogram to start with a non-zero bin
        idx = np.nonzero(hist)[0][0]
        return hist[idx:], bin_centers[idx:]
    else:
        hist, bin_edges = np.histogram(image.flat, bins=nbins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        return hist, bin_centers