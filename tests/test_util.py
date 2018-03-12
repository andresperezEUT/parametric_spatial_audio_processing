import pytest
import numpy as np
from parametric_spatial_audio_processing.util import cartesian_2_spherical, _validate_bformat_array, \
    convert_bformat_acn_2_fuma, convert_bformat_fuma_2_acn, convert_bformat_sn3d_2_fuma, convert_bformat_fuma_2_sn3d, \
    convert_bformat_n3d_2_sn3d


def test_validate_bformat_array():

    incorrect_args = [
        None,
        1.2,
        [1,2,3],
        [[1,2,3],[4,5,6]],
        np.zeros(shape=(1)),
        np.zeros(shape=(1,2,3)),
        np.zeros(shape=(5,100))
    ]
    for arg in incorrect_args:
        pytest.raises(AssertionError,_validate_bformat_array,arg)

    correct_args = [
        np.zeros(shape=(4,100))
    ]
    for arg in correct_args:
        try:
            _validate_bformat_array(arg)
        except AssertionError:
            raise


def test_convert_bformat_acn_2_fuma():

    # ACN Test signal (WYZX)
    acn = np.zeros((4,10))
    for i in range(1,4):
        acn[i:] = float(i)

    # FUMA Groundtruth (WXYZ)
    fuma = np.zeros((4,10))
    fuma[1:] = 3.0
    fuma[2:] = 1.0
    fuma[3:] = 2.0

    assert np.array_equal(fuma,convert_bformat_acn_2_fuma(acn))


def test_convert_bformat_fuma_2_acn():

    # FUMA Test signal (WXYZ)
    fuma = np.zeros((4,10))
    for i in range(1,4):
        fuma[i:] = float(i)

    # ACN Groundtruth (WYZX)
    acn = np.zeros((4,10))
    acn[1:] = 2.0
    acn[2:] = 3.0
    acn[3:] = 1.0

    assert np.array_equal(acn,convert_bformat_fuma_2_acn(fuma))

def test_convert_bformat_sn3d_2_fuma():

    # fuma[0] = sn3d[0]/sqrt(2)
    sn3d = np.ones((4,10))

    fuma = np.ones((4, 10))
    fuma[0] = 1/np.sqrt(2)

    assert np.array_equal(fuma,convert_bformat_sn3d_2_fuma(sn3d))

def test_convert_bformat_fuma_2_sn3d():

    # sn3d[0] = fuma[0]*sqrt(2)
    fuma = np.ones((4,10))

    sn3d = np.ones((4, 10))
    sn3d[0] = np.sqrt(2)

    assert np.array_equal(sn3d,convert_bformat_fuma_2_sn3d(fuma))

def test_convert_bformat_n3d_2_sn3d():

    # sn3d[1:3] = n3d[1:3]/sqrt(3)
    n3d = np.ones((4,10))
    n3d[1:] = np.sqrt(3)

    sn3d = np.ones((4, 10))

    assert np.allclose(sn3d,convert_bformat_n3d_2_sn3d(n3d))


def test_cartesian_2_spherical():

    ##### 2D #####

    # Test incorrect args
    incorrect_args = [
        [1,2],
        np.zeros([4]),
        np.ones([4,5,6]),
    ]
    for arg in incorrect_args:
        pytest.raises(TypeError,cartesian_2_spherical,arg)


    # Test correct values

    # [1,0,0] -> [0,0,1]
    c_array = np.zeros(shape=(3,10))
    c_array[0,:] += 1.0
    groundtruth = np.zeros(shape=(3,10))
    groundtruth[2, :] += 1.0
    assert np.allclose(cartesian_2_spherical(c_array),groundtruth)

    # [-1,0,0] -> [pi,0,1]
    c_array = np.zeros(shape=(3,10))
    c_array[0,:] -= 1.0
    groundtruth = np.zeros(shape=(3,10))
    groundtruth[0, :] += np.pi
    groundtruth[2, :] += 1
    assert np.allclose(cartesian_2_spherical(c_array),groundtruth)

    # [0,1,0] -> [pi/2,0,1]
    c_array = np.zeros(shape=(3, 10))
    c_array[1, :] += 1.0
    groundtruth = np.zeros(shape=(3, 10))
    groundtruth[0, :] += np.pi/2
    groundtruth[2, :] += 1
    assert np.allclose(cartesian_2_spherical(c_array), groundtruth)

    # [0,0,1] -> [0,0,1]
    c_array = np.zeros(shape=(3, 10))
    c_array[2, :] += 1.0
    groundtruth = np.zeros(shape=(3, 10))
    groundtruth[1, :] += np.pi/2
    groundtruth[2, :] += 1
    assert np.allclose(cartesian_2_spherical(c_array), groundtruth)


    ##### 3D #####

    # Test incorrect args
    incorrect_args = [
        [[[1, 2]]],
        np.zeros([4]),
        np.ones([4, 5, 6,7]),
    ]
    for arg in incorrect_args:
        pytest.raises(TypeError, cartesian_2_spherical, arg)

    # Test correct values

    # [1,0,0] -> [0,0,1]
    c_array = np.zeros(shape=(3, 4, 5))
    c_array[0, :, :] += 1.0
    groundtruth = np.zeros(shape=(3, 4, 5))
    groundtruth[2, :, :] += 1.0
    assert np.allclose(cartesian_2_spherical(c_array), groundtruth)

    # [-1,0,0] -> [pi,0,1]
    c_array = np.zeros(shape=(3, 4, 5))
    c_array[0, :, :] -= 1.0
    groundtruth = np.zeros(shape=(3, 4, 5))
    groundtruth[0, :, :] += np.pi
    groundtruth[2, :, :] += 1
    assert np.allclose(cartesian_2_spherical(c_array), groundtruth)

    # [0,1,0] -> [pi/2,0,1]
    c_array = np.zeros(shape=(3, 4, 5))
    c_array[1, :, :] += 1.0
    groundtruth = np.zeros(shape=(3, 4, 5))
    groundtruth[0, :, :] += np.pi / 2
    groundtruth[2, :, :] += 1
    assert np.allclose(cartesian_2_spherical(c_array), groundtruth)

    # [0,0,1] -> [0,0,1]
    c_array = np.zeros(shape=(3, 4, 5))
    c_array[2, :, :] += 1.0
    groundtruth = np.zeros(shape=(3, 4, 5))
    groundtruth[1, :, :] += np.pi / 2
    groundtruth[2, :, :] += 1
    assert np.allclose(cartesian_2_spherical(c_array), groundtruth)