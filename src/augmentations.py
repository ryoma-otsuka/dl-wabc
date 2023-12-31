"""
Data Augmentation for time-series sensor data

Otsuka et al., (2024) Methods in Ecology and Evolution
"Exploring deep Learning techniques for wild animal behaviour classification using animal-borne accelerometers"

"""

# The code is based on https://github.com/Tian0426/CL-HAR, but modified for this study.

import numpy as np
import torch
import scipy


def gen_aug(sample, da_type, da_param1=None, da_param2=None):
    '''Apply Data Augmentation
    Args:
        sample (): a data X (a time window, or a data instance) 
                sample.shape: (1, 50, 3) # (1, T, CH) (see DatasetLogbot2 in data_module.py)
        da_type (str): data augmentation type -> da_type
        da_param1 (float or int): data augmentation parameter 1
        da_param2 (float or int): data augmentation parameter 2 (only for t-warp)
    Returns:
        transformed sample
    '''
    if da_type == 'na':
        return sample
    elif da_type == 'scale':
        if da_param1 is None: # -> default value 
        # if there is no cfg.train.da_param1 in config file, da_param1 will be None (see data_module.py)
            # print(f"scaling da_param1: {da_param1}")
            scale_sample = scaling(sample, sigma=0.2)
        else:
            # print(f"scaling da_param1: {da_param1}")
            scale_sample = scaling(sample, sigma=da_param1) # 0.1, 0.2, 0.5
        return torch.from_numpy(scale_sample)
    elif da_type == 'noise':
        if da_param1 is None:
            # print(f"jittering da_param1: {da_param1}")
            noise_sample = jitter(sample, sigma=0.05)
        else:
            # print(f"jittering da_param1: {da_param1}")
            noise_sample = jitter(sample, sigma=da_param1) # 0.05, 0.1, 0.2
        return noise_sample
    elif da_type == 'perm':
        if da_param1 is None:
            # print(f"permutation da_param1: {da_param1}")
            permutation_sample = permutation(sample, max_segments=10)
        else:
            # print(f"permutation da_param: {da_param1}")
            permutation_sample = permutation(sample, max_segments=da_param1)
        return permutation_sample
    elif da_type == 't_warp':
        if da_param1 is None and da_param2 is None:
            # print(f"t_warp da_param1: {da_param1}, da_param2: {da_param2}")
            t_warp_sample = time_warp(sample, sigma=0.2, num_knots=4)
        elif da_param1 is None and da_param2 is not None:
            # print(f"t_warp da_param1: {da_param1}, da_param2: {da_param2}")
            t_warp_sample = time_warp(sample, sigma=0.2, num_knots=da_param2)
        elif da_param1 is not None and da_param2 is None:
            # print(f"t_warp da_param1: {da_param1}, da_param2: {da_param2}")
            t_warp_sample = time_warp(sample, sigma=da_param1, num_knots=4)
        else:
            # print(f"t_warp da_param1: {da_param1}, da_param2: {da_param2}")
            t_warp_sample = time_warp(sample, sigma=da_param1, num_knots=da_param2)
        return torch.from_numpy(t_warp_sample)
    elif da_type == 'rotation':
        # determine the range (maximum) from which angles are sampled 
        if da_param1 is None or da_param1 in [180, "180"]:
            # print(f"rotation da_param1: {da_param1}")
            max_rotation_angle = np.pi # -> -np.pi to np.pi
        elif da_param1 in [90, "90"]:
            # print(f"rotation da_param1: {da_param1}")
            max_rotation_angle = np.pi/2
        elif da_param1 in [45, "45"]:
            # print(f"rotation da_param1: {da_param1}")
            max_rotation_angle = np.pi/4
        else:
            raise Exception(f'da_param1 "{da_param1}" is not appropriate for rotation')
        # apply rotation matrix
        if isinstance(multi_rotation(sample, max_rotation_angle), np.ndarray):
            return torch.from_numpy(multi_rotation(sample, max_rotation_angle))
        else:
            return multi_rotation(sample, max_rotation_angle)
    else:
        print('The task is not available!\n')

def shuffle(x):
    sample_ssh = []
    for data in x:
        p = np.random.RandomState(seed=21).permutation(data.shape[1])
        data = data[:, p]
        sample_ssh.append(data)
    return torch.stack(sample_ssh)


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1): # apply same distortion to the signals from each sensor
    # factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[1]))
    # print(f"x.shape[0]: {x.shape[0]}, x.shape[1]: {x.shape[1]}") # x.shape[0]: 1, x.shape[1]: 50 
    
    # sample one scaling factor per window
    factor = np.random.normal(loc=1.0, scale=sigma, size=1)
    # print(f"factor.shape: {factor.shape}") # factor.shape: (1,)
    # print(f"factor: {factor}")
    
    ai = []
    for i in range(x.shape[2]):
        xi = x[:, :, i]
        ai.append(np.multiply(xi, factor)[:, :, np.newaxis])
        # print(f"len(ai): {len(ai)}")
        # print(ai)
    return np.concatenate((ai), axis=2)

def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, 
                                                num_segs[i] - 1, 
                                                replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            np.random.shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[warp, :]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def multi_rotation(x, max_rotation_angle=None):
    """
    Args:
        x: X.shape = (1, 50, 3)
            x.shape = (1, T, CH)
        max_rotation_angle: 
    return:
        x_rot: 
    """
    n_channel = x.shape[2]
    n_rot = n_channel // 3 # n_channel = 3 -> n_rot = 0
    x_rot = np.array([])
    for i in range(n_rot):
        if x_rot.size:
            x_rot = np.concatenate((x_rot, rotation(x[:, :, i * 3:i * 3 + 3]), max_rotation_angle), axis=2)
        else:
            x_rot = rotation(x[:, :, i * 3:i * 3 + 3], max_rotation_angle)
        # n_rot = 0 -> rotation(x[:, :, 0:3]) 
    return x_rot

def rotation(X, max_rotation_angle=None):
    """
    Applying a random 3D rotation
    "rotate the 3-axial (x, y and z) readings of each sensor by a random degree, 
    which follows a uniform distribution between −𝜋 and 𝜋, around a random axis in the 3D space" (Qian et al. 2022)
    Args:
        X: X.shape=(1, 50, 3) in the case of dl-wabc implementation (see data_module.py)
        angle_range: 
    Return:

    """

    # sample random axes in the 3D space
    axes = np.random.uniform(low=-1, high=1, size=(X.shape[0], X.shape[2])) # size=(1, 3)

    # sample random angles 
    if max_rotation_angle is None:
        angles = np.random.uniform(low=-np.pi, high=np.pi, size=(X.shape[0])) # size = (1) -> one angle per window 
    else:
        angles = np.random.uniform(low = (-1)*max_rotation_angle, high = max_rotation_angle, size = (X.shape[0])) # size = (1) -> one angle per window 
    
    # get rotation matrix
    matrices = axis_angle_to_rotation_matrix_3d_vectorized(axes, angles)

    return np.matmul(X, matrices) # apply rotation matrix


def axis_angle_to_rotation_matrix_3d_vectorized(axes, angles):
    """
    Get the rotational matrix corresponding to a rotation of (angle) radian around the axes
    Reference: the Transforms3d package - transforms3d.axangles.axangle2mat
    Formula: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    axes = axes / np.linalg.norm(axes, ord=2, axis=1, keepdims=True) # ord=2 -> L2 norm
    x = axes[:, 0]; y = axes[:, 1]; z = axes[:, 2]
    c = np.cos(angles)
    s = np.sin(angles)
    C = 1 - c

    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC

    m = np.array([
        [ x*xC+c,   xyC-zs,   zxC+ys ],
        [ xyC+zs,   y*yC+c,   yzC-xs ],
        [ zxC-ys,   yzC+xs,   z*zC+c ]])
    matrix_transposed = np.transpose(m, axes=(2,0,1))
    return matrix_transposed

def get_cubic_spline_interpolation(x_eval, x_data, y_data):
    """
    Get values for the cubic spline interpolation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
    """
    cubic_spline = scipy.interpolate.CubicSpline(x_data, y_data)
    return cubic_spline(x_eval)


def time_warp(X, sigma=0.2, num_knots=4):
    """
    Stretching and warping the time-series
    "stretch and warp each signal in the temporal dimension with an arbitrary cubic spline" (Qian et al. 2022)
    Args:
        sigma:
        num_knots:
    Return:
        X_transformed
    """
    time_stamps = np.arange(X.shape[1])

    # When num_knots = 4,
    # six equidistant time stamps: t=0.0, 9.8, 19.6, 29.4, 39.2, and 49.0
    knot_xs = np.arange(0, num_knots + 2, dtype=float) * (X.shape[1] - 1) / (num_knots + 1)

    # Sampling y-values for spline interpolation at each point of knot_xs    
    spline_ys = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0] * X.shape[2], num_knots + 2))

    # Cubic spline -> 50 values per axis
    spline_values = np.array(
        [get_cubic_spline_interpolation(time_stamps, 
                                        knot_xs, 
                                        spline_ys_individual) for spline_ys_individual in spline_ys]
    )

    # cumulative sum per axis
    cumulative_sum = np.cumsum(spline_values, axis=1)

    # divided by the maximum value and multiplied by 49 -> distorted time stamps per axis (maximum = 49)
    distorted_time_stamps_all = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (X.shape[1] - 1)

    # Linear interpolation -> stretched and warped time window
    X_transformed = np.empty(shape=X.shape)
    for i, distorted_time_stamps in enumerate(distorted_time_stamps_all):
        # https://numpy.org/doc/stable/reference/generated/numpy.interp.html
        X_transformed[i // X.shape[2], :, i % X.shape[2]] = np.interp(
            time_stamps, distorted_time_stamps, X[i // X.shape[2], :, i % X.shape[2]])
    return X_transformed
