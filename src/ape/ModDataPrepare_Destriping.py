#!/usr/bin/env python3
# coding: utf-8
"""
Destriping orbits by smoothening

Created on Dec 2 09:00 2020.
@author: Manu Goudar
"""

import numpy as np


# Initialize this class only once
# this class does either median or mean smoothening of the data
class Smoothdata:
    """Smoothen data."""

    def __init__(self, x, dx_half, operator, boundary_extension=True):
        self.operator = operator
        self.dx_half = dx_half
        self.x_size = x.size
        self.x = x
        self.boundary_extension = boundary_extension
        if self.boundary_extension:
            self.halfwin_start, self.halfwin_end, self.x1 = self.init_boundary()
            ind_win = self.get_averaging_window(self.x1)
        else:
            ind_win = self.get_averaging_window(self.x)
        # define start point
        self.start_pt = ind_win[0]
        if self.operator == "mean":
            # As it uses trapizodial rule it is important to subtract one
            # If mean is used then do not subtract
            _window = len(ind_win) - 1
            self.window = _window
        elif self.operator == "median":
            _window = len(ind_win)
            self.window = _window

    def get_averaging_window(self, x1):
        # compute window of averaging or median filtering
        # whether points in input data that are within
        # the window around the current center point
        sel = np.abs(x1 - self.x[0]) < self.dx_half
        ind_win = np.where(sel)[0]
        return ind_win

    def init_boundary(self):
        """
        This is computed once for all orbits as the box size remains same and x is invarient
        Parameters:
        Input:
            x: x values where data is sampled
            dx_half: half window length in x
        Output:
            halfwin_start: beginning block
            halfwin_end: last block
            x1: Extended x
        """
        # Compute hal lengths for boundary extension
        # first entries that are within a half window length
        halfwin_start = ((self.x - self.x[0]) <= self.dx_half) & (self.x != self.x[0])
        # similar for the end of the data
        halfwin_end = ((self.x[-1] - self.x) <= self.dx_half) & (self.x != self.x[-1])
        # extend x vector; note that x[0] - (x[1]-x[0]) = 2*x[0]+x[1];
        # take care that x remains sorted
        _first = np.flipud(2 * self.x[0] - self.x[halfwin_start])
        _last = np.flipud(2 * self.x[-1] - self.x[halfwin_end])
        x1 = np.concatenate((_first, self.x, _last))
        return halfwin_start, halfwin_end, x1

    def smooth_by_mean(self, y):
        """
        compute mean in a window
        TODO: Can be made faster by using cumsum or np.convolve
        """
        y0 = (y[:, :-1] + y[:, 1:]) / 2
        # window
        _win = self.window + self.start_pt
        y_smoothed = np.zeros((y.shape[0], self.x_size))
        for i in range(self.x_size):
            y_smoothed[:, i] = np.mean(y0[:, i + self.start_pt : i + _win], axis=1)
        return y_smoothed

    def smooth_by_median(self, y):
        """
        compute median in a window
        """
        # window
        _win = self.window + self.start_pt
        y_smoothed = np.zeros((y.shape[0], self.x_size))
        for i in range(self.x_size):
            y_smoothed[:, i] = np.median(y[:, i + self.start_pt : i + _win], axis=1)
        return y_smoothed

    def extend_and_smooth_data(self, y):
        if self.boundary_extension:
            # Extend data
            # y value of inversion point at beginning of data
            y_inv_start = np.median(y[:, 0:5], axis=1)
            # y value of inversion point at end of data
            y_inv_end = np.median(y[:, -5:], axis=1)
            # extend y vector (similar to above)
            _first = np.flip((2 * y_inv_start[:, np.newaxis] - y[:, self.halfwin_start]), axis=1)
            _last = np.flip((2 * y_inv_end[:, np.newaxis] - y[:, self.halfwin_end]), axis=1)
            y = np.concatenate((_first, y, _last), axis=1)
            # smooth and return data
            if self.operator == "mean":
                return self.smooth_by_mean(y)
            elif self.operator == "median":
                return self.smooth_by_median(y)


# Class for FFT Weights
class GaussianWeights:
    def __init__(self, max_sigma=7, min_sigma=0.3, center_hw=7, n_ground_px=2001, n_swath_pix=215):
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.center_hw = center_hw
        self.n_swath_pix = n_swath_pix
        self.n_ground_px = n_ground_px
        self.wt_pos, self.wt_neg, self.id_pos, self.id_neg = self.get_weight(
            self.n_ground_px, self.n_swath_pix
        )

    def gaussian(self, idx, n, mu, sig):
        if n > 0:
            n_half = 1 + np.int_(n / 2)
            xi = np.arange(n_half)
            result = np.zeros((n_half))
            for i in range(n_half):
                try:
                    result[i] = np.exp(-np.power((xi[i] - mu) / sig, 2.0) / 2.0)
                except FloatingPointError:
                    result[i] = 0.0
            rst = 1 - np.flip(result)
            fld = rst[:idx]
            fld1 = np.flip(fld[1:])
        return fld, fld1

    def get_sigma(self, m):
        sig = np.zeros((m))
        m_half = np.int_(m / 2)
        for i in range(m):
            if (i < m_half - self.center_hw) | (i > m_half + self.center_hw):
                sig[i] = np.maximum(
                    np.abs((i - m_half) / (m_half) * self.max_sigma), self.min_sigma
                )
        return sig[m_half:]

    def find_index_limit(self, n, mu, max_sigma):
        if n > 0:
            n_half = 1 + np.int_(n / 2)
            x = np.arange(n_half)
            result = np.zeros((n_half))
            for i in range(n_half):
                try:
                    result[i] = np.exp(-np.power((x[i] - mu) / max_sigma, 2.0) / 2.0)
                except FloatingPointError:
                    result[i] = 0.0
            rst = 1 - np.flip(result)
            rst[(1 - rst) < 1e-16] = 1
            return np.where(rst == 1)[0][0]

    def get_weight(self, n, m):
        """ """
        # mean for ground pixels for gaussian distribution
        mu = np.int_(n / 2)
        # Number of sigma values for computation in swatch direction
        m_half = np.int_(m / 2)
        # Get index where gaussian values are less than 1
        # These also represent first n frequencies
        idx = self.find_index_limit(n, mu, self.max_sigma)
        # compute sigma
        sigma = self.get_sigma(m)
        # size of the weights in two directions
        # From 1-idx frequency in ground pixel and 0-m_half+1 in swatch direction
        weight = np.zeros((idx, m_half + 1))  # For negative frequencies
        weight2 = np.zeros((idx - 1, m_half + 1))  # For positive frequencies
        for i in range(m_half + 1):
            if sigma[i] != 0:
                _l1, _l2 = self.gaussian(idx, n, mu, sigma[i])
                weight[:, i] = _l1
                weight2[:, i] = _l2
            else:
                weight[:, i] = 1
                weight2[:, i] = 1
        return weight, weight2, idx, -idx + 1

    def check_array_size(self, n):
        if n < self.id_pos * 2:
            # Recompute values
            self.wt_pos, self.wt_neg, self.id_pos, self.id_neg = self.get_weight(
                self.n_ground_px, self.n_swath_pix
            )


def median_mat(mat, axis=0, interp_nan=False, fill_nan=None):
    """docstring for median_mat"""
    dim = mat.shape
    result = np.zeros(dim)
    # calculate the median vector along the axis
    # median_vec = np.nanmedian(mat, axis=axis)
    if axis == 1:
        median_vec = np.zeros(dim[0])
        leng = dim[0]
        for i in range(dim[0]):
            if np.mean(np.isnan(mat[i, :])) != 1:
                median_vec[i] = np.nanmedian(mat[i, :])
            else:
                median_vec[i] = np.nan
    else:
        median_vec = np.zeros(dim[1])
        leng = dim[1]
        for i in range(dim[1]):
            if np.mean(np.isnan(mat[:, i])) != 1:
                median_vec[i] = np.nanmedian(mat[:, i])
            else:
                median_vec[i] = np.nan
    # fill possible nan values
    idx_finite = np.isfinite(median_vec)
    idx_nan = ~idx_finite
    if interp_nan is True:
        if idx_finite[0] is False:
            median_vec[0] = median_vec[idx_finite][0]
            idx_finite[0] = True
        if idx_finite[-1] is False:
            median_vec[-1] = median_vec[idx_finite][-1]
            idx_finite[-1] = False
        median_vec[idx_nan] = np.interp(
            np.arange(leng)[idx_nan], np.arange(leng)[idx_finite], median_vec[idx_finite]
        )
    if not (fill_nan is None):
        median_vec[idx_nan] = fill_nan
    # create the output matrix
    for i in range(dim[axis]):
        if axis == 0:
            result[i, :] = median_vec
        else:
            result[:, i] = median_vec
    return result


def preprocess_mat(mat):
    """
    Replace all nans by median values.
    This done as fft doesn't take nan values
    """
    # estimate the background
    background = median_mat(mat, axis=1, interp_nan=True)
    # estimate stripe pattern
    stripes = median_mat(mat - background, axis=0, fill_nan=0)
    stripes = stripes - np.median(stripes)
    # fill nan values
    idx_nan = np.isnan(mat)
    mat[idx_nan] = background[idx_nan] + stripes[idx_nan]
    result = mat
    return {"background": background, "stripes": stripes, "result": result, "idx_nan": idx_nan}


# Initialize variables for Median Filtering
x = np.arange(215)
med_smoothening = Smoothdata(x, 4, "median", True)
# mean static variables
mean_smoothening = Smoothdata(x, 3, "mean", True)

# initialize variables for FFT Filtering
wts = GaussianWeights()


# Actual function to calculate strip
def calc_stripe_mask_median(swath_mat, min_fraction=0.6):
    # prepare output
    n_swath, n_pix = swath_mat.shape
    res = np.zeros((n_swath, n_pix)) * np.nan
    res_m_mat = np.zeros((n_swath, n_pix)) * np.nan
    # check if the length is more than min_fraction
    _ids = np.where((np.sum(~np.isnan(swath_mat), axis=1) > min_fraction * n_pix))[0]
    # loop over the values that are more than min fraction
    # extend and smoothen data
    y1 = med_smoothening.extend_and_smooth_data(swath_mat[_ids])
    # extend and smoothen data
    y2 = mean_smoothening.extend_and_smooth_data(y1)
    res[_ids] = swath_mat[_ids] - y2
    res_m = np.nanmedian(res, axis=0)
    for i in range(n_swath):
        res_m_mat[i, :] = res_m
    return res, res_m, res_m_mat


def calc_stripe_mask_fft(swath_data):
    n, m = swath_data.shape
    # Check weight
    wts.check_array_size(n)
    idx_nan = np.isnan(swath_data)
    if idx_nan.all():  # if we have only nan data
        return {"stripe_correction": np.zeros(swath_data.shape), "avg_stripes": np.nan}
        # pre process the input data
    prep = preprocess_mat(swath_data)
    f = np.fft.rfft2(prep["result"])
    # multiply by weights
    f[: wts.id_pos, :] = np.multiply(f[: wts.id_pos, :], wts.wt_pos)
    f[wts.id_neg :, :] = np.multiply(f[wts.id_neg :, :], wts.wt_neg)
    # compute inverse fft and fill nan values in their locations
    res_m_mat = prep["result"] - np.fft.irfft2(f, s=(n, m))
    res_m_mat[idx_nan] = np.nan
    # Compute median along swath axis
    # res_m = np.nanmedian(res_m_mat, axis=1)
    return {"stripe_correction": res_m_mat}  # "avg_stripes": res_m}


def calc_stripe_mask(swath_mat, kind="both"):
    """
    swath_mat : Preprocessed CO data with mask replaced by nan values
    kind: median, fft or both
    preprocess: This is always True for FFT and can be True or False for median
    """
    # co_tmp = data['co_column'].filled(fill_value=np.nan)
    # qa_idx = (data["qa_value"] <= 0) | (data["qa_value"] > 1)
    # co_tmp[qa_idx] = np.nan
    if kind == "median":
        stripe_mat_median = calc_stripe_mask_median(swath_mat.copy())[2]
        return stripe_mat_median
    elif kind == "fft":
        rst = calc_stripe_mask_fft(swath_mat.copy())
        return rst["stripe_correction"]
    elif kind == "both":
        stripe_mat_median = calc_stripe_mask_median(swath_mat.copy())[2]
        rst = calc_stripe_mask_fft(swath_mat.copy())
        return stripe_mat_median, rst["stripe_correction"]
