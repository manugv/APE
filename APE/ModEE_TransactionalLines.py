#!/usr/bin/env python
# coding: utf-8
"""
compute emissions by massflux.

Created on Sept 2021
@author: Manu Goudar
"""

# Import for plume detection
import numpy as np
import pandas as pd

from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.signal import argrelmin

from shapely.geometry import LineString
from skimage.morphology import skeletonize

from .ModuleRefineGrids import RefineGridsUniform
from .ModuleDataContainers import DataContainer


##################################################################
# General Functions
##################################################################

def weighted_mean(y, wts=[]):
    if len(wts) == 0:
        return np.mean(y)
    else:
        # compute weighted average
        return (wts * y).sum() / wts.sum()


##################################################################
# Line fitting mainly for plume line
##################################################################


def compute_fit(x, y, deg):
    _nx = len(x)
    _w = np.ones(_nx) * 0.3
    _w[0] = 1.5
    line = np.zeros((_nx, 2))
    line[:, 0] = x.copy()
    # y_cp = y.copy()
    coeff = np.polyfit(x, y, deg, w=_w)
    p2 = np.poly1d(coeff)
    line[:, 1] = p2(x)
    return line


def rearrange_data(ddf, _ix, _nx):
    if ((_nx - _ix) / _nx < 0.35) or ((_nx - _ix) < 3):
        df2 = ddf.iloc[: _ix + 1, :]
        return df2.iloc[::-1]
    else:
        return ddf.iloc[_ix:, :]


def fit_plumeline(_lat, _lon, src, deg=1):
    # fit polynominal
    df = pd.DataFrame()
    df["lat"] = np.concatenate(([src[0]], _lat))
    df["lon"] = np.concatenate(([src[1]], _lon))
    _nx = len(df.lat)
    if df.lat.std() > df.lon.std():
        df1 = df.sort_values(by=["lat"])
        # Arrange in ascending order and compute start
        _ix = np.where(df1.lat == src[0])[0][0]
        df2 = rearrange_data(df1, _ix, _nx)
        if len(df2 > 15):
            _lin = compute_fit(df2.lat[:], df2.lon[:], deg)
        else:
            _lin = compute_fit(df2.lat, df2.lon, deg)
        return _lin
    else:
        df1 = df.sort_values(by=["lon"])
        _ix = np.where(df1.lon == src[1])[0][0]
        df2 = rearrange_data(df1, _ix, _nx)
        if len(df2 > 15):
            _lin = compute_fit(df2.lon[:], df2.lat[:], deg)
        else:
            _lin = compute_fit(df2.lon, df2.lat, deg)
        line = np.zeros_like(_lin)
        line[:, 0] = _lin[:, 1]
        line[:, 1] = _lin[:, 0]
        return line


##################################################################
# Functions for plume line
##################################################################
def get_point_at_within_dist(lat, lon, src, trans, dist=120):
    # get all points below 120km
    _sx, _sy = trans.latlon2xykm(src[0], src[1])
    x, y = trans.latlon2xykm(lat, lon)
    di = np.sqrt((x - _sx) ** 2 + (y - _sy) ** 2)
    _dis = np.minimum(di.max(), dist)
    return lat[di < _dis], lon[di < _dis]


def _get_distance_from_firstpt(_line, trans, fact=2.5):
    x, y = trans.latlon2xykm(_line[:, 0], _line[:, 1])
    _d = np.int_((np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)) / fact)
    return _d * fact


def compute_medial_line(lat_nodes, lon_nodes, plumemask, src, trans):
    """
    compute raw medial line
    This is not a refined line
    """
    refine = RefineGridsUniform(4)
    xn, yn, xc, yc = refine.resize_coordinates(lat_nodes.copy(), lon_nodes.copy())
    zval = refine.resize_values(plumemask)
    ml_id = skeletonize(zval)
    ln_lat, ln_lon = get_point_at_within_dist(xc[ml_id], yc[ml_id], src, trans, dist=60)
    return fit_plumeline(ln_lat, ln_lon, src, deg=2)


def check_wind_plume_alignment(interp, plume_pts, norm_dir_vector):
    # CHECK ALIGNEMENT
    # Interpolate values
    uv_ = np.zeros_like(plume_pts)
    uv_[0, :], uv_[1, :] = interp.interpolate_vel(plume_pts[0, :], plume_pts[1, :])
    # normalize velocity
    vel_dir = uv_ / np.linalg.norm(uv_, axis=0)
    # Compute angles between
    sh = plume_pts.shape
    agl = np.zeros(sh[1])
    for i in range(sh[1]):
        agl[i] = 180 * np.arccos(np.dot(vel_dir[:, i], norm_dir_vector[:, i])) / np.pi

    _med = np.median(abs(agl))
    # print("Median & mean of the angles is : ", _med, np.mean(agl))
    if _med < 20:
        _flag = True
    else:
        _flag = False
    return agl, _flag


def get_plumepoints_slope(_line, trans, ds=2.5, plume_len_km=50):
    # ds is spacing between transaction lines and it is 2.5 default
    # 100 km is the max length
    dist_tot = _get_distance_from_firstpt(_line, trans, ds)
    total_len_km = np.minimum(dist_tot, plume_len_km)
    # minimum number of theoritical points possible
    _min_pts = total_len_km / ds
    # convert lat-lon data to xy
    ll = np.zeros_like(_line)
    ll[:, 0], ll[:, 1] = trans.latlon2xykm(_line[:, 0], _line[:, 1])
    # Decide on number of points
    dist = np.sqrt(np.power(ll[:, 0], 2) + np.power(ll[:, 1], 2))
    # number of lines are defined based on min between possible and max
    n = np.int_(np.minimum(np.floor_divide(dist.max(), ds), _min_pts))
    # Create Linestring
    _lxy = LineString(tuple(map(tuple, ll)))
    # Get values at given ds
    plumeline_xy = [_lxy.interpolate(i * ds) for i in range(1, n + 1)]
    pts = np.zeros((2, n))
    for ii in range(n):
        pts[0, ii], pts[1, ii] = trans.xykm2latlon(plumeline_xy[ii].x, plumeline_xy[ii].y)
    # Get values to compute slope
    plumeline_xy_ds = [_lxy.interpolate(i * ds + 0.1) for i in range(1, n + 1)]
    # compute directional vector and normalize it
    vec_xy = np.zeros((2, n))
    for ii in range(n):
        vec_xy[0, ii] = plumeline_xy_ds[ii].x - plumeline_xy[ii].x
        vec_xy[1, ii] = plumeline_xy_ds[ii].y - plumeline_xy[ii].y
    vec_norm = vec_xy / np.linalg.norm(vec_xy, axis=0)

    # compute slope
    slope = np.zeros((n))
    for ii in range(n):
        if (plumeline_xy_ds[ii].x - plumeline_xy[ii].x) == 0:
            slope[ii] = 10000
        else:
            slope[ii] = (plumeline_xy_ds[ii].y - plumeline_xy[ii].y) / (plumeline_xy_ds[ii].x - plumeline_xy[ii].x)

    line_dist = (np.arange(n) + 1) * ds
    _dataline = {"plumeline": pts, "dist_from_src": line_dist, "direction_vector": vec_norm}
    return _dataline, LineString(plumeline_xy)


##################################################################
# Define Transaction line
##################################################################


def get_transactionline(n, ds, no_pts_one_side, pline):
    length_side = no_pts_one_side * ds
    m = np.int_(2 * no_pts_one_side + 1)
    m1 = no_pts_one_side
    line_pts = np.zeros((n, m, 2))
    for i in range(m1):
        left = pline.parallel_offset(ds * (i + 1), "left")
        right = pline.parallel_offset(ds * (i + 1), "right")
        line_pts[:, m1, :] = np.array(pline.coords)
        dxl = left.length / (n - 1)
        dxr = right.length / (n - 1)
        for j in range(n):
            line_pts[j, m1 - 1 - i, :] = np.array(left.interpolate((j) * dxl).coords)[0]
            # line_pts[j, m1+1+i, :] = np.array(right.interpolate((n-1-j) * dxr).coords)[0]
            line_pts[j, m1 + 1 + i, :] = np.array(right.interpolate((j) * dxr).coords)[0]
    return line_pts, np.arange(-length_side, length_side + 1, ds)


class InterpolateConcToTransactionLines:
    def __init__(self, co_edit, x_km, y_km):
        self.co_edit = co_edit
        self.x_km = x_km
        self.y_km = y_km
        self.interp = LinearNDInterpolator(
            np.column_stack((self.x_km.ravel(), self.y_km.ravel())), self.co_edit.ravel()
        )
        id_nan = ~np.isnan(co_edit)
        self.interp_nan = LinearNDInterpolator(
            np.column_stack((self.x_km[id_nan].ravel(), self.y_km[id_nan].ravel())), self.co_edit[id_nan].ravel()
        )
        self.sh = self.x_km.shape
        cc = np.c_[self.x_km.ravel(), self.y_km.ravel()]
        self.tree = cKDTree(cc)

    def compute_weights(self, dd, modified=False):
        if modified:
            r = 8
            wts = np.zeros_like(dd)
            wts = ((r - dd) / (r * dd)) ** 2
            return wts / np.sum(wts)
        else:
            wts = 1.0 / dd**2
            return wts / np.sum(wts)

    def check_nan_interpolate(self, coords_xy):
        line_co = self.interp_nan(coords_xy)
        return line_co

    def interpolate(self, coords_xy):
        """
        Normal interpolate. Uses IWD
        """
        line_co = self.interp(coords_xy)
        return line_co


def get_change_sign(yy, limit=15):
    _tmp = argrelmin(yy, order=5)[0]
    # find the point where things start to increase
    if len(_tmp) == 0:
        return len(yy) - 1
    elif len(_tmp) == 1:
        _n = _tmp[0]
        return _n
    else:
        # check if it is within the limit and if it is switch it to next
        # if the value increased by really small amount (1%) or
        # if the values go down in next 4 indices w.r.t value at index _n
        if _tmp[0] < limit:
            return _tmp[1]
        else:
            return _tmp[0]


def get_center_and_cutoff_index(ydata, limit=10):
    """
    Center the data and get index of changing sign
    """
    nos = np.int_(len(ydata) / 2)
    _tmp = np.where(ydata == np.max(ydata[nos - limit : nos + limit]))[0]
    nos = _tmp[len(_tmp) // 2]
    yy1 = np.flip(ydata[: nos + 1])
    yy2 = ydata[nos:]
    _idx1 = get_change_sign(yy1)
    _idx2 = get_change_sign(yy2)
    return nos, [nos - _idx1, nos + _idx2 + 1]


def get_tlines(plume, pline_xy, co_column, sat_xkm, sat_ykm, trans):
    """
    Compute transaction lines and remove background
    """
    nos = plume.line_nopoints
    # Initialize IDW interpolation class
    interp = InterpolateConcToTransactionLines(co_column.copy(), sat_xkm.copy(), sat_ykm.copy())

    nn = plume.plumeline.shape[1]
    line_pts, xdata = get_transactionline(nn, plume.line_spacing_km, plume.line_nopoints, pline_xy)

    # get interpolation values
    tlines = []
    for i in range(nn):
        _ln = DataContainer()
        # xy pre-coordinates
        setattr(_ln, "pre_coords_xy", line_pts[i, :, :])
        # deg coordinates
        lat, lon = trans.xykm2latlon(line_pts[i, :, 0], line_pts[i, :, 1])
        setattr(_ln, "pre_coords_deg", np.column_stack((lat, lon)))
        setattr(_ln, "km_from_src", (i + 1) * plume.transectspacing_km)
        setattr(_ln, "dir_vect", plume.direction_vector[:, i])
        # compute pre_co
        pre_co = interp.interpolate(_ln.pre_coords_xy)
        if np.sum(np.isnan(pre_co[nos - 10 : nos + 10])) > 10:
            continue
        else:
            ydata = interp.check_nan_interpolate(_ln.pre_coords_xy.copy())
            c_id, idx = get_center_and_cutoff_index(ydata)
            # change origin based on new data and assign new line data
            _ln.__setattr__("origin", _ln.pre_coords_deg[c_id])
            _ln.__setattr__("pre_co", pre_co)
            _ln.__setattr__("pre_co_int", ydata)
            _ln.__setattr__("co", ydata[idx[0] : idx[1]])
            _ln.__setattr__("line_dist", xdata[idx[0] : idx[1]] - xdata[c_id])
            _ln.__setattr__("coords_deg", _ln.pre_coords_deg[idx[0] : idx[1]])
            _ln.__setattr__("coords_xy", _ln.pre_coords_xy[idx[0] : idx[1]])
            # Remove background and check if the background removed conc is above zero
            remove_background(_ln)
            # append the line to transaction line
            tlines.append(_ln)
    return tlines


def emission_indices(x, y):
    """
    Get indices for data to be used for computing emissions
    """
    _ix = np.argwhere(x == 0)[0][0]
    fst = np.where(y[:_ix] < 0)[0]
    lst = np.where(y[_ix:] < 0)[0]
    if len(fst) > 0:
        i1 = fst[-1]
    else:
        i1 = 0
    if len(lst) > 0:
        i2 = lst[0] + _ix
    else:
        i2 = len(y)
    _idx = np.zeros_like(y, dtype=np.bool_)
    _idx[i1 + 1 : i2] = True
    return _idx


def remove_background(_ln, diff=0.15):
    """Compute enhancement by removing background along a transect.

    Parameters
    ----------
    _ln : type
        Transaction line `_ln`.
    diff : type
        Description of parameter `diff` (the default is 0.15).

    Returns
    -------
    None
    """
    # Get cut off indices and pad the array for background computation
    x11 = _ln.line_dist.copy() / 1000
    y11 = _ln.co.copy()
    x1, y1 = pad_arrays(x11.copy(), y11.copy(), 20)

    # set default flag to False and set true later if background removal is successful
    setattr(_ln, "flag_backgroundremovalsuccess", False)

    # IF the difference between two sides is not high then continue
    if abs(y1[0] - y1[-1]) / np.max(y1) < diff:
        _params = np.zeros((6))
        _params[0] = (y1[0] - y1[-1]) / np.max(y1)
        # fit a gaussian curve and compute background removed CO
        _params[1:], yfit1 = fit_things(x1, y1, [])
        ynew_back_removed = y11 - (_params[1] + _params[2] * x11)
        # store gaussian fit values
        setattr(_ln, "gaussfit_x", x1)
        setattr(_ln, "gaussfit_co", yfit1)
        setattr(_ln, "gaussfit_params", _params)
        setattr(_ln, "back_removed_co", ynew_back_removed)

        # Get indices of background removed conc is greater than zero
        eid = emission_indices(_ln.line_dist, _ln.back_removed_co)
        if len(_ln.back_removed_co[eid]) > 0:
            setattr(_ln, "final_co", _ln.back_removed_co[eid])
            setattr(_ln, "final_line_dist", _ln.line_dist[eid])
            setattr(_ln, "final_coords_deg", _ln.coords_deg[eid, :])
            setattr(_ln, "final_coords_xy", _ln.coords_xy[eid, :])
            setattr(_ln, "flag_backgroundremovalsuccess", True)


##################################################################
# Compute transaction lines
##################################################################


def create_tlines_remove_background(satellitedata, plumecontainer, transform):
    """_summary_

    Args:
        firecontainer (_type_): _description_
        plumecontainer (_type_): _description_
        transform (_type_): _description_
        particles (_type_): _description_
        flow (_type_): _description_
    """
    massflux = DataContainer()

    # Plume and transaction lines
    _pline = compute_medial_line(
        satellitedata.lat_nodes, satellitedata.lon_nodes, plumecontainer.plumemask, satellitedata.source, transform
    )
    massflux.__setattr__("fitted_plumeline", _pline)

    # Create line
    setattr(massflux, "transectspacing_km", 2.5)

    # extract points along plume line at plume_ds
    dataline, plumeline_xy = get_plumepoints_slope(
        massflux.fitted_plumeline, transform, ds=massflux.transectspacing_km, plume_len_km=50
    )
    # this is done as dataline is a dict and massflux is a container
    for key, value in dataline.items():
        massflux.__setattr__(key, value)

    # Transform lat-lon of satellite
    xx, yy = transform.latlon2xykm(satellitedata.lat, satellitedata.lon)
    satellitedata.__setattr__("xkm", xx)
    satellitedata.__setattr__("ykm", yy)

    # set transect attributes
    setattr(massflux, "line_spacing_km", 0.5)
    setattr(massflux, "line_nopoints", 80)

    # Define transaction lines and remove background
    tlines = get_tlines(
        massflux, plumeline_xy, satellitedata.co_column_corr, satellitedata.xkm, satellitedata.ykm, transform
    )
    massflux.__setattr__("tlines", tlines)

    # Filter plume based on background of individual lines and set a flag
    setattr(massflux, "flag_goodplume", False)
    _ed = min(20, len(tlines))
    lines_total = 0
    for _ln in tlines[:_ed]:
        lines_total += _ln.flag_backgroundremovalsuccess
    if lines_total > 5:
        setattr(massflux, "flag_goodplume", True)
    return massflux


##################################################################
# Curve fitting
##################################################################
def pad_arrays(x, y, _pad=0):
    """
    Pad x and y arrays
    """
    # pad y
    _l1 = np.ones(_pad) * y[0]
    _l2 = np.ones(_pad) * y[-1]
    _y = np.concatenate((_l1, y, _l2))
    # pad x
    _l1 = np.arange(-1 * _pad, 0) * 0.5 + x[0]
    _l2 = np.arange(1, _pad + 1) * 0.5 + x[-1]
    _x = np.concatenate((_l1, x, _l2))
    return _x, _y


def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def gauss_fit(x, y):
    mean = np.median(x)
    sigma = 4
    _p0 = [min(y), max(y), mean, sigma]
    popt, pcov = curve_fit(gauss, x, y, p0=_p0, check_finite=True, maxfev=15000)
    return popt


def gauss_linear(x, H0, H1, A, x0, sigma):
    _tmp = -((x - x0) ** 2) / (2 * sigma**2)
    _idx = _tmp > -200
    _tmp1 = np.zeros_like(_tmp)
    _tmp1[_idx] = np.exp(_tmp[_idx])
    return H0 + H1 * x + A * _tmp1


def gauss_fit_linear(x, y, consts=[]):
    _bounds = [(0, -0.001, 0, -5, -10), (np.mean(y), 0.001, np.max(y), 5, 10)]
    try:
        if len(consts) > 0:
            H0 = consts[0]
            H1 = consts[1]
            A = consts[2]
            mean = consts[3]
            sigma = consts[4]
            __p0 = [H0, H1, A, mean, sigma]
            popt, pcov = curve_fit(
                gauss_linear, x, y, p0=__p0, check_finite=True, bounds=_bounds, method="trf", maxfev=15000
            )
        else:
            H0 = min(y)
            H1 = 0.0001
            A = max(y)
            mean = 0
            sigma = 4
            __p0 = [H0, H1, A, mean, sigma]
            popt, pcov = curve_fit(
                gauss_linear, x, y, p0=__p0, check_finite=True, bounds=_bounds, method="trf", maxfev=15000
            )
        return popt
    except (RuntimeError, TypeError, NameError):
        return []


def fit_things(x1, y1, info):
    # compute background
    if len(info) == 0:
        H0, H1, A, mean, sigma = gauss_fit_linear(x1, y1)
    else:
        H0, H1, A, mean, sigma = gauss_fit_linear(x1, y1, info)
    y1_fit = gauss_linear(x1, H0, H1, A, mean, sigma)
    return [H0, H1, A, mean, sigma], y1_fit
