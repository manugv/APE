#!/usr/bin/env python
# coding: utf-8
"""
compute emissions by massflux.

Created on Sept 2021
@author: Manu Goudar
"""

# Import for plume detection
import h5py
from netCDF4 import Dataset
import numpy as np
import pandas as pd
# from scipy import interpolate

from scipy.interpolate import NearestNDInterpolator as n_int
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.signal import argrelmin

from shapely.geometry import LineString
from skimage.morphology import skeletonize

from .ModuleRefineGrids import RefineGridsUniform


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
    #
    ll = np.zeros_like(_line)
    ll[:, 0], ll[:, 1] = trans.latlon2xykm(_line[:, 0], _line[:, 1])
    # Decide on number of points
    dist = np.sqrt(np.power(ll[:, 0], 2) + np.power(ll[:, 1], 2))
    # number of lines are defined based on min between possible and max
    n = np.int_(np.minimum(np.floor_divide(dist.max(), ds), _min_pts))
    # Create Linestring
    _lxy = LineString(tuple(map(tuple, ll)))
    # Get values at given ds
    splitter = [_lxy.interpolate(i * ds) for i in range(1, n + 1)]
    pts = np.zeros((2, n))
    for ii in range(n):
        pts[0, ii], pts[1, ii] = trans.xykm2latlon(splitter[ii].x, splitter[ii].y)
    # Get values to compute slope
    splitter1 = [_lxy.interpolate(i * ds + 0.1) for i in range(1, n + 1)]
    # compute directional vector and normalize it
    vec_xy = np.zeros((2, n))
    for ii in range(n):
        vec_xy[0, ii] = splitter1[ii].x - splitter[ii].x
        vec_xy[1, ii] = splitter1[ii].y - splitter[ii].y
    vec_norm = vec_xy / np.linalg.norm(vec_xy, axis=0)

    # compute slope
    slope = np.zeros((n))
    for ii in range(n):
        if (splitter1[ii].x - splitter[ii].x) == 0:
            slope[ii] = 10000
        else:
            slope[ii] = (splitter1[ii].y - splitter[ii].y) / (splitter1[ii].x - splitter[ii].x)

    line_dist = (np.arange(n) + 1) * 2.5
    _dataline = {
        "plumeline": pts,
        "plumeslope": slope,
        "dist_from_src": line_dist,
        "direction_vector": vec_norm,
        "spacing": ds,
    }
    return _dataline


##################################################################
# Define Transaction line
##################################################################
def get_co_interpolation(lat, lon, co):
    # Interpolate function for CO
    return n_int((lat.ravel(), lon.ravel()), co.ravel())


class TransactionLine:
    """
    Creates a transaction line perpendicular to plume line
    Parameters
    -----------
    origin : 1d array of size 2
    dist_from_src : Distance in km from source
    slope : Slope 'a' of the line (ax + b)
    spacing : spacing between two points on transaction line
              In mts
    nos : Number of points on one side of the transaction line.
          Total number of points will be 2*nos + 1
    trans : Trans is class that transforms lat-lon to xy and vice-versa
    a_vect : This is directional vector that is used to align velocity

    """

    def __init__(self, origin, dist_from_src, slope, spacing, nos, trans, a_vect):
        self.dist_from_src = dist_from_src
        self.pre_origin = origin
        self.coeffs = self.__get_coeffs(slope)
        self.ds = spacing
        self.nos = nos
        self.trans = trans
        self.dir_vect = a_vect
        # variables that are stored during simulations
        self.pre_coords_deg = np.zeros((self.nos * 2 + 1, 2))
        self.pre_coords_xy = np.zeros((self.nos * 2 + 1, 2))
        # Create a polyfit and a polynomial equation
        self.poly = np.poly1d(self.coeffs)
        self.get_line()

    def __get_coeffs(self, slope):
        # define b and return coefficients for creating line
        b = self.pre_origin[0] - slope * self.pre_origin[1]
        return np.array([slope, b])

    def get_line(self, spacing=0.5):
        """
        Create transaction line at given point from the coefficients
        spacing : This is given in degrees, the extent of line that
                  is possible in degrees
        """
        # Set mid point (Origin) and convert it to its corresponding xy
        self.pre_coords_deg[self.nos, :] = self.pre_origin
        _xy = self.trans.latlon2xymts(self.pre_origin[0], self.pre_origin[1])
        self.pre_coords_xy[self.nos, :] = _xy
        # Right side of line
        st_ln = self.pre_origin[1] - spacing
        st_lt = self.poly(st_ln)
        pl_lat = np.array([self.pre_origin[0], st_lt])
        pl_lon = np.array([self.pre_origin[1], st_ln])
        _pts_tmp, _deg_tmp = self.get_pts(pl_lat, pl_lon)
        self.pre_coords_deg[: self.nos, 0] = np.flip(_deg_tmp[:, 0])
        self.pre_coords_deg[: self.nos, 1] = np.flip(_deg_tmp[:, 1])
        self.pre_coords_xy[: self.nos, 0] = np.flip(_pts_tmp[:, 0])
        self.pre_coords_xy[: self.nos, 1] = np.flip(_pts_tmp[:, 1])
        # left side of line
        en_ln = self.pre_origin[1] + spacing
        en_lt = self.poly(en_ln)
        pl_lat = np.array([self.pre_origin[0], en_lt])
        pl_lon = np.array([self.pre_origin[1], en_ln])
        __xy, __latlon = self.get_pts(pl_lat, pl_lon)
        self.pre_coords_xy[self.nos + 1:] = __xy
        self.pre_coords_deg[self.nos + 1:] = __latlon
        # convert mts to kms
        self.pre_coords_xy /= 1000

    def get_pts(self, pl_lat, pl_lon):
        """
        Split the line into number of points (nos) based on two points
        Input:
        ------
        pl_lat : array of two points in deg
        pl_lon : array of two points in deg

        Returns:
        --------
        _pts : Array of size (nos,2) in mts
        _deg : Array of size (nos,2) in deg
        """
        x_mt, y_mt = self.trans.latlon2xymts(pl_lat, pl_lon)
        # Create Linestring
        _lxy = LineString([(x_mt[0], y_mt[0]), (x_mt[1], y_mt[1])])
        # Get values at given ds
        _split = [_lxy.interpolate(i * self.ds) for i in range(1, self.nos + 1)]
        _pts = np.zeros((self.nos, 2))
        _deg = np.zeros((self.nos, 2))
        for ii in range(self.nos):
            _pts[ii, 0] = _split[ii].x
            _pts[ii, 1] = _split[ii].y
            _deg[ii, 0], _deg[ii, 1] = self.trans.xymts2latlon(_split[ii].x, _split[ii].y)
        return _pts, _deg


class InterpolateTransactionLines:
    def __init__(self, co_edit, x_km, y_km):
        self.co_edit = co_edit
        self.x_km = x_km
        self.y_km = y_km
        self.interp = LinearNDInterpolator(
            np.column_stack((self.x_km.ravel(), self.y_km.ravel())), self.co_edit.ravel()
        )
        id_nan = ~np.isnan(co_edit)
        self.interp_nan = LinearNDInterpolator(
            np.column_stack((self.x_km[id_nan].ravel(), self.y_km[id_nan].ravel())),
            self.co_edit[id_nan].ravel(),
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


def get_tlines(plume, co_column, sat_xkm, sat_ykm, trans, nos, spacing=500):
    """
    Compute transaction lines and remove background
    """
    # Initialize IDW interpolation class
    interp = InterpolateTransactionLines(co_column.copy(), sat_xkm.copy(), sat_ykm.copy())

    xdata = np.arange(-nos, nos + 1) * spacing
    # get interpolation values
    tlines = []
    for i in range(len(plume.plumeslope)):
        if plume.plumeslope[i] == 0:
            _plm = 0.0000001
        else:
            _plm = plume.plumeslope[i]
        _ln = TransactionLine(
            plume.plumeline[:, i],
            plume.dist_from_src[i],
            -1 / _plm,
            spacing,
            nos,
            trans,
            plume.direction_vector[:, i],
        )
        # compute pre_co
        pre_co = interp.interpolate(_ln.pre_coords_xy)
        if np.sum(np.isnan(pre_co[nos - 10: nos + 10])) > 10:
            continue
        else:
            ydata = interp.check_nan_interpolate(_ln.pre_coords_xy.copy())
            c_id, idx = get_center_and_cutoff_index(ydata)
            # change origin based on new data and assign new line data
            _ln.__setattr__("origin", _ln.pre_coords_deg[c_id])
            _ln.__setattr__("pre_co", pre_co)
            _ln.__setattr__("pre_co_int", ydata)
            _ln.__setattr__("co", ydata[idx[0]: idx[1]])
            _ln.__setattr__("line_dist", xdata[idx[0]: idx[1]] - xdata[c_id])
            _ln.__setattr__("coords_deg", _ln.pre_coords_deg[idx[0]: idx[1]])
            _ln.__setattr__("coords_xy", _ln.pre_coords_xy[idx[0]: idx[1]])
            # Remove background and create new variables
            remove_background(_ln)
            if _ln.f_background_good:
                eid = emission_indices(_ln.line_dist, _ln.back_removed_co)
                if len(_ln.back_removed_co[eid]) > 0:
                    _ln.__setattr__("final_co", _ln.back_removed_co[eid])
                    _ln.__setattr__("final_line_dist", _ln.line_dist[eid])
                    _ln.__setattr__("final_coords_deg", _ln.coords_deg[eid, :])
                    _ln.__setattr__("final_coords_xy", _ln.coords_xy[eid, :])
                else:
                    _ln.f_background_good = False
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
    _idx[i1 + 1: i2] = True
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

    # IF the difference between two sides is not high then continue
    if abs(y1[0] - y1[-1]) / np.max(y1) < diff:
        _ln.__setattr__("f_background_good", True)
        _params = np.zeros((6))
        _params[0] = (y1[0] - y1[-1]) / np.max(y1)
        # fit a gaussian curve and compute background removed CO
        _params[1:], yfit1 = fit_things(x1, y1, [])
        _ln.__setattr__("gaussfit_x", x1)
        _ln.__setattr__("gaussfit_co", yfit1)
        ynew_back_removed = y11 - (_params[1] + _params[2] * x11)
        _ln.__setattr__("gaussfit_params", _params)
        _ln.__setattr__("back_removed_co", ynew_back_removed)
    else:
        _ln.__setattr__("f_background_good", False)


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
                gauss_linear,
                x,
                y,
                p0=__p0,
                check_finite=True,
                bounds=_bounds,
                method="trf",
                maxfev=15000,
            )
        else:
            H0 = min(y)
            H1 = 0.0001
            A = max(y)
            mean = 0
            sigma = 4
            __p0 = [H0, H1, A, mean, sigma]
            popt, pcov = curve_fit(
                gauss_linear,
                x,
                y,
                p0=__p0,
                check_finite=True,
                bounds=_bounds,
                method="trf",
                maxfev=15000,
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


####################################################################
#    PARTICLES RELATED
####################################################################
def create_topology_interpolate(flname):
    # Read topology
    ff = Dataset(flname, "r")
    ll = ff["z"][0]
    geop = ll.data / 9.80665
    lon = ff["longitude"][:]
    if lon.max() > 180:
        flon = lon - 360
    else:
        flon = lon
    lat = ff["latitude"][:]
    ff.close()
    flat = np.flip(lat)
    surface = np.flip(geop, axis=0)
    return RGI((flat, flon), surface, bounds_error=False)


def get_endloc_particles(filename, topo):
    f = h5py.File(filename, "r")
    tt = len(f["Particles"].keys()) - 1
    fl = "Step" + str(tt)
    _df = pd.DataFrame()
    pos = f["Particles"][fl]["position"][:]
    _df["lat"] = pos[:, 0]
    _df["lon"] = pos[:, 1]
    _df["height"] = pos[:, 2] - topo((pos[:, 0], pos[:, 1]))
    _df["vert_id"] = f["Particles"][fl]["VerticalIds"][:]
    _df["source_id"] = f["Particles"][fl]["SourceIds"][:]
    _df["global_id"] = f["Particles"][fl]["GlobalIds"][:]
    f.close()
    nn = len(_df.vert_id.unique()) - 1
    dx = 1 / (nn // 2)
    wts = np.zeros_like(_df.vert_id.values, np.float32)
    for i in range(nn // 2):
        wts[_df.vert_id.values == i] = i * 0.1 + dx
        wts[_df.vert_id.values == nn - i] = i * 0.1 + dx
    wts[_df.vert_id.values == nn / 2] = 1
    _df["wts"] = wts
    return pos, _df


def particles_height_at_tlines(coords, ff, weight=True):
    """
    Finds all particles around given coordinates.

    Parameters
    ----------
    coords : Float array type
        Description of parameter `coords of a transaction line`.
    ff : Pandas dataframe
        Description of parameter `ff`.
    weight : bool
        Description of parameter should use weights or not. Default is False.

    Returns
    -------
    type
        height of the transaction line.

    """

    # for each line find the center and remove > 1 particles
    ll = coords[coords.shape[0] // 2]
    f2 = ff[np.sqrt((ff.lat.values - ll[0]) ** 2 + (ff.lon.values - ll[1]) ** 2) < 1]
    data_line = pd.DataFrame()
    for ll in coords:
        _tmpdata = f2[np.sqrt((f2.lat.values - ll[0]) ** 2 + (f2.lon.values - ll[1]) ** 2) < 0.0025]
        data_line = pd.concat((data_line, _tmpdata), axis=0, ignore_index=True)
    if weight:
        return (
            data_line.height.values,
            (data_line.height * data_line.wts).sum() / data_line.wts.sum(),
        )
    else:
        if data_line.empty:
            return 0, np.nan
        else:
            return data_line.height.values, np.mean(data_line.height.values)


def get_weighted_height(f3, weight=True):
    """_summary_

    Args:
        f3 (_type_): _description_
        weight (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if weight:
        # compute weighted average
        nn = len(f3.vert_id.unique()) - 1
        wts = np.zeros_like(f3.vert_id.values, np.float32)
        for i in range(nn // 2):
            wts[f3.vert_id.values == i] = i * 0.1 + 0.05
            wts[f3.vert_id.values == nn - i] = i * 0.1 + 0.05
        wts[f3.vert_id.values == nn / 2] = 1
        return (f3.height.values * wts).sum() / wts.sum()
    else:
        return np.mean(f3.height.values)
