import xarray as xr
import numpy as np
from scipy.interpolate import RectSphereBivariateSpline
from pathlib import Path
from netCDF4 import Dataset, num2date


class VelocityInterpolation:
    """
    Interpolations is set up such that the longitude goes from -180 to 180
    """
    def __init__(self, dirprefix, height=100):
        """ Initialize the class.

        Parameters
        ----------
        dirprefix : string
            Directory of the data
        height : float
            plume height.
        """
        self.height = height
        if self.height == 10.0:
            self.u_name = "u10"
            self.v_name = "v10"
        elif self.height == 100.0:
            self.u_name = "u100"
            self.v_name = "v100"
        else:
            self.u_name= "u"
            self.v_name= "v"
        self.dirprefix = dirprefix
        self.deg_to_rad = np.pi/180

    def computefunction(self, measuretime):
        if (self.height == 10.0) or (self.height == 100.0):
            u, v, lat, lon = self._computefunction1(measuretime)
        else:
            u, v, lat, lon = self._computefunction2(measuretime)
        self._create_interpolationfunctions(u, v, lat, lon)
            
    def _computefunction1(self, measuretime):
        self.measuretime = measuretime
        # filename
        filename = (self.dirprefix + self.measuretime.strftime("%Y") + "/sl_"
                    + self.measuretime.strftime("%Y%m%d") + ".grib")
        file = Path(filename)
        if file.is_file():
            # read grib data
            grbs = xr.open_dataset(filename, engine="cfgrib", backend_kwargs={"indexpath": ""})
        else:
            print("Velocity field file does not exist")
        # get velocity and interpolate at time t
        uvel = grbs[self.u_name].interp(time=self.measuretime)
        vvel = grbs[self.v_name].interp(time=self.measuretime)
        return uvel.data, vvel.data, uvel.latitude, uvel.longitude

    def _computefunction2(self, measuretime):
        # step 1: Check if the file exists
        self.measuretime = measuretime
        # filename
        filename = (self.dirprefix + self.measuretime.strftime("%Y%m%d_%H%M") + ".nc")
        file = Path(filename)
        if file.is_file():   # check if file exists
            # read grib data
            hf = Dataset(filename, "r")
        else:
            print("Velocity field file does not exist")
            exit()
        # step 2: Check if time exists
        dtimes = num2date(hf["time"], hf["time"].units)
        if (self.measuretime >= dtimes[0]) and (self.measuretime <= dtimes[-1]):
            index = np.searchsorted(dtimes, self.measuretime)
        else:
            index = -999
            print("time not available")
            exit()
        
        # read data based on time
        lon = hf["longitude"][:].data
        lat = hf["latitude"][:].data
        _u = np.flip(hf["u"][index-1:index+1].data, axis=1)  # height is axis 1 and time is axis 0
        _v = np.flip(hf["v"][index-1:index+1].data, axis=1)
        z = np.flip(hf["z"][index-1:index+1].data, axis=1)
        hf.close()   # close the file

        # step 3: Convert z to altitude. This corresponds to above sea level
        Re = 6371000 # in m
        g = 9.80665  # in m/s^2
        alt = Re*(z/g)/(Re-(z/g))
        # first level is height from the sea level.
        # Subtracting it will get us height above ground
        for i in range(alt.shape[0]):
            alt[i] = alt[i] - alt[i,0]
        
        # step 4: For each time step get the velocity at the desired height
        u_vel_0 = interpolatedataatheight(alt[0], self.height, _u[0])
        v_vel_0 = interpolatedataatheight(alt[0], self.height, _v[0])
        u_vel_1 = interpolatedataatheight(alt[1], self.height, _u[1])
        v_vel_1 = interpolatedataatheight(alt[1], self.height, _v[1])        
        
        # step 5: Interpolate in time
        dz = ((dtimes[index] - self.measuretime).seconds)/(60*60)
        u_vel = linear_1d_interpolation(u_vel_0, u_vel_1, dz)
        v_vel = linear_1d_interpolation(v_vel_0, v_vel_1, dz)
        return u_vel, v_vel, lat, lon
    
    def _create_interpolationfunctions(self, uvel, vvel, lat, lon):
        self.lat_deg = np.flip(lat)
        self.lat_rad = (self.lat_deg + 90)*self.deg_to_rad
        self.lon_deg = lon
        self.lon_rad = self.lon_deg*self.deg_to_rad

        self.u_vel = np.flip(uvel, axis=0)
        self.v_vel = np.flip(vvel, axis=0)
        self.fu = RectSphereBivariateSpline(self.lat_rad[1:-1], self.lon_rad, self.u_vel[1:-1,:])
        self.fv = RectSphereBivariateSpline(self.lat_rad[1:-1], self.lon_rad, self.v_vel[1:-1,:])

    def latitude_in_radians(self, _lt):
        """Convert latitude to radians.

        Convert from degree to rad

        Parameters
        ----------
        _lt : Array
            Array of latitudes

        """
        return (_lt+90)*self.deg_to_rad

    def longitude_in_radians(self, _ln):
        """
        Data is from -180 to 180
        """
        if _ln < 0:
            _ln = _ln + 360
        return _ln*self.deg_to_rad

    def interpolate(self, lat, lon):
        """
        Interpolation for scalar, 1d array and 2d array
        Data should be given as:
        latitude: -90 to 90
        longitude: -180 to 180
        """
        if np.isscalar(lat):
            _lt_rd = self.latitude_in_radians(lat)
            _ln_rd = self.longitude_in_radians(lon)
            return self.fu.ev(_lt_rd, _ln_rd), self.fv.ev(_lt_rd, _ln_rd)
        # one dimensional array
        elif lat.ndim == 1:
            # Get longitude individually
            _ln_rd = np.zeros_like(lat)
            for i in range(lat.shape[0]):
                _ln_rd[i] = self.longitude_in_radians(lon[i])
            _lt_rd = self.latitude_in_radians(lat)
            return self.fu.ev(_lt_rd, _ln_rd), self.fv.ev(_lt_rd, _ln_rd)
        # two dimensional array
        elif lat.ndim == 2:
            # Get longitude individually
            _ln_rd = np.zeros_like(lat)
            sh = lat.shape
            for i in range(sh[0]):
                for j in range(sh[1]):
                    _ln_rd[i, j] = self.longitude_in_radians(lon[i, j])
            _lt_rd = self.latitude_in_radians(lat)
            u = self.fu.ev(_lt_rd.ravel(), _ln_rd.ravel()).reshape(sh)
            v = self.fv.ev(_lt_rd.ravel(), _ln_rd.ravel()).reshape(sh)
            return u, v
        else:
            print("something went wrong")


def interpolatedataatheight(alt, height, u):
    uint = np.zeros_like((alt[0,:,:]))
    for i in range(np.shape(alt)[1]):
        for j in range(np.shape(alt)[2]):
            # check bounds
            if (height < alt[0,i,j]) or (height > alt[-1,i,j]):
                uint[i,j] = np.nan
            else:
                index = np.searchsorted(alt[:,i,j], height)
                if index == 0:  # if height = alt at zero index grid
                    uint[i,j] = u[index, i, j]
                else:
                    dz = (alt[index,i,j] - height) / (alt[index,i,j] - alt[index-1,i,j])
                    uint[i,j] = linear_1d_interpolation(u[index-1,i,j], u[index,i,j], dz)
    return uint


def linear_1d_interpolation(u1, u2, dz):
    interpolation = dz*u1 + (1-dz)*u2
    return interpolation
