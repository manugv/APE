# cython: language_level=3, boundscheck=False
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.cdivision(True)
cpdef get_velocity(double[:,:] pos, double[:] lat, double[:] lon, double[:,:,:] z, double[:,:,:] u, double[:,:,:] v, double[:,:,:] w):
    """
    get velocity
    Position is in [lat, lon, z]
    u,v,w,z is in [lat, lon, z]

    """
    # Define empty variables
    cdef int nos = pos.shape[0]
    cdef int zlen = z.shape[2]
    cdef double[2] xd, yd
    cdef double[2] zd = np.zeros((2), np.float64)
    cdef int[2] zid = np.zeros((2), np.int_)
    cdef double[:, :, :] _u = np.zeros((3, 2, 2), dtype=np.float64)
    cdef double[:, :] velo = np.zeros((nos, 3), dtype=np.float64)
    lmt = np.zeros((nos), dtype=np.bool_)
    _ijt = np.zeros((4, 2), dtype=np.int32)
    # define limits
    _ijt[1, 1] = 1
    _ijt[2, 0] = 1
    _ijt[3, :] = 1
    cdef int[:, :] _ij = _ijt
    cdef double low_lat = lat[0]
    cdef double up_lat = lat[-1]
    cdef double dlat_i = 1./(lat[1] - lat[0])
    cdef double low_lon = lon[0]
    cdef double up_lon = lon[-1]
    cdef double dlon_i = 1./(lon[1] - lon[0])
    cdef int kx, ky, i, j, ii, k, info1
    cdef bint info

    for p in range(nos):
        # Check if the given point is inside the domain
        if (pos[p, 0] < low_lat or pos[p, 0] > up_lat or
                pos[p, 1] < low_lon or pos[p, 1] > up_lon):
            velo[p, :] = 0
            lmt[p] = False
            continue
        else:
            # Get 2 indices of regular grid xy (get lower index)
            kx = int((pos[p, 0]-low_lat)*dlat_i)
            ky = int((pos[p, 1]-low_lon)*dlon_i)

            # compute difference ratio between given pt and
            # coordinates of vortices
            xd[1] = (pos[p, 0] - lat[kx])/(lat[kx+1] - lat[kx])
            xd[0] = 1 - xd[1]
            yd[1] = (pos[p, 1] - lon[ky])/(lon[ky+1] - lon[ky])
            yd[0] = 1 - yd[1]
            # identify the ratio for linear interpolation in z
            info1 = 0

            for ii in range(4):
                i, j = _ij[ii, :]
                # check if point lies inside the z domain and get ids
                # identify the ratio for linear interpolation in z
                if ((pos[p, 2] <= z[kx+i, ky+j, -1]) & (pos[p, 2] >= z[kx+i, ky+j, 0])):
                    for k in range(1, zlen):
                        if pos[p, 2] < z[kx+i, ky+j, k]:
                            zid[0] = k - 1
                            zid[1] = k
                            zd[0] = ((pos[p, 2] - z[kx+i, ky+j, zid[0]]) /
                                     (z[kx+i, ky+j, zid[1]] - z[kx+i, ky+j, zid[0]]))
                            zd[1] = 1 - zd[0]
                            break

                    # interpolate velocity: get velocity values
                    _u[0, i, j] = (u[kx+i, ky+j, zid[1]]*zd[0] +
                                   u[kx+i, ky+j, zid[0]]*zd[1])
                    _u[1, i, j] = (v[kx+i, ky+j, zid[1]]*zd[0] +
                                   v[kx+i, ky+j, zid[0]]*zd[1])
                    _u[2, i, j] = (w[kx+i, ky+j, zid[1]]*zd[0] +
                                   w[kx+i, ky+j, zid[0]]*zd[1])
                else:
                    info1 += 1
                    _u[:, i, j] = 0.0

            if info1 == 4:
                lmt[p] = False
                velo[p, :] = 0
            else:
                for k in range(3):
                    # interpolate to y
                    velo[p, k] = ((_u[k, 0, 0]*yd[0] + _u[k, 0, 1]*yd[1])*xd[0]
                                  + (_u[k, 1, 0]*yd[0] + _u[k, 1, 1]*yd[1])*xd[1])
                    lmt[p] = True
    return velo, lmt
