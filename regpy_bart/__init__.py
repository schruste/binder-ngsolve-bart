import ctypes

import numpy as np
import regpy as rp

bartso = ctypes.CDLL('bart.so');


class bart_iovec(ctypes.Structure):
    _fields_ = [
        ("N", ctypes.c_uint),
        ("dims", ctypes.POINTER(ctypes.c_long)),
        ("strs", ctypes.POINTER(ctypes.c_long)),
        ("size", ctypes.c_size_t)
    ]


bartso.nlop_apply.restype = None;
bartso.nlop_derivative.restype = None;
bartso.nlop_adjoint.restype = None;
bartso.nlop_free.restype = None;
bartso.nlop_codomain.restype = ctypes.POINTER(bart_iovec)
bartso.nlop_domain.restype = ctypes.POINTER(bart_iovec)


class nlop:
    class bart_nlop(ctypes.Structure):
        pass

    def __init__(self, cnlop: bart_nlop):
        self.cnlop = cnlop
        self.codomain = bartso.nlop_codomain(cnlop).contents
        self.domain = bartso.nlop_domain(cnlop).contents
        self.oshape = ctypes.cast(self.codomain.dims, ctypes.POINTER(self.codomain.N * ctypes.c_long)).contents
        self.ishape = ctypes.cast(self.domain.dims, ctypes.POINTER(self.domain.N * ctypes.c_long)).contents

    def __del__(self):
        bartso.nlop_free(self.cnlop)

    def apply(self, src: np.array):
        dst = np.asfortranarray(np.empty(self.oshape, dtype=np.complex64))
        assert self.codomain.size == dst.itemsize
        src = src.astype(np.complex64, copy=False)
        bartso.nlop_apply(self.cnlop,
            dst.ndim, dst.ctypes.shape_as(ctypes.c_long), dst.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)),
            src.ndim, src.ctypes.shape_as(ctypes.c_long), src.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)))
        return dst


    def derivative(self, src: np.array):
        dst = np.asfortranarray(np.empty(self.oshape, dtype=np.complex64))
        assert self.codomain.size == dst.itemsize
        src = src.astype(np.complex64, copy=False)
        bartso.nlop_derivative(self.cnlop,
            dst.ndim, dst.ctypes.shape_as(ctypes.c_long), dst.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)),
            src.ndim, src.ctypes.shape_as(ctypes.c_long), src.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)))
        return dst

    def adjoint(self, src: np.array):
        dst = np.asfortranarray(np.empty(self.ishape, dtype=np.complex64))
        assert self.domain.size == dst.itemsize
        src = src.astype(np.complex64, copy=False)
        bartso.nlop_adjoint(self.cnlop,
            dst.ndim, dst.ctypes.shape_as(ctypes.c_long), dst.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)),
            src.ndim, src.ctypes.shape_as(ctypes.c_long), src.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)))
        return dst



class noir_model_conf_s(ctypes.Structure):
    pass


class bart_linop(ctypes.Structure):
    pass


class bart_noir_op_s(ctypes.Structure):
    pass


class noir_s(ctypes.Structure):
    _fields_ = [
        ("nlop", ctypes.POINTER(nlop.bart_nlop)),
        ("linop", ctypes.POINTER(bart_linop)),
        ("noir_op", ctypes.POINTER(bart_noir_op_s))
    ]


class BartNoir(rp.operators.Operator):
    """Operator that implements the multiplication between density and coil profiles. The domain
    is a direct sum of the `grid` (for the densitiy) and a `regpy.discrs.UniformGrid` of `ncoils`
    copies of `grid`, stacked along the 0th dimension.

    Parameters
    ----------
    grid : regpy.discrs.UniformGrid
        The grid on which the density is defined.
    ncoils : int
        The number of coils.
    """

    def __init__(self, grid, ncoils, psf):
        assert isinstance(grid, rp.discrs.UniformGrid)
        assert grid.ndim == 2
        self.grid = grid
        """The density grid."""
        self.coilgrid = rp.discrs.UniformGrid(ncoils, *grid.axes, dtype=grid.dtype)
        """The coil grid, a stack of copies of `grid`."""
        self.ncoils = ncoils
        """The number of coils."""
        super().__init__(
            domain = self.grid + self.coilgrid,
            codomain = rp.discrs.Discretization(np.count_nonzero(psf)*ncoils, dtype=self.grid.dtype)
        )

        sd = list(self.grid.shape)
        sd += [1] * (3 - len(sd))

        shaped = [1]*16
        shaped[:3] = sd

        shapec = shaped.copy()
        shapec[3] = ncoils

        self.dimsd = (ctypes.c_long * len(shaped))( *shaped)
        self.dimsc = (ctypes.c_long * len(shaped))( *shapec)

        self.psff = psf.astype(np.complex64)

        psfnz = np.zeros(self.coilgrid.shape, dtype=bool)
        for c in range(ncoils):
            psfnz[c,:,:] = psf

        self.psfnz = psfnz.flatten()

        bartso.noir_create.restype = noir_s;
        self.nconf = noir_model_conf_s.in_dll(bartso, "noir_model_conf_defaults");
        self.ns = bartso.noir_create(
            self.dimsc, None,
            self.psff.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)),
            ctypes.pointer(self.nconf)
        )
        self.nl = nlop(self.ns.nlop)

    def _bartpreproc(self, x):
        density, coils = self.domain.split(x)
        d = np.asfortranarray(np.reshape(density.transpose(), self.dimsd))
        c = np.asfortranarray(np.reshape(coils.transpose(), self.dimsc))
        x_b = np.asfortranarray(np.concatenate((d, c), 3))
        return x_b

    def _bartpostproc(self, d_b):
        d = np.transpose(d_b).flatten()
        return d[self.psfnz]

    def _eval(self, x, differentiate=False):
        density, coils = self.domain.split(x)

        x_b = self._bartpreproc(x)
        dst = self.nl.apply(x_b)

        return self._bartpostproc(dst)

    def _derivative(self, x):
        x2 = self._bartpreproc(x)
        dst = self.nl.derivative(x2)
        dst2 = self._bartpostproc(dst)
        return dst2

    def _adjoint(self, y):
        y_tmp = self.coilgrid.zeros().flatten()
        y_tmp[self.psfnz] = y.flatten()
        y_b = np.asfortranarray(y_tmp.transpose()).reshape(self.dimsc)

        x = self.nl.adjoint(y_b)
        d = x[:,:,:,0:1,...]
        c = x[:,:,:,1:,...]

        dst = self.domain.zeros()
        dd, dc = self.domain.split(dst)
        dd[...] = np.ascontiguousarray(d.transpose().squeeze())
        dc[...] = np.ascontiguousarray(c.transpose().squeeze())

        return dst

    def _forward_coils(self, x):
        xc = x.copy()
        density, coils = self.domain.split(xc)

        c = np.asfortranarray(
            np.reshape(np.transpose(coils), self.dimsc)
        ).astype(np.complex64, copy=False)

        bartso.noir_forw_coils.restype = None;
        bartso.noir_forw_coils(
            self.ns.linop, c.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(2 * ctypes.c_float))
        )
        coils[...] = np.ascontiguousarray(c.transpose().squeeze())

        return xc

    def __repr__(self):
        return rp.util.make_repr(self, self.grid, self.ncoils)
