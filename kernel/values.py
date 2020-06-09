import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from .utils import get_dt, searchsorted
from .base import Kernel


class KernelBasisValues(Kernel):

    def __init__(self, basis_values, support, dt, coefs=None, prior=None, prior_pars=None):
        super().__init__(prior=prior, prior_pars=prior_pars)
        self.dt = dt
        self.basis_values = basis_values
        self.coefs = np.array(coefs) if coefs is not None else np.ones(self.nbasis)
        self.support = np.array(support)
        self.nbasis = basis_values.shape[1]

    def copy(self):
        kernel = KernelBasisValues(self.basis_values.copy(), self.support.copy(), self.dt, coefs=self.coefs.copy(), 
                                   prior=self.prior, prior_pars=self.prior_pars.copy())
        return kernel

#     def area(self, dt):
#         return np.sum(self.interpolate(np.arange(self.support[0], self.support[1] + dt, dt))) * dt

    def interpolate(self, t):

        t = np.atleast_1d(t)
        res = np.zeros(len(t))

#         arg0, argf = searchsorted(t, self.support, side='left')
        arg0 = int(self.support[0] / self.dt)
        argf = int(np.ceil(self.support[1] / self.dt))
        
        if arg0 >= 0 and argf <= len(t):
            res[arg0:argf] = np.matmul(self.basis_values, self.coefs)
        elif arg0 == 0 and argf > len(t):
            res = np.matmul(self.basis_values, self.coefs)[:len(t)]
        else:
            res = None
        
        return res

    def interpolate_basis(self, t):
        # kwargs = {self.key_par: self.vals_par[None, :], **self.shared_kwargs}
#         kwargs = {**{key: vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}
        return self.basis_values

#     def convolve_basis_continuous(self, t, x):
#         """# Given a 1d-array t and an nd-array x with x.shape=(len(t),...) returns X_te,
#         # the convolution matrix of each rectangular function of the base with axis 0 of x for all other axis values
#         # so that X_te.shape = (x.shape, nbasis)
#         # Discrete convolution can be achieved by using an x with 1/dt on the correct timing values
#         Assumes sorted t"""

#         dt = get_dt(t)
#         arg0, argf = searchsorted(t, self.support)
#         X = np.zeros(x.shape + (self.nbasis, ))

#         basis_shape = tuple([argf] + [1 for ii in range(x.ndim - 1)] + [self.nbasis])
#         # basis = np.zeros(basis_shape)
#         # kwargs = {self.key_par: self.vals_par[None, :], **self.shared_kwargs}
#         kwargs = {**{key: vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}
#         basis = self.fun(t[:argf, None], **kwargs).reshape(basis_shape)

#         X = fftconvolve(basis, x[..., None], axes=0)
#         X = X[:len(t), ...] * dt

#         return X

    def convolve_basis_discrete(self, t, s, shape=None):

        if type(s) is np.ndarray:
            s = (s,)

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)
        arg0, argf = searchsorted(t, self.support)
        # print(argf)

        if shape is None:
            shape = tuple([len(t)] + [max(s[dim]) + 1 for dim in range(1, len(s))] + [self.nbasis])
        else:
            shape = shape + (self.nbasis, )

        X = np.zeros(shape)

#         kwargs = {**{key: vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}

#         basis = self.fun(t[:argf, None], **kwargs).reshape(basis_shape)
        
        for ii, arg in enumerate(arg_s):
            indices = tuple([slice(arg, min(arg + argf, len(t)))] + [s[dim][ii] for dim in range(1, len(s))] + [slice(0, self.nbasis)])
#             print(indices)
#             print(indices[0])
#             print(kwnkwn)
#             print(ii, self.nbasis, self.fun(t[arg:, None] - t[arg], **kwargs).shape)
#             X[indices] += self.fun(t[arg:, None] - t[arg], **kwargs).reshape((len(t[arg:]), self.nbasis))
#             print(X.shape)
#             print(self.basis_values[:min(arg + argf, len(t)) - arg, None, :].shape)
            X[indices] += self.basis_values[:min(arg + argf, len(t)) - arg, :]

        return X
