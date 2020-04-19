import logging

import matplotlib.pyplot as plt
import numpy as np

import regpy as rp
import regpy.operators.mri as rpm
from regpy.solvers.irgnm import IrgnmCG

import cfl
from regpy_bart import BartNoir

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
# )


def plot_nl(x, d, ni):
    nr = int(np.trunc(np.sqrt(ni)))
    f, ax = plt.subplots(nrows=nr, ncols=ni//nr, constrained_layout=True)
    im,c = d.split(x)
    print(ax.shape)
    for i, ax in enumerate(ax.flat):
        if i == 0:
            ax.imshow(np.abs(im))
            ax.set_title("Image")
        else:
            ax.imshow(np.abs(c[i-1,...]))
            ax.set_title("Coil {!s}".format(i-1))

def plot_ci(x):
    nr = int(np.trunc(np.sqrt(nc)))
    f, ax = plt.subplots(nrows=nr, ncols=nc//nr, constrained_layout=True)
    for i, ax in enumerate(ax.flat):
        ax.imshow(np.abs(x[i,...]))
        ax.set_title("Coil {!s}".format(i))


#%% Read and preprocess data

sobolev_index = 32
a = 220
noiselevel = None

datafile = 'data/unders_2_v8'

exact_data_b = cfl.readcfl(datafile)
exact_data = np.ascontiguousarray(np.transpose(exact_data_b)).squeeze()

bart_reference = cfl.readcfl(datafile + '_bartref')



X = exact_data_b.shape[1]
ncoils = exact_data_b.shape[3]

grid = rp.discrs.UniformGrid((-1, 1, X), (-1, 1, X), dtype=np.complex64)

pattern = rpm.estimate_sampling_pattern(exact_data)

#%% Reconstruction

bartop = BartNoir(grid, ncoils, pattern)

setting = rp.solvers.HilbertSpaceSetting(op=bartop, Hdomain=rp.hilbert.L2, Hcodomain=rp.hilbert.L2)

exact_data_itreg = exact_data[:, pattern].flatten()
exact_data_itreg = exact_data_itreg / setting.Hcodomain.norm(exact_data_itreg) * 100

if noiselevel is not None:
    data = (exact_data_itreg + noiselevel * bartop.codomain.randn(dtype=complex)).astype(np.complex64)
else:
    data = exact_data_itreg

init = bartop.domain.zeros()
init_density, init_coils = bartop.domain.split(init)
init_density[...] = 1.
init_coils[...] = 0.

solver = IrgnmCG(
    setting, data, init=init,
    regpar=1, regpar_step=1/2, cgstop=5
)

stoprule = (
    rp.stoprules.CountIterations(max_iterations=11) +
    rp.stoprules.Discrepancy(
        setting.Hcodomain.norm, data,
        noiselevel=setting.Hcodomain.norm(exact_data_itreg - data),
        tau=0.5
    )
)

# Plotting setup
plt.ion()
#fig, axes = plt.subplots(ncols=3, constrained_layout=True)
fig, axes = plt.subplots(ncols=3)
# bars = [mp.colorbar.make_axes(ax)[0] for ax in axes]

axes[0].set_title('reference solution')
axes[1].set_title('reconstruction')
axes[2].set_title('difference x10')

# Plot exact solution
ref = axes[0].imshow(np.fliplr(np.abs(bart_reference.transpose().squeeze()).transpose()), origin='lower')
ref.set_clim((0, ref.get_clim()[1]))

# Run the solver, plot iterates
for reco, reco_data in solver.until(stoprule):
    reco_postproc = rpm.normalize(*bartop.domain.split(bartop._forward_coils(reco)))
    im = axes[1].imshow(np.fliplr(np.abs(reco_postproc).transpose()), origin='lower')
    im.set_clim(ref.get_clim())
    diff = axes[2].imshow(np.fliplr((np.abs(bart_reference.transpose().squeeze()) - np.abs(reco_postproc)).transpose()), origin='lower')
    diff.set_clim((0, ref.get_clim()[1]/10.))
    plt.pause(0.5)

plt.ioff()
plt.show()
