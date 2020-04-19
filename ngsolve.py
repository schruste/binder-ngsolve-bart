import logging

import ngsolve as ngs
import ngsolve.meshes
import numpy as np
import pyvista as pv

from regpy.discrs.ngsolve import NgsSpace
from regpy.hilbert import L2, Sobolev
from regpy.operators.ngsolve import Coefficient
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm import IrgnmCG
import regpy.stoprules as rules


# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
# )

def plotmeshes(dom, **kwargs):
    names = list(kwargs.keys())
    coefs = list(kwargs.values())
    ngs.VTKOutput(
        dom.fes.mesh, coefs=coefs, names=names, subdivision=2,
        filename="plotmeshes"
    ).Do()
    mesh = pv.read("plotmeshes.vtk")
    panels = {}
    for n in names:
        p = pv.Plotter()
        p.add_mesh(mesh.warp_by_scalar(n), scalars=n)
        panels[n] = p.show(use_panel=True)
    return panels

noiselevel = 0.0001

domain = NgsSpace(ngs.L2(ngs.meshes.MakeQuadMesh(nx=10, ny=10), order=2))
codomain = NgsSpace(ngs.H1(ngs.meshes.MakeQuadMesh(nx=10, ny=10), order=3, dirichlet="left|top|right|bottom"))
noise_domain = NgsSpace(ngs.L2(codomain.fes.mesh, order=1))
noise = codomain.from_ngs(noise_domain.to_ngs(noiselevel * np.random.randn(noise_domain.fes.ndof)))

cfu_exact_solution = ngs.x + 1
exact_solution = domain.from_ngs(cfu_exact_solution)

op = Coefficient(
    domain=domain, codomain=codomain,
    rhs=10 * ngs.sin(ngs.x) * ngs.sin(ngs.y),
    bc_left=0, bc_right=0, bc_bottom=0, bc_top=0,
    diffusion=False, reaction=True
)

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=Sobolev)

data = op(exact_solution) + noise

reco, reco_data = IrgnmCG(
    setting, data,
    init=domain.from_ngs(1 + ngs.x + 5*ngs.x*(1-ngs.x)*ngs.y*(1-ngs.y)),
    regpar=1, regpar_step=2/3, cgstop=50,
).run(
    rules.CountIterations(15) +
    rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.1)
)

gfu_reco = op.domain.to_ngs(reco)
plots = plotmeshes(domain, reco=gfu_reco, error=cfu_exact_solution - gfu_reco)

plots['reco']

plots['error']
