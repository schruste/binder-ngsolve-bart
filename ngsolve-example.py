import logging

from netgen.geom2d import SplineGeometry
import ngsolve as ngs
import numpy as np
import pyvista as pv

from regpy.discrs.ngsolve import NgsSpace
from regpy.hilbert import L2, Sobolev
from regpy.operators.ngsolve import Coefficient
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.irgnm import IrgnmCG
import regpy.stoprules as rules


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)

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

noiselevel = 0.01

geo = SplineGeometry()
geo.AddRectangle((0.4, 0.45), (0.6, 0.55), leftdomain=0, rightdomain=1)
geo.AddRectangle((0, 0), (1, 1), bcs=["bottom","right","top","left"], leftdomain=1)

domain = NgsSpace(ngs.H1(ngs.Mesh(geo.GenerateMesh(maxh=0.4)), order=1))
codomain = NgsSpace(ngs.H1(ngs.Mesh(geo.GenerateMesh(maxh=0.1)), order=1, dirichlet="left|top|right|bottom"))

cfu_exact_solution = 1 + ngs.x
exact_solution = domain.from_ngs(cfu_exact_solution)

op = Coefficient(
    domain=domain, codomain=codomain,
    rhs=10 * ngs.sin(ngs.x) * ngs.sin(ngs.y),
    bc_left=0, bc_right=0, bc_bottom=0, bc_top=0,
    diffusion=False, reaction=True
)

noise = noiselevel * np.random.randn(codomain.fes.ndof)
data = op(exact_solution) + noise
plotmeshes(codomain, data=codomain.to_ngs(data))['data']

# ---

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=L2)

reco, reco_data = IrgnmCG(
    setting, data,
    init=domain.from_ngs(1 + ngs.x + 5*ngs.x*(1-ngs.x)*ngs.y*(1-ngs.y)),
    regpar=0.1, regpar_step=2/3, cgstop=100,
).run(
    rules.CountIterations(20) +
    rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.01)
)

gfu_reco = op.domain.to_ngs(reco)
plots = plotmeshes(domain, reco=gfu_reco, error=cfu_exact_solution - gfu_reco)

plots['reco']

# ---

plots['error']
