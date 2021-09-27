import logging

from netgen.geom2d import SplineGeometry
import ngsolve as ngs
import numpy as np
from ngsolve.webgui import Draw

from regpy.discrs.ngsolve import NgsSpace
from regpy.hilbert import L2, Sobolev
from regpy.operators.ngsolve import Coefficient
from regpy.solvers import HilbertSpaceSetting
from regpy.solvers.landweber import Landweber
from regpy.solvers.irgnm import IrgnmCG
import regpy.stoprules as rules

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s'
)


noiselevel = 0.0001

geo = SplineGeometry()
geo.AddRectangle((0.4, 0.45), (0.6, 0.55), leftdomain=0, rightdomain=1)
geo.AddRectangle((0, 0), (1, 1), bcs=["bottom","right","top","left"], leftdomain=1)

mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.4))
fes_domain = ngs.H1(mesh, order=1)
domain = NgsSpace(fes_domain)

bdr = "left|top|right|bottom"
fes_codomain = ngs.H1(mesh, order=3, dirichlet=bdr)
codomain = NgsSpace(fes_codomain, bdr=bdr)

cfu_exact_solution = 1 + ngs.x
exact_solution = domain.from_ngs(cfu_exact_solution)

rhs=10 * ngs.sin(ngs.x) * ngs.sin(ngs.y)
op = Coefficient(
    domain, rhs, codomain=codomain, bc = 0.1, diffusion=False,
    reaction=True
)

noise = noiselevel * np.random.randn(codomain.fes.ndof)
data = op(exact_solution) + noise

Draw(mesh, codomain.to_ngs(data), "data")

# ---

setting = HilbertSpaceSetting(op=op, Hdomain=L2, Hcodomain=Sobolev)

init = domain.from_ngs ( 1 )
init_data = op(init)

landweber = Landweber(setting, data, init, stepsize=1)
stoprule = (
        rules.CountIterations(500) +
        rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.1))

reco, reco_data = landweber.run(stoprule)

#reco, reco_data = IrgnmCG(
#    setting, data,
#    init=domain.from_ngs(1 + ngs.x + 5*ngs.x*(1-ngs.x)*ngs.y*(1-ngs.y)),
#    regpar=0.1, regpar_step=2/3, cgstop=100,
#).run(
#    rules.CountIterations(20) +
#    rules.Discrepancy(setting.Hcodomain.norm, data, noiselevel=setting.Hcodomain.norm(noise), tau=1.01)
#)

gfu_reco = op.domain.to_ngs(reco)

Draw(mesh, cfu_exact_solution, "exact")

Draw(mesh, gfu_reco, "reco")

Draw(mesh, gfu_reco - cfu_exact_solution, "diff")

