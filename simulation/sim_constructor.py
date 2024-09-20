from omegaconf import OmegaConf

from .burgers import Burgers
from .darcy import Darcy
# from .branin import Branin
# from .hartmann import Hartmann
from .virtual import *
from .ks import KS
from .ns import *
from .reac import Reac
from .reac_big import Reac_big
from .burgers_big import Burgers_big
from .ks_big import KS_big
from .heat import Heat

SIM_LIST = {
    'burgers': Burgers,
    'darcy': Darcy,
    'branin': Branin,
    'hartmann': Hartmann,
    'borehole': Borehole,
    'piston': Piston,
    'wingweight': WingWeight,
    'otlcircuit': OTLCircuit,
    'ks': KS,
    'ns': NS,
    'styblinski': Styblinski,
    'reac': Reac,
    'synthetic1': Synthetic1,
    'synthetic2': Synthetic2,
    'reac_big': Reac_big,
    'burgers_big': Burgers_big,
    'ks_big': KS_big,
    'heat': Heat,
    'particles': Particles,
    }

def construct_sim(cfg: OmegaConf):
    sim_class = SIM_LIST[cfg.sim.name]
    sim = sim_class(cfg)
    # append useful sim parameters to cfg for use in other modules
    cfg.sim.ndim = sim.ndim
    cfg.sim.fid = sim.fid
    # cfg.sim.fidelity_list = sim.fidelity_list
    # cfg.sim.M = sim.M
    # cfg.sim.costs = sim.costs
    # cfg.sim.N_m = sim.N_m
    if hasattr(sim, 'dim_out'):
        cfg.sim.dim_out = sim.dim_out
    return sim
