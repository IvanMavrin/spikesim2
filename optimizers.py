import numpy as np
from dataclasses import dataclass

def warmup_linear(TP = 0, TN = 100, DT = 100, t = 0):
    
    def inner():
        nonlocal TP
        nonlocal TN
        nonlocal DT
        nonlocal t
        t = t+1
        if TP == 0 and t < TN:
            return t/DT
        else:
            if t > TN:
                DT = 2*DT
                TP, TN = TN, TN+DT
            return 1-(t-TP)/DT
        
    return inner

def ADAMW(
    parameters, 
    l = 25e-4, 
    a = 1e-3, 
    b1 = 0.9, 
    b2= 0.999, 
    eps = 1e-8, 
    eta = None,
    clip_norm = 1e-3):
    
    """
    implementation after  Ilya Loshchilov, Frank Hutter "Decoupled Weight Decay Regularization", 2017, arXiv:1711.05101 [cs.LG]
    """
    
    params = parameters
    m = np.zeros_like(parameters)
    v = np.zeros_like(parameters)
    
    TP = 0
    TN = 100
    DT = 100
    t = 0
    
    def SchMul(t):
        nonlocal TP
        nonlocal TN
        nonlocal DT
        if TP == 0:
            return t/DT
        else:
            if t > TN:
                DT = 2*DT
                TP, TN = TN, TN+DT
            return 1-(t-TP)/DT
    
    if eta is None:
        eta = SchMul
    
    def inner(grad):
        nonlocal params
        nonlocal m 
        nonlocal v 
        nonlocal t
        t += 1
        norm = np.sqrt(np.sum(grad**2))
        if norm > clip_norm:
            gradh = grad/norm*clip_norm
        else:
            gradh = grad
        m = b1*m+(1-b1)*gradh
        v = b2*v+(1-b2)*gradh**2
        mhat = m/(1-b1**t)
        vhat = v/(1-b2**t)
        params = params - eta(t)*(a*mhat/(np.sqrt(vhat)+eps)+l*params)
        return params
    
    return inner    

def SimulatedAnnealing(parameters, upbond, lobond, vel0, prob_accept=0.2):
    stable = np.array(list(parameters.values()))
    params = stable.copy()
    last_best = stable.copy()
    cerr_best = 1
    labels = parameters.keys()
    cerr_p = 1
    
    TP = 0
    DT = 3
    TN = DT
    t = 0
    
    def SchMul(t):
        nonlocal TP
        nonlocal TN
        nonlocal DT
        if t > TN:
            DT = 2*DT
            TP, TN = TN, TN+DT
        return 1-(t-TP)/DT
    
    def inner(cerr):
        nonlocal stable
        nonlocal params
        nonlocal cerr_p
        nonlocal cerr_best
        nonlocal last_best
        nonlocal t
        if cerr < cerr_best:
            cerr_best = cerr
            last_best = stable.copy()
            t = 0
        if cerr < cerr_p or np.random.random() < prob_accept:
            stable = params.copy()
            cerr_p = cerr
        t += 1
        params = np.random.normal(stable, vel0)*SchMul(t)+last_best*(1-SchMul(t))
        params = np.where(params>upbond, upbond, params)
        params = np.where(params<lobond, lobond, params)
        return dict(zip(labels, params))
    return inner


@dataclass
class ParticleSwarm:
    
    labels: list
    upbond: np.ndarray
    lobond: np.ndarray
    vel: np.ndarray
    scorer: object
    pool_n:int | None = None
    n_particles: int = 10
    particle_params: dict | None = None
    
    def __post_init__(self):
        if self.particle_params is None:
            self.particle_params = {}
        self.particle_params.update({"swarm": self})
        self.particles = [Particle(**self.particle_params) for _ in range(self.n_particles)]
        self.gb = np.inf
        self.gb_loc = None
        self.map = {}
    
    def search_step(self):
        estimates = []
        for particle in self.particles:
            new_p, err = particle.search_step()
            if particle.loc is not None:
                self.map[tuple(particle.loc.copy())] = err
                if err < self.gb:
                    self.gb = err
                    self.gb_loc = particle.loc.copy()
                estimates.append((new_p, err))
        return estimates
    
@dataclass
class Particle:
    
    
    swarm: ParticleSwarm
    b: int = 0.9
    a: int = 0.1
    DT: int = 5
    init_search_breadth: int = 10
    
    def __post_init__(self):
    
        self.TP = 0
        self.t = 0
        self.TN = self.DT
        self.pb_loc = None
        self.loc = None
        self.pb = np.inf
        self.imp = np.zeros_like(self.swarm.vel)

    
    def SchMul(self):
        self.t = self.t+1
        if self.t > self.TN:
            self.DT = 2*self.DT
            self.TP, self.TN = self.TN, self.TN+self.DT
        return 1-(self.t-self.TP)/self.DT
        
    def search_step(self,):
        
        if self.init_search_breadth:
            self.init_search_breadth -= 1
            
            self.loc = np.random.random(*self.swarm.vel.shape)*(self.swarm.upbond-self.swarm.lobond)
            r = dict(zip(self.swarm.labels, self.loc))
            err = self.swarm.scorer(r)
            if err < self.pb:
                self.pb = err
                self.pb_loc = self.loc.copy()
            return r, err
        v = np.random.normal(0, self.swarm.vel)
        drift_pb = np.zeros_like(v)
        if (self.pb_loc != self.loc).any():
            drift_pb = (self.pb_loc - self.loc)/np.sqrt(np.sum((self.pb_loc - self.loc)**2))*np.sqrt(np.sum(self.swarm.vel**2))
        drift_gb = np.zeros_like(v)    
        if (self.swarm.gb_loc != self.loc).any():
            drift_gb = (self.swarm.gb_loc - self.loc)/np.sqrt(np.sum((self.swarm.gb_loc - self.loc)**2))*np.sqrt(np.sum(self.swarm.vel**2))
        drift_tot = (v+(1-self.pb)*drift_pb+(1-self.swarm.gb)*drift_gb)/(2.1-self.pb-self.swarm.gb)
        self.m = self.b*self.imp+(1-self.b)*drift_tot
        self.loc += self.SchMul()*self.a*self.m
        self.loc = np.where(self.loc>self.swarm.upbond, self.swarm.upbond, self.loc)
        self.loc = np.where(self.loc<self.swarm.lobond, self.swarm.lobond, self.loc)
        r = dict(zip(self.swarm.labels, self.loc))
        err = self.swarm.scorer(r)
        if err < self.pb:
            self.pb = err
            self.pb_loc = self.loc.copy()
        return r, err
