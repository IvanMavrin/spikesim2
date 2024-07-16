import numpy as np
from dataclasses import dataclass
from AER import *
from coding import *
from typing import Callable, List    

class ConvWeights:
    def __init__(self, pos, kernels, shape):
        self.pos = tuple(pos)
        self.kernels = tuple(kernels)
        self.shape = tuple(shape)
    
    def __matmul__(self, m):
        sr, sc = self.shape
        out = np.zeros(sr)
        if len(m.shape) == 1:
            for i in range(len(self.pos[0])):
                (r, c) = self.pos[0][i]
                (k, l) = self.pos[1][i]
                val = self.kernels[k][l]
                out[r] += val*m[c]
        else:
            mr, mc = m.shape
            if sc != mr:
                raise ValueError(f"Dimension mismatch, self dim 1 = {sc} not equal matrix dim 0 = {mr}")
            out = np.zeros((sr, mc))
            for i in range(len(self.pos[0])):
                (r, c) = self.pos[0][i]
                (k, l) = self.pos[1][i]
                val = self.kernels[k][l]
                out[r, :] += val*m[c, :]
        return out
    
    def transpose(self):
        pos_tr = [self.pos[0][:, ::-1], self.pos[1]]
        return ConvWeights(pos_tr, self.kernels, self.shape[::-1])
    
    def copy(self):
        return ConvWeights(self.pos, self.kernels, self.shape)

@dataclass(kw_only=True)
class BaseLayer:
    
    neurons: list
    thres: int
    opt_class: Callable | None = None
    opt_params: dict | None = None
    weights: List[np.ndarray] | None = None
    b: np.ndarray | None = None
    inputs: list | None = None
    prevl: object | None = None
    nextl: object | None = None
    learning: bool = True
    
    def __post_init__(self):
        if self.inputs is None:
            self.inputs = self.prevl.neurons
        if self.weights is None:
            self.weights = np.random.rand(len(self.neurons), len(self.inputs))
        if self.b is None:
            self.b = np.zeros(len(self.neurons))
            
    def reset(self):
        pass


@dataclass(kw_only=True)
class BaseConvLayer(BaseLayer):
   
    i: int|np.ndarray #input dim
    s: int|np.ndarray #stride speed
    p: int #padding
    
    time_conv: bool = False

    def __post_init__(self):
        super().__post_init__()
        rep = 1
        i, s, p = np.array(self.i), np.array(self.s), np.array(self.p)
        if len(i.shape) == 0:
            i = np.array([i, i])
        elif i.shape == (3,):
            rep = i[0]
            i = i[1:]
        if len(i.shape) == 0:
            s = np.array([s, s])
        if len(p.shape) == 0:
            p = np.array([p, p, p, p])
        elif p.shape == (2,):
            p = np.array([p[0], p[1], p[0], p[1]])
    
        d = (i[0]+p[0]+p[1],i[1]+p[2]+p[3]) #размеры входного окна с полями
        
        mask = np.zeros(d, dtype=bool)
        mask[p[0]:p[0]+i[0], p[2]:p[2]+i[1]] = True #выделяем входное окно
        line_mask = np.concatenate(mask, dtype=bool)#разворачиваем в линию
        line_mask = np.arange(d[0]*d[1])[line_mask]  #находим индексы, соответствующие входам
        indices = np.ones(d[0]*d[1])*np.nan
        indices[line_mask] = np.arange(i[0]*i[1]) #устанавливаем соответствие номеров ячеек в окне с полями и без окон
        self.mask = []
        self.kernels = [np.concatenate(w) for w in self.weights]
        
        if (
                (i[0]*i[1]*rep != len(self.inputs) and not self.time_conv) or 
                (i[0]*rep != len(self.inputs) and self.time_conv)
            ):
            raise ValueError('Input map area should be equal to the input number')
        
        if len(set([w.shape for w in self.weights])) != 1:
            raise ValueError('All kernels must me the same size')
        
        k = self.weights[0].shape
        
        maskp = []
        maskv = []
        N = []
        n = 0
        for shift in range(rep):
            for nk, w in enumerate(self.weights):
                apert = k[0]*k[1]
                pos = np.zeros(d, dtype=np.short)
                pos[:k[0], :k[1]] = 1
                posvec = np.where(pos)[0]*d[1]+np.where(pos)[1]
                
                vals = np.array([(nk,x) for x in np.arange(apert)])
                for x in range(0, d[0]-k[0]+1, s[0]):
                    for y in range(0, d[1]-k[1]+1, s[1]):
                        rolled = posvec+x*d[1]+y
                        rmask = np.in1d(rolled, line_mask)
                        rolled_pos = indices[rolled[rmask]]
                        maskp += [(n, r+shift*i[0]*i[1]) for r in rolled_pos]
                        maskv += list(vals[rmask])
                        N.append((n, self.neurons[nk]))
                        n += 1
                        
        maskp = np.array(maskp, dtype=int)
        maskv = np.array(maskv, dtype=int)
        
        self.mask = (maskp, maskv)
        self.weights = ConvWeights(self.mask, self.kernels, (n, rep*i[0]*i[1]))
        if self.time_conv:
            self.buf = [np.zeros(i[0]) for _ in range(i[1])]
        self.neurons = N
        
        self.shape = (
            len(self.kernels)*rep, 
            np.floor((i[0]+p[0]+p[1]-k[0])/s[0]).astype(int)+1,
            np.floor((i[1]+p[2]+p[3]-k[1])/s[1]).astype(int)+1,
            )
        
    
    def time_buf(self, i):
        self.buf.insert(0, i)
        self.buf.pop(-1)
        return np.concatenate(self.buf)
    
    def reset(self):
        if self.time_conv:
            self.buf = [np.zeros_like(b) for b in self.buf]
         
@dataclass 
class BaseNetwork:
    
    layers: list
    labels: list
    _train: Callable
    _score: Callable
    dt: int
    
    def __post_init__(self):
        self.inputs = self.layers[0].inputs
        self.outputs = self.layers[-1].neurons
        if len(self.layers) > 1:
            for l, pl in zip(self.layers[:-1], self.layers[1:]):
                l.prevl = pl
                pl.nextl = l
                
    def fit(self, X, y, dt=False):
        if not dt:
            dt = self.dt
        self._train(self, X, y, dt)
        return self
    
    def predict(self, X, dt=False):
        if not dt:
            dt = self.dt
        output = self.layers[-1].predict(X, dt)['output']
        return output
    
    def score(self, X, y, dt=False):
        if not dt:
            dt = self.dt
        output = self.predict(X, dt)
        score = self._score(output, y, dt)
        return score

