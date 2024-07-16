from numba import njit, prange
import numpy as np
from coding import AER_to_discrete, AER_noise
from network import BaseConvLayer, BaseLayer, ConvWeights
from line_profiler import profile

@njit
def predict_step(up, op, fp, w,  b, V_TH, F):
    x = (np.ascontiguousarray(w))@(np.ascontiguousarray(op))
    u = up*fp + x + b
    o = np.where(u >= V_TH, 1, 0)
    f = np.where(o, 0, F)
    return  u, o, f

@njit
def predict_step_convolution(up, op, fp, pos, kernels, shape,  b, V_TH, F):
    sr, _ = shape
    x = np.zeros(sr)
    for i in range(len(pos[0])):
        (r, c) = pos[0][i]
        (k, l) = pos[1][i]
        val = kernels[k][l]
        x[r] += val*op[c]
    u = up*fp + x + b
    o = np.where(u >= V_TH, 1, 0)
    f = np.where(o, 0, F)
    return  u, o, f

@njit
def h1(u, a, V_TH):
    return 1/a*(np.abs((u-V_TH))<(a/2))
@njit
def h2(u, a, V_TH):
    return (np.sqrt(a)/2-a/4*np.abs(u-V_TH))*((2/np.sqrt(a)-np.abs(u-V_TH))>0)
    

class Network_STBP():
 
    def __init__(self, layers, dt, tau):
        self.inputs = layers[0].inputs
        self.outputs = layers[-1].neurons
        self.layers=layers
        for l in self.layers:
            l.network = self
        self.o = None
        self.u = None
        self.f = None
        try:
            int(tau)
            self.tau = [tau]*len(layers)
        except:
            self.tau = []
            for l,x in zip(layers, tau):
                try:
                    int(x)
                    self.tau.append(x)
                except: 
                    self.tau.append(np.random.rand(len(l.neurons))*(x[1]-x[0])+x[0])
        neurons_count = [len(l.neurons) for l in layers]
        neurons_count.insert(0, len(self.inputs))
        
        self.dt = dt
        
        self.specific_predicts= {
            FCLayer: predict_step,
            ConvLayer: predict_step_convolution
        }
    
    def predict(self, X, dt=None, full_output = False, discrete_input=False):
        if dt is None:
            dt = self.dt
        neurons_count = [len(l.neurons) for l in self.layers]
        neurons_count.insert(0, len(self.inputs))
        layers = self.layers.copy()
        layers.insert(0, None)
        nl = len(layers)
        
        if not discrete_input:
            i = AER_to_discrete(X, dt, addresses=self.inputs)
        else:
            i = X
        K = i.shape[1]

        nl = len(neurons_count)
        u = [np.zeros((nc, K)) for nc in neurons_count]
        f = [np.zeros((nc, K)) for nc in neurons_count]
        o = [np.zeros((nc, K)) for nc in neurons_count]
        
        for l in self.layers:
            l.reset()
        
        o[0][:, :] = i
        for t in range(K-1):
            for n in np.arange(1, nl):
                F = np.exp(-dt/self.tau[n-1])
                op = o[n-1][:, t+1]
                if layers[n].type == "FC":
                    u[n][:, t+1], o[n][:, t+1], f[n][:, t+1] = predict_step(
                        u[n][:, t], 
                        op, 
                        f[n][:, t], 
                        layers[n].weights, 
                        layers[n].b, 
                        layers[n].thres, 
                        F)
                if layers[n].type == "CV":
                    if layers[n].time_conv:
                        op = layers[n].time_buf(o[n-1][:, t+1])
                    u[n][:, t+1], o[n][:, t+1], f[n][:, t+1] = predict_step_convolution(
                        u[n][:, t], 
                        op, 
                        f[n][:, t], 
                        layers[n].weights.pos, 
                        layers[n].weights.kernels, 
                        layers[n].weights.shape, 
                        layers[n].b, 
                        layers[n].thres, 
                        F)
        self.u = u
        self.o = o
        self.f = f
        if full_output:        
            return u, o, f
        else:
            return o
    
    def fit(self, X, Y, dt=None, discrete_input=False):
        if dt is None:
            dt = self.dt
        layers = self.layers.copy()
        layers.insert(0, None)
        k = 0
        
        if dt is None:
            dt = self.dt
        neurons_count = [len(l.neurons) for l in self.layers]
        neurons_count.insert(0, len(self.inputs))
        nl = len(neurons_count)-1
        N = nl
        
        for x, y in zip(X, Y):
            k += 1
            
            u, o, f = self.predict(x, dt, discrete_input=discrete_input, full_output=True)
            K = o[0].shape[1]-1

            dL_do = [np.zeros((nc, K+1)) for nc in neurons_count]
            dL_du = [np.zeros((nc, K+1)) for nc in neurons_count]
            
            df_do = [None]
            for n in np.arange(1, N+1):
                df_do.append(f[n]/dt)
                df_do[n][:, 1:] -= df_do[n][:, :-1]

            a = 2.5
            h = h2

            #t = T, output layer, neuron i
            r = self.firing_rate()
            dL_do[-1][:, -1] = -1/(K)*(y-r)
            # print(dL_do[-1][:, -1])
            dL_du[-1][:, -1] = dL_do[-1][:, -1]*h(u[-1][:, -1], a, layers[-1].thres)
            
            for t in np.arange(K-1, -1, -1):
                #t < T, output layer, neuron i
                dg_do_tp = h(u[-1][:, t+1], a, layers[-1].thres)
                dL_do[-1][:, t] = dL_do[-1][:, t+1]*dg_do_tp*u[-1][:, t]*df_do[N][:, t]+dL_do[-1][:, K]
                dL_du[-1][:, t] = dL_do[-1][:, t+1]*dg_do_tp*f[-1][:, t]


            #t = T, hidden layer, neuron i
            
            for n in np.arange(N-1, 0, -1):
                dg_do_tp = h(u[n][:, -1], a, layers[n].thres)
                bp = layers[n+1].weights.transpose()@dL_do[n+1]
                if layers[n+1].type == "CV":
                    if layers[n+1].time_conv:
                        depth = layers[n+1].i[1]
                        chunk = int(bp.shape[0]/depth)
                        acc = np.zeros((chunk, bp.shape[1]))
                        acc += bp[0:chunk]
                        for i in range(1, depth):
                            acc[:, i:] += bp[chunk*i:chunk*(i+1), i:]
                        bp = acc
                dL_do[n][:, -1] = bp[:, -1]*dg_do_tp
                dL_du[n][:, -1] = dL_do[n][:, -1]*dg_do_tp
                    

                for t in np.arange(K-1, -1, -1):

                #t < T, hidden layer, neuron i
                    dg_do_t = h(u[n][:, t], a, layers[n].thres)
                    dL_do[n][:, t] = bp[:, t]*dg_do_t+dL_do[n][:, t+1]*dg_do_t*u[n][:, t]*df_do[n][:, t]
                    dL_du[n][:, t] = dL_do[n][:, t]*dg_do_t+dL_do[n][:, t+1]*dg_do_tp*f[n][:, t]
                    dg_do_tp = dg_do_t
      
            for l, grad in zip(self.layers, dL_du[1:]):
                l.calc_grad(grad) 
        for l in self.layers:
            l.fit() 
        return self
    
    def firing_rate(self):
        return np.array([np.mean(self.o[-1][i, :]) for i in np.arange(len(self.outputs))])

    def score(self, X, Y, dt=None, discrete_input=False):
        k = 0
        c = 0
        for x, y in zip(X, Y):
            
            c0 = np.mean(y)**2
            k+=1
            u, o, f = self.predict(x, dt, discrete_input=discrete_input, full_output=True)
            r = self.firing_rate()
            c += np.sum((y-r)**2)/c0
        return c/k
    
    def accuracy(self, X, Y, l=0.3, h=0.7, dt=None, discrete_input=False):
        k = 0
        c = 0
        for x, y in zip(X, Y):
            k+=1
            u, o, f = self.predict(x, dt, discrete_input=discrete_input, full_output=True)
            r = self.firing_rate()
            
            r_true = r>=h
            r_false = r<=l
            
            y_true = y>=h
            y_false = y<=l
            
            TP = np.count_nonzero(r_true&y_true)/np.count_nonzero(y_true)
            TF = np.count_nonzero(r_false&y_false)/np.count_nonzero(y_false)
            FP = np.count_nonzero(r_true&y_false)/np.count_nonzero(y_false)
            FF = np.count_nonzero(r_false&y_true)/np.count_nonzero(y_true)
            
            c += (TP+TF-FP-FF)/(len(y))
        return c/k
    
    @staticmethod
    def random_weights(n_neurons, n_inputs):
        w = np.random.rand(n_neurons, n_inputs)
        _w = np.sqrt(np.sum(w**2, axis=1))
        for i in range(n_neurons):
            w[i] = w[i]/_w[i]
        return w
    
class LIOutLayer(BaseLayer):
    
    type = "FC"
    
    def __init__(self, **kwargs):
        kwargs['thres'] = np.nan
        super().__init__(**kwargs)
        if self.b is None:
            self.b = np.zeros(len(self.neurons))
        
        if self.learning:
            self.w_optimizer = self.opt_class(self.weights, **self.opt_params) 
            self.grads = []
        
    def calc_grad(self, dL_du):
        if self.learning:
            i = [x is self for x in self.network.layers].index(True) #грязный хак потому что index возвращает ошибку
            self.grads.append(dL_du@np.transpose(self.network.o[i]))
        
    def fit(self):
        if self.learning:
            self.weights = self.w_optimizer(np.mean(self.grads, axis=0))
            self.grads = []
        
class FCLayer(BaseLayer):
    
    
    type = "FC"
    
    def __post_init__(self):
        super().__post_init__()
        if self.b is None:
            self.b = np.zeros(len(self.neurons))
        if self.learning:
            self.w_optimizer = self.opt_class(self.weights, **self.opt_params) 
            self.b_optimizer = self.opt_class(self.b, **self.opt_params)
            self.b_grads = []
            self.w_grads = []
        
    def calc_grad(self, dL_du):
        if self.learning:
            i = [x is self for x in self.network.layers].index(True) #грязный хак потому что index возвращает ошибку
            self.b_grads.append(np.sum(dL_du, axis=1))
            self.w_grads.append(dL_du@np.transpose(self.network.o[i]))
        
    def fit(self):
        if self.learning:
            #self.b = self.b_optimizer(np.mean(self.b_grads, axis=0))
            self.weights = self.w_optimizer(np.mean(self.w_grads, axis=0))
            self.b_grads = []
            self.w_grads = []

@njit
def conv_grad_calc(dL_du, dL_dk, mask, op):
    for i in range(len(mask[0])):
        (r, c) = mask[0][i]
        (k, l) = mask[1][i]
        dL_dk[k][l] += np.sum(dL_du[r]*op[c, :])
    return dL_dk

class ConvLayer(BaseConvLayer):
    
    type = "CV"
    def __post_init__(self):
        super().__post_init__()
        self.b = np.zeros(len(self.neurons))
        self.update_kernels([k/np.sqrt(np.sum(k**2)) for k in self.kernels])
        if self.learning:
            self.b_optimizer = self.opt_class(self.b, **self.opt_params)
            self.w_optimizers = [self.opt_class(k, **self.opt_params) for k in self.kernels]
            self.b_grads = []
            self.w_grads = []
            
    def update_kernels(self, kernels):
        self.kernels = kernels
        self.weights = ConvWeights(self.weights.pos, self.kernels, self.weights.shape)
        if self.learning:
            self.w_optimizers = [self.opt_class(k, **self.opt_params) for k in self.kernels]
        
    def calc_grad(self, dL_du):
        
        if self.learning:
            i = [x is self for x in self.network.layers].index(True)
            op = self.network.o[i]
            if self.time_conv:
                _ = [op]
                for s in range(1, self.i[1]):
                    _.append(np.pad(op[:, :-s], ((0,0),(s,0)), 'constant', constant_values=0))
                op = np.concatenate(_)
            dL_dk = [np.zeros_like(k) for k in self.kernels]
            dL_dk = conv_grad_calc(dL_du, dL_dk, self.mask, op)
            self.w_grads.append(dL_dk)
        
    def fit(self):
        if self.learning:
            w_grads = np.mean(self.w_grads, axis=0)
            self.kernels = [opt(grad) for opt, grad in zip(self.w_optimizers, w_grads)]
            self.weights = ConvWeights(self.weights.pos, self.kernels, self.weights.shape)
            #self.b = self.b_optimizer(b_grads)
            self.b_grads = []
            self.w_grads = []
            
class PoolingLayer(ConvLayer):
    
    def __init__(self, *, prevl, i, k, p, th=0.1):
        super().__init__(
            neurons = np.array(["pool"]), 
            prevl = prevl,
            weights = [np.ones(k)/(k[0]*k[1])],
            thres = th,
            opt_class = None,
            opt_params = None,
            i = i,
            s = k,
            p = p,
            time_conv = False,
            learning=False
        )