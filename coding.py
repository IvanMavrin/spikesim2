from AER import *
import numpy as np

class TemporalContrast:
    """
    After Petro, B., Kasabov, N., & Kiss, R. M. (2020). Selection and Optimization of Temporal Spike Encoding Methods for Spiking Neural Networks. IEEE Transactions on Neural Networks and Learning Systems, 31(2), 358–370. doi:10.1109/tnnls.2019.2906158 
    """
    def ThresholdRepresentation(vector, dt, ts, threshold, onset_address='ONSET', offset_address='OFFSET'):
        """
        vector: 1d array of signal samples
        dt: signal sampling interval
        ts: spike coding frequency, ts >= dt
        threshold: threshold to compare signal against
        """
        o = {}
        T = len(vector)*dt
        K = round(T/ts)
        for i in range(0, K):
            k = round(i*ts/dt)
            o[i] = [onset_address] if vector[i]>threshold else [offset_address]
        for i in range(0, len(vector), ks):
            o[i*dt] = [onset_address] if vector[i]>threshold else [offset_address]
        return o

    def StepForward(vector, dt, ts, delta, onset_address='ONSET', offset_address='OFFSET'):
        """
        vector: 1d array of signal samples
        dt: signal sampling interval
        ts: spike coding frequency, ts >= dt
        threshold: threshold to compare signal against
        """
        o = {}
        ks = round(ts/dt)
        base = vector[0]
        for i in range(0, len(vector), ks):
            if vector[i]>base+delta:
                o[i*dt] = [onset_address]
                base += delta
            if vector[i]<base-delta:
                o[i*dt] = [offset_address]
                base -= delta
                
        return o

    def MovingWindow(vector, dt, dw, ts, delta, onset_address='ONSET', offset_address='OFFSET'):
        """
        vector: 1d array of signal samples
        dt: signal sampling interval
        dw: moving window width, dw > dt
        ts: spike coding frequency, dw > ts >= dt
        threshold: threshold to compare signal against
        """
        o = {}
        dw = round(dw/dt)
        ks = round(ts/dt)
        base = sum(vector[0:dw])/dw
        for i in range(0, dw, ks):
            if vector[i]>base+delta:
                o[i*dt] = [onset_address]
            if vector[i]<base-delta:
                o[i*dt] = [offset_address]
        for i in range(dw, len(vector), ks):
            base = sum(vector[i-dw:i])/dw
            if vector[i]>base+delta:
                o[i*dt] = [onset_address]
            if vector[i]<base-delta:
                o[i*dt] = [offset_address]
        return o
    
    def ThresholdRepresentationDecode(AER, dt, low, high, onset_address='ONSET', offset_address='OFFSET'):
        """
        AER: spike coded signal
        dt: signal sampling interval
        low: low signal level
        high: high signal level
        """
        REA = AER_inverse(AER)
        t = []
        if onset_address in REA:
            t += REA[onset_address]
        if offset_address in REA:
            t += REA[offset_address]
        t.sort()
        
        T = max(t)
        K = round(T/dt)
        o = np.zeros(K)
        base = 0
        for t1, t2 in zip(t[:-1], t[1:]):
            if onset_address in AER[t1]:
                base = high
            if offset_address in AER[t2]:
                base = low
            o[round(t1/dt):round(t2/dt)] = base
        return o

    def StepForwardDecode(AER, dt, delta, onset_address='ONSET', offset_address='OFFSET'):
        """
        AER: spike coded signal
        dt: signal sampling interval
        delta: signal change threshold
        """
        REA = AER_inverse(AER)
        t = []
        if onset_address in REA:
            t += REA[onset_address]
        if offset_address in REA:
            t += REA[offset_address]
        if len(t) == 0:
            o = np.zeros(1)
            return o
        t.sort()
        T = max(t)
        K = round(T/dt)
        o = np.zeros(K)
        base = 0
        for t1, t2 in zip(t[:-1], t[1:]):
            if onset_address in AER[t1]:
                base += delta
            if offset_address in AER[t2]:
                base -= delta
            o[round(t1/dt):round(t2/dt)] = base
        return o

    def MovingWindowDecode(AER, dt, dw, delta, onset_address='ONSET', offset_address='OFFSET'):
        """
        AER: spike coded signal
        dt: signal sampling interval
        dw: moving window width, dw > dt
        delta: signal change threshold
        """
        REA = AER_inverse(AER)
        t = []
        if onset_address in REA:
            t += REA[onset_address]
        if offset_address in REA:
            t += REA[offset_address]
        t.sort()
        T = max(t)
        K = round(T/dt)
        o = np.zeros(K)
        dw = round(dw/dt)
        base = 0
        for t1, t2 in zip(t[:-1], t[1:]):
            if onset_address in AER[t1]:
                o[round(t1/dt):round(t2/dt)] = base + delta
            if offset_address in AER[t2]:
                o[round(t1/dt):round(t2/dt)] = base - delta
            base = sum(o[round(t2/dt)-dw:round(t2/dt)])/dw
        return o
    
class TimeBased:
    
    def ISI(vector, dt, max_delay, min_delay=0.1, max_value=1, min_value=0, address='OUT'):
        """
        vector: 1d array of signal samples
        dt: signal sampling interval
        max_value: signal value for which the delay should be minimal
        max_delay: maximum time interval to emit spike, max_delay > 0
        """
        o = {}
        i = 0
        while round(i) < len(vector):
            delay = ((max_value-vector[round(i)])/(max_value-min_value)*max_delay)
            o[i*dt+delay] = [address]
            i += max(delay/dt, min_delay)
            
        return o
    
    
    def ISIDecode(AER, dt, max_delay,  max_value=1, min_value=0, address='OUT'):
        """
        AER: spike coded signal
        dt: signal sampling interval
        max_value: signal value for which the delay should be minimal
        min_delay: minimal time interval to emit spike, min_delay < 1/fs
        """
        REA = AER_inverse(AER)
        T = max(REA[address])
        K = round(T/dt)
        o = np.zeros(K)
        t_spike = sorted(REA[address])
        for t1, t2 in zip(t_spike[:-1], t_spike[1:]):
            delay = t2-t1
            value = max_value-(delay/max_delay)*(max_value-min_value)
            o[round(t1/dt):round(t2/dt)] = value
        return o
    def TTFS(vector, dt, ts, max_value, min_value, address='OUT'):
        """
        vector: 1d array of signal samples
        dt: signal sampling interval
        ts: spike coding frequency, ts >= dt
        max_value: signal value for which the delay should be minimal
        min_value: signal value for which the delay should be maximal
        """
        o = {}
        ks = round(ts/dt)
        
        for i in range(0, len(vector), ks):
            delay = (max_value-vector[i])/(max_value-min_value)*ks
            o[(i+delay)*dt] = [address]
            
        return o
 
    
    def TTFSDecode(AER, dt, ts, max_value, min_value, address='OUT'):
        """
        AER: spike coded signal
        dt: signal sampling interval
        ts: spike coding frequency, ts >= dt
        max_value: signal value for which the delay should be minimal
        min_delay: minimal time interval to emit spike, min_delay < 1/fs
        """
        REA = AER_inverse(AER)
        T = max(REA[address])
        K = round(T/dt)
        ks = round(ts/dt)
        o = np.zeros(K)
        t_spike = sorted(REA[address])
        for t1, t2, ti in zip(np.arange(0, T-ts, ts), np.arange(ts, T, ts), t_spike):
            delay = (ti-t1)
            value = max_value-(delay/ts)*(max_value-min_value)
            o[round(t1/dt):round(t2/dt)] = value
        return o
 
    
class KernelBased:
    def BSA(signal, dt, ts, kernel, threshold):
        KF = len(kernel)
        KS = len(signal)
        o = {}
        signal = np.pad(signal.astype(np.float64), (0, KF), 'edge')
        for i in range(0, KS, round(ts/dt)):
            
            error1 = np.sum(np.abs(signal[i:i+KF]-kernel))
            error2 = np.sum(np.abs(signal[i:i+KF]))
            
            if error1 + threshold < error2:
                o[i*dt] = ['OUT']
                signal[i:i+KF] -= kernel
        return o
    
    def BSADecode(AER, dt, kernel, address="OUT"):
        REA = AER_inverse(AER)
        T = max(REA[address])
        KF = len(kernel)
        KS = round(T/dt)
        o = np.zeros(KF+KS)
        
        for t in REA[address]:
            o[round(t/dt):round(t/dt)+KF] += kernel
            
        return o
    
class Custom:
    def SF_ISI(vector, dt, max_value, min_value, max_delay, address='OUT'):
        """
        vector: 1d array of signal samples
        dt: signal sampling interval
        max_value: signal value for which the delay should be minimal
        max_delay: maximum time interval to emit spike, max_delay > 0
        """
        o = {0: ['OUT']}
        i = 0
        base = 0
        while i < len(vector):
            v_norm = (base-vector[i])/(max_value-min_value)
            delay = round(((v_norm/(1+v_norm**2)+0.5)*max_delay)/dt)
            delay = max(1, delay)
            o[(i+delay)*dt] = [address]
            base = vector[i]
            i += delay
        return o
    
    def SF_ISIDecode(AER, dt, max_value, min_value, max_delay, address='OUT'):
        """
        AER: spike coded signal
        dt: signal sampling interval
        max_value: signal value for which the delay should be minimal
        min_delay: minimal time interval to emit spike, min_delay < 1/fs
        """
        REA = AER_inverse(AER)
        T = max(REA[address])
        K = round(T/dt)
        o = np.zeros(K)
        t_spike = sorted(REA[address])
        base = 0
        for t1, t2 in zip(t_spike[:-1], t_spike[1:]):
            delay = t2-t1
            delay_norm = delay/max_delay-0.5
            if delay_norm == 0:
                value = base
            else:
                _val = (1-np.sqrt(1-4*delay_norm**2))/(2*delay_norm)
                value = base-_val*(max_value-min_value)
            o[round(t1/dt):round(t2/dt)] = value
            base = value
        return o
        
    
class KernelFunctions:
    def LeakyExponentialKernel(tau, epsilon):
        """
        tau: time constant
        epsilon: desireable error
        """
        def kernel(dt):
            T = -np.log(epsilon)*tau
            K = round(T/dt)
            t = np.arange(0, K)*dt
            eps = np.exp(-(K-1)*dt/tau)
            return {
                'timescale': t, 
                'value': np.exp(-t/tau)-eps
            }
        return kernel

    def ExponentialKernel(tau):
        """
        tau: time constant
        """
        def kernel(dt, T):
            K = round(T/dt)
            t = np.arange(0, K)*dt
            return {
                'timescale': t, 
                'value': np.exp(-t/tau)
            }
        return kernel
    
    def CExponentialKernel(tau, C):
        """
        tau: time constant
        """
        def kernel(dt, T):
            K = round(T/dt)
            t = np.arange(0, K)*dt
            return {
                'timescale': t, 
                'value': C*np.exp(-t/tau)
            }
        return kernel

    def AlphaKernel(tau):
        """
        tau: time constant
        """
        def kernel(dt, T):
            K = round(T/dt)
            t = np.arange(0, K)*dt
            return {
                'timescale': t, 
                'value': np.exp(1-t/tau)*t/tau
            }
        return kernel

    def GaussKernel(sigma):
        """
        sigma: standart deviation
        """
        def kernel(dt, T, mu=0):
            t = np.arange(-T/2, T/2, dt)
            return {
                'timescale': t, 
                'value': np.exp(-0.5*(t-mu)**2/sigma**2)
            }
        return kernel

    def NormGaussKernel(sigma, T):
        """
        sigma: standart deviation
        """
        def kernel(dt, mu=0):
            t = np.arange(-T/2, T/2, dt)
            return {
                'timescale': t, 
                'value': np.exp(-0.5*(t-mu)**2/sigma**2)/(sigma*np.sqrt(2*np.pi))
            }
        return kernel

                
                         
            