import numpy as np
from matplotlib import pyplot as plt
def AER_to_discrete(AER, dt, addresses=None):
    """
    AER: {time: [address]}
    dt: int
    addresses: list(address)
    returns 2-d array of len(addresses) X np.ceil(max(AER.keys())/dt)
    """
    REA = AER_inverse(AER)
    if addresses is None:
        addresses = list(set(REA.keys()))
    r = np.zeros((len(addresses), np.ceil(max(AER.keys())/dt).astype(int)+1), dtype=int)
    addr_to_index = {a: i for i, a in enumerate(addresses)}
    for a in addresses:
        if a in REA:
            for t in REA[a]:
                r[addr_to_index[a], np.round(t/dt).astype(int)] += 1
    return r

def discrete_to_AER(discrete, dt, addresses):
    """
    dicrete: 2-d array of len(addresses) X np.ceil(max(AER.keys())/dt)
    dt: int
    addresses: list(address)
    returns AER reoresentation: {time: address}
    """
    AER = {}
    for i, k in zip(*np.nonzero(discrete)):
        t = k*dt
        if t not in AER:
            AER[t] = []
        AER[t] += [addresses[i]]*int(discrete[i, k])
    return AER

def AER_to_pairs(AER):
    pairs = []
    for k, v in AER.items():
        for vv in v:
            pairs.append([k, vv])
    return pairs
    
def pairs_to_AER(pairs):
    AER = {}
    for k, v in pairs:
        if k not in AER:
            AER[k] = []
        AER[k].append(v)
    return AER

def AER_inverse(AER):
    REA = {}
    for k, v in AER.items():
        for vv in v:
            if vv not in REA:
                REA[vv] = []
            REA[vv].append(k)
    return REA

def AER_noise(labels, T, f):
    """
        Returns pseudo-noise signal with frequency=f 
    """
    noise = {k: [] for k in labels}
    for k in range(round(T*f)):
        l = labels[np.random.randint(len(labels))]
        noise[l].append(np.random.rand()*T)
    return AER_inverse(noise)

def display_AER(AER, labels_list=None, ax=None, T=None):
    REA = AER_inverse(AER)
    if labels_list is None:
        labels_list = list(REA.keys())
    if ax is None:
        ax = plt.subplot()
    if T is not None:
        ax.set_xlim(T)
    ax.set_yticks(np.arange(len(labels_list))+0.5, labels_list)
    for k,v in REA.items():
        i = labels_list.index(k)
        ax.stem(v, np.ones_like(v)*(i+1), bottom=i, basefmt="w", markerfmt="", linefmt="k")
    ax.set_ylim((0, len(labels_list)))
def merge_AER(AER1, AER2):
    o = {}
    timeline = sorted(list(set(AER1.keys()).union(set(AER2.keys()))))
    for t in timeline:
        o[t] = []
        if t in AER1:
            o[t] += AER1[t]
        if t in AER2:
            o[t] += AER2[t]
    return o
