import pickle
import os
import numpy as np
import compress_pickle

folderName = 'data'

def pathMaker(folderName=folderName, subFolderName=None):
    path = folderName + '\\' + (subFolderName+'\\' if subFolderName is not None else '')
    return path;

def save(f, obj, folderName=folderName, subFolderName=None, suffix='.pkl', compression=None):
    path = pathMaker(folderName=folderName, subFolderName=subFolderName);
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path+f+suffix, 'wb') as handle:
        compress_pickle.dump(obj, handle, compression=compression)

def check(f, folderName=folderName, subFolderName=None, suffix='.pkl'):
    path = pathMaker(folderName=folderName, subFolderName=subFolderName);
    return os.path.exists(path+f+suffix);

def load(f, folderName=folderName, subFolderName=None, suffix='.pkl', compression=None):
    path = pathMaker(folderName=folderName, subFolderName=subFolderName);
    with open(path+f+suffix, 'rb') as handle:
        return compress_pickle.load(handle, compression=compression)

def remove(f, folderName=folderName, subFolderName=None, suffix='.pkl'):
    path = pathMaker(folderName=folderName, subFolderName=subFolderName);
    return os.remove(path+f+suffix);



def _loggap():
    print('\n\n\n-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------\n\n\n')


def markov(alpha=0.2, beta=0.8, numStates=100, initState=None):
    k = np.zeros(numStates);
    steady_state = np.array([beta/(alpha+beta), alpha/(alpha+beta)])
    if initState is None:
        k[0] = np.random.rand()<steady_state[1]
    else:
        k[0] = initState;
    for i in range(1,numStates):
        if k[i-1]==0:
            k[i] = np.random.rand()<alpha;      # p10
        else:
            k[i] = np.random.rand()<(1-beta);   # p11
    return k

def run_avg(a,window=10):    
    return np.array([np.mean(a[i:i+window]) for i in range(len(a)-window)])

class Empty(): pass;
