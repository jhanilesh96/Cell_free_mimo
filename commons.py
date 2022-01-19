ftype = 'float64'; ctype = 'complex'+str(2*int(ftype[-2:]))
import os, sys
import time
import numpy as np;
import tensorflow as tf;
tf.keras.backend.set_floatx(ftype)
import tensorflow_probability as tfp;
tfk = tf.keras;
tfd = tfp.distributions;
from scipy.stats import unitary_group, cauchy, levy, t;
np.random.seed(1233);
tf.random.set_seed(1233);
import matplotlib.pyplot as plt
import scipy
import spektral as sk
import myutils

dist = lambda _c1,_c2 : np.sqrt( np.square(_c1[0]-_c2[0]) + np.square(_c1[1]-_c2[1]) )
c = lambda _a : tf.dtypes.cast(_a, ctype)
RfC = lambda _a : tf.concat([tf.math.real(_a), tf.math.imag(_a)], axis=-1)
CfR = lambda _a : c(_a[:,:_a.shape[1]//2]) + 1j*c(_a[:,_a.shape[1]//2:])
eps = 1e-10
eps_2 = 1e-8
do_we_reduce_mean=True
coopMax, Ns, N, M = 8, 1, 16, 1
bf_norm_max = tf.constant(1.0, dtype=ftype)
assert M == Ns # to make it backwards compatible with Baselines


def SparseToCSR(A):
    return scipy.sparse.csr_matrix((A.values.numpy(),(A.indices.numpy()[:,0], A.indices.numpy()[:,1])),shape=A1.shape)


def CSRToSparse(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.SparseTensor(indices, coo.data, coo.shape)

