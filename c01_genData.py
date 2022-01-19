'''
Data generator
The project is run on environment tf2, yaml file attached
'''

# activate spec/tf2 to execute this file
import commons; ftype = commons.ftype; ctype = commons.ctype;
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
'''
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
----------------------------------------------------------------------------------
'''

dist = commons.dist
c = commons.c
RfC = commons.RfC
CfR = commons.CfR
SparseToCSR = commons.SparseToCSR
CSRToSparse = commons.CSRToSparse
# csr = SparseToCSR(A1)
# AA = CSRToSparse(csr)
# np.allclose(csr.toarray(), tf.sparse.to_dense(AA))

def MapSquareToCircle(sqside, Lx, Ly):
    radius = sqside/2
    Lx = (Lx - radius)/radius # map to [-1,1]
    Ly = (Ly - radius)/radius # map to [-1,1]
    X = Lx*np.sqrt(1-0.5*Ly*Ly)
    Y = Ly*np.sqrt(1-0.5*Lx*Lx)
    X = radius + radius * X
    Y = radius + radius * Y
    return X, Y


def ShrinkCircle(sqside, ratio, Rx, Ry):
    radius = sqside/2
    Rx = ratio * (Rx - radius)
    Ry = ratio * (Ry - radius)
    return Rx + radius, Ry + radius


class WCNGraph():
    '''
    class of custom Graph designed for WCN
        numNode = number of vertices (typically L+K in our problem)
        numEdgeTypes = number of types of edges (typicall 3, cooperative, intra and inter cell interference type)
        As = binary Sparse Adjacency matrix (can be None), corresponding to each edge type
        V = feature aided Node matrix, (shape = numNode * #node_features)
        E = feature aided Edge matrix, (shape = numEdges * #edge_features) for each edge type
    '''
    def __init__(self, numNodes, numEdgetypes, As, V, E):
        self.numNode = numNodes
        self.numEdgetypes = numEdgetypes
        self.As = As 
        self.V = V
        self.E = E

class genData():
    '''
    Generates users distributed uniformly in a square of side (self.params.sqside)
        Ns : number of data points
        Nss : number of realisations for delta and H per data point
        Nsss : number of realisations for small scale fading channel
        rhoK : density of users, uniformly from [mean, half-width]
        rhoL : density of APs, uniformly from [mean, half-width]
        sqside : length of square of graph
        sqrep : replicating block size (for very large graphs, generation is done via replicating this sized block)
        threshold_coop : cooperative distance threshold (unit same as sqside), \
            this is changed to min(dkl, )
        threshold_inf : interference distance threshold (unit same as sqside), \
            this is typically > threshold_inf
        coopMax : Max users served by an AP
        N : antennas at AP
        M : antennas at UE
        ple : path loss exponent (Urban; 2.7 - 3.5), (SubUrban; 3-5)\
            2/np.log10(threshold_inf/threshold_coop) for 2 orders of difference
            # 2.86 -> threshold_inf/threshold_coop = 5
            # 3.32 -> threshold_inf/threshold_coop = 4
            # 4.19 -> threshold_inf/threshold_coop = 3
            # 5.02 -> threshold_inf/threshold_coop = 2.5
        tauMax : max path gain (taken as (4)**(-ple)), \
            models minimum distance between AP and user.
    '''
    def __init__(self, Ns=10, Nss=50, Nsss=500, d_min=4):
        self.params = myutils.Empty();
        self.params.Ns = Ns;
        self.params.Nss = Nss;
        self.params.Nsss = Nsss;
        self.params.rhoK = [0.1, 0.03];
        self.params.rhoL = [0.2, 0.03];
        self.params.sqside = 100;
        self.params.sqrep = 100;
        self.params.threshold_coop = 12.5;
        self.params.threshold_inf = 50;  # as d_i = d_c (10**(2/3.32)) ~ 4 d_c
        self.params.p0 = 80;
        self.params.coopMax = commons.coopMax;
        self.params.N = commons.N;
        self.params.M = commons.M;
        self.params.Ns = commons.Ns;
        self.params.ple = 3.32;
        self.params.d_min = d_min
        self.params.noise_sigma = [1e-4,2e-6]
        self.debug = myutils.Empty();

    def get_tau_kl(self, dkl):
        self.params.tauMax = self.params.d_min**(-self.params.ple/2);
        tau_kl = np.clip(dkl**(-self.params.ple/2), a_min=0, a_max=self.params.tauMax)
        return self.params.p0*tau_kl

    def sampleOneArea(self, sqside, ensureConsistency=True):
        rhoK = self.params.rhoK;
        rhoL = self.params.rhoL;
        threshold_coop = self.params.threshold_coop;
        d_min = np.inf
        _t = time.time(); _check=True
        while d_min > threshold_coop:
            K = int(sqside*np.random.uniform(low=rhoK[0]-rhoK[1], high=rhoK[0]+rhoK[1]));
            L = int(sqside*np.random.uniform(low=rhoL[0]-rhoL[1], high=rhoL[0]+rhoL[1]));
            Kx = np.random.uniform(low=0,high=sqside,size=K)
            Ky = np.random.uniform(low=0,high=sqside,size=K)
            Lx = np.random.uniform(low=0,high=sqside,size=L)
            Ly = np.random.uniform(low=0,high=sqside,size=L)
            dkl = [[dist([Kx[k],Ky[k]],[Lx[l],Ly[l]]) for k in range(K)] for l in range(L)]
            dkl = np.array(dkl).T
            d_min = np.max(np.min(dkl, axis=1));
            if time.time() - _t > 1 and _check:
                print('WARNING :: deciding AP is taking time')
                _check = False;
            if not ensureConsistency:
                break;
        return [(time.time() - _t) if ensureConsistency else -1, L,K,Lx,Ly,Kx,Ky,dkl]

    '''
    sample a WCN (wireless cooperative network) 
        input:
            show : plots a graph of APs and Users
            ensureConsistency : ensures the following -> \
                            1. forall k, there exists an AP within threshold_coop
        returns
            [$,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK] = 
                # $ time
                # L : users
                # K : APs
                # coordinate X of users
                # coordinate Y of users
                # coordinate X of APs
                # coordinate Y of APs
                # distances of dimension K x L (unit same as sqside)
    '''
    def sampleWCN(self, ensureConsistency=True):
        rhoK = self.params.rhoK;
        rhoL = self.params.rhoL;
        sqside = self.params.sqside;
        sqrep = self.params.sqrep;
        threshold_coop = self.params.threshold_coop;
        Nsss = self.params.Nsss;
        N,M = self.params.N, self.params.M;
        noise_sigma = self.params.noise_sigma
        _t = time.time(); _check=True
        if sqside <= sqrep:
            [_time,L,K,Lx,Ly,Kx,Ky,dkl] = self.sampleOneArea(sqside,ensureConsistency=ensureConsistency)
        else:
            assert sqside%sqrep == 0;
            _time = 0; L=0; K=0; Lx=[]; Ly=[]; Kx=[]; Ky=[]; dkl=None;
            for _x in range(sqside//sqrep):
                for _y in range(sqside//sqrep):
                    [__time, _L,_K,_Lx,_Ly,_Kx,_Ky,_dkl] = self.sampleOneArea(sqrep, ensureConsistency=ensureConsistency)
                    _time = _time + __time;
                    L=L+_L;
                    K=K+_K;
                    Lx.extend(_Lx+_x*sqrep);
                    Ly.extend(_Ly+_y*sqrep);
                    Kx.extend(_Kx+_x*sqrep);
                    Ky.extend(_Ky+_y*sqrep);
        Lx, Ly, Kx, Ky = np.array(Lx), np.array(Ly), np.array(Kx), np.array(Ky)
        Lx, Ly = MapSquareToCircle(sqside, Lx, Ly)
        Kx, Ky = MapSquareToCircle(sqside, Kx, Ky)
        dkl = [[dist([Kx[k],Ky[k]],[Lx[l],Ly[l]]) for k in range(K)] for l in range(L)]; dkl = np.array(dkl).T
        # shrinking to circle might increase d_min
        d_min = np.max(np.min(dkl, axis=1));
        if d_min > threshold_coop:
            goodUsers = np.min(dkl,axis=1)<threshold_coop
            if np.sum(goodUsers)/K > 0.95:
                self.debug.goodUsers = np.sum(goodUsers)/K
                K = np.sum(goodUsers)
                Kx = Kx[goodUsers]
                Ky = Ky[goodUsers]
                dkl = [[dist([Kx[k],Ky[k]],[Lx[l],Ly[l]]) for k in range(K)] for l in range(L)]; dkl = np.array(dkl).T
                d_min = np.max(np.min(dkl, axis=1));
            else:
                ratio = threshold_coop/(d_min+0.01)
                self.debug.Lx = np.copy(Lx); self.debug.Ly = np.copy(Ly); self.debug.Kx = np.copy(Kx); self.debug.Ky = np.copy(Ky); 
                self.debug.ratio = ratio
                Lx, Ly = ShrinkCircle(sqside, ratio, Lx, Ly)
                Kx, Ky = ShrinkCircle(sqside, ratio, Kx, Ky)
                dkl = [[dist([Kx[k],Ky[k]],[Lx[l],Ly[l]]) for k in range(K)] for l in range(L)]; dkl = np.array(dkl).T
                d_min = np.max(np.min(dkl, axis=1));
        assert d_min < threshold_coop;
        H_kl = (1/np.sqrt(2)) * (np.random.randn(*[Nsss,K,L,N,M]) + 1j * np.random.randn(*[Nsss,K,L,N,M]))
        sigmaK = np.random.uniform(size=[Nsss,K],low=noise_sigma[0]-noise_sigma[1], high=noise_sigma[0]+noise_sigma[1])
        _time = (time.time() - _t) if ensureConsistency else -1;
        return [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK]
    
    ''' 
        Appends the wcnparams with sets (all are K x L), each call assigns a random delta_kl 
            from U_l the set of serviceable APs
        D_l : "randomly" chose {1,...,params.coopMax} users within params.threshold_coop
        I_k : all users within params.threshold_inf
        tau_kl : np.clip(dkl**(-ple), a_min=0, a_max=tauMax)
    '''
    def getSystemStatefromWCN(self, wcnParams, override=False):
        [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK] = wcnParams;
        if _time < 0 and (not override):
            raise Exception('Possibly Inconsistent WCN, use override=True')
        threshold_coop = self.params.threshold_coop;
        threshold_inf = self.params.threshold_inf;
        coopMax = self.params.coopMax
        tau_kl = self.get_tau_kl(dkl)
        I_k = dkl < threshold_inf
        min_C_k = -np.inf
        _t = time.time();
    
        _gotThrough = False
        while not _gotThrough:
            D_l = dkl < threshold_coop
            for l in range(L):
                idx_in = np.where(D_l[:,l])[0]
                if len(idx_in) != 0:
                    D_l[:,l] = False
                    _permidx = np.random.permutation(len(idx_in))[:int(np.random.rand()*(coopMax+1))]
                    D_l[:,l][idx_in[_permidx]] = True
            # find users with no APs and put their D_l as 1
            _noAPu = np.where(np.sum(D_l, axis=1) == 0)[0]; #print(len(_noAPu)/K)
            _D_l = dkl < threshold_coop
            for _k in _noAPu:
                _AP_within_k = np.where(_D_l[_k])[0]
                D_l[_k, _AP_within_k[np.random.randint(len(_AP_within_k))]] = True
            # find APs which are not serving any and set I_k[:,l] = False
            # Also remove cooperative links from I_k
            I_k[:,np.where(np.sum(D_l,axis=0)==0)[0]] = False;
            I_k[np.where(D_l)] = False;
            _gotThrough = (np.max(np.sum(D_l, axis=0)) < coopMax) and (np.min(np.sum(D_l, axis=1)) > 0)
            if (time.time() - _t) > 5:
                raise Exception('Unable to find a suitable configuration')

        _time = _time + (time.time() - _t)
        return [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK,  D_l,I_k,tau_kl]

    '''
        Plots the WCN
        bold lines for cooperative and dashed lines for interference links
    '''
    def plotWCN(self, wcnParams):
        try:
            [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK,  D_l,I_k,tau_kl] = wcnParams
        except:
            raise Exception('Check if wcnParams have been processed by self::getSystemStatefromWCN()')
        plt.figure()
        for i in range(len(Kx)):
            plt.text(Kx[i], Ky[i], str(i))
        for i in range(len(Lx)):
            plt.text(Lx[i], Ly[i], str(i))
        for k in range(K):
            for l in range(L):
                if D_l[k,l]:
                    plt.arrow(Lx[l],Ly[l],Kx[k]-Lx[l],Ky[k]-Ly[l], length_includes_head=True, head_width=0, width=0.01);
                if I_k[k,l]:
                    plt.arrow(Lx[l],Ly[l],Kx[k]-Lx[l],Ky[k]-Ly[l], length_includes_head=True, head_width=0, width=0.005, linestyle='-.', alpha=0.2, color='blue');
        plt.scatter(Kx,Ky,marker='o', s=100)
        plt.scatter(Lx,Ly,marker='^', s=80)
        # plt.show()

    ''' 
        Gets a processed WCN graph and returns a WCNGraph, additionally samples
        returns
        A1   : cooperative adjacency graph (Sparse)
        A1F1 : cooperative adjacency graph edge features H_kl
        A2F2 : cooperative adjacency graph long scale fading
        A2 : intra-cell interference
        A3 : inter-cell interference
    '''
    def getNodeLevelGraphFromWCN(self, wcnParams, load_in_gpu=False):
        try:
            [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK,  D_l,I_k,tau_kl] = wcnParams
        except:
            raise Exception('Check if wcnParams have been processed by self::getSystemStatefromWCN()')
        numNodes = L + K
        numEdgetypes = 3
        As = D_l
        V = None
        E = None;
        _t = time.time()
        # make graph, C:Cooperative, A:Intracell, E:Intercell
        # 1. Creating Cooperative Edges
        # obtain from and to nodes from D_l
        # give new identities to APs according to edge number\
        #               as we replicate AP if serving multiple Users
        # Rx id is changed to #Edges + idxtoNode, fromNode = np.where(D_l);
        toNode, fromNode = np.where(D_l);
        nE = len(fromNode)
        fromNodeNew = nAPid = np.arange(nE)
        toNodeNew = toNode + nE;
        dictOldtoNew = {l:np.where(fromNode==l)[0] for l in fromNode}
        A1F1 = H_kl[:,toNode, fromNode,:,:]
        A1F2 = tau_kl[toNode, fromNode]
        # 2. Intracell interference
        # Find which nodes serve multiple users
        # Say AP 3 serves ul=[1,8]; np.where(D_l[:,3]). 
        # New ID of AP 3 is nl = [1,13]; np.where(fromNode==3), 
        # nl=[1,13] serves ul=[1,8] respectively by design
        # We need edges from set (nl - nl[ik]) to ul[ik] (+ nE), for ik in ul
        _APwmoreUE = np.where(np.sum(D_l, axis=0)>1)[0]
        if len(_APwmoreUE)>0:
            fromNodeNew_A = []
            toNodeNew_A = []
            for l in _APwmoreUE:
                ul = np.where(D_l[:,l])[0]
                nl = np.where(fromNode==l)[0]
                for ik in range(len(ul)):
                    inf_l = np.setdiff1d(nl, nl[ik])
                    fromNodeNew_A.extend(inf_l);
                    toNodeNew_A.extend([ul[ik]+nE for _ in range(len(inf_l))]);
            fromNodeNew_A, toNodeNew_A = np.array(fromNodeNew_A), np.array(toNodeNew_A)
            A2F1 = H_kl[:,toNodeNew_A-nE, fromNode[fromNodeNew_A],:,:]
            A2F2 = tau_kl[toNodeNew_A-nE, fromNode[fromNodeNew_A]]
        # Intercell Interference
        # Obtain from and to nodes from I_k, np.where(I_k)
        # say AP 3, interferes with UE 7, 
        # and AP 3 serves UE [1, 8] with ID _infidx=[1,17], np.where(fromNode==3)
        # Then add edges from [1,17] to AP 7
        # add edges from all [fromNodeNew]
        toNode_E, fromNode_E = np.where(I_k)
        fromNodeNew_E = []
        toNodeNew_E = []
        for il, l in enumerate(fromNode_E):
            _infidx = np.where(fromNode==l)[0]
            fromNodeNew_E.extend(_infidx)
            toNodeNew_E.extend([toNode_E[il]+nE for _ in range(len(_infidx))]);
        fromNodeNew_E, toNodeNew_E = np.array(fromNodeNew_E), np.array(toNodeNew_E)
        A3F1 = H_kl[:,toNodeNew_E-nE, fromNode[fromNodeNew_E],:,:]
        A3F2 = tau_kl[toNodeNew_E-nE, fromNode[fromNodeNew_E]]
        if load_in_gpu:
            A1 = tf.sparse.SparseTensor(indices=np.array([fromNodeNew, toNodeNew]).T, values=np.ones(nE), dense_shape=[nE+K,nE+K])
            if len(_APwmoreUE)>0:
                A2 = tf.sparse.SparseTensor(indices=np.array([fromNodeNew_A, toNodeNew_A]).T, values=np.ones(len(fromNodeNew_A)), dense_shape=[nE+K,nE+K])
            A3 = tf.sparse.SparseTensor(indices=np.array([fromNodeNew_E, toNodeNew_E]).T, values=np.ones(len(fromNodeNew_E)), dense_shape=[nE+K,nE+K])
        else:
            with tf.device('/cpu:0'):
                A1 = tf.sparse.SparseTensor(indices=np.array([fromNodeNew, toNodeNew]).T, values=np.ones(nE), dense_shape=[nE+K,nE+K])
                if len(_APwmoreUE)>0:
                    A2 = tf.sparse.SparseTensor(indices=np.array([fromNodeNew_A, toNodeNew_A]).T, values=np.ones(len(fromNodeNew_A)), dense_shape=[nE+K,nE+K])
                A3 = tf.sparse.SparseTensor(indices=np.array([fromNodeNew_E, toNodeNew_E]).T, values=np.ones(len(fromNodeNew_E)), dense_shape=[nE+K,nE+K])
        _time = _time + (time.time() - _t)
        if len(_APwmoreUE)>0:
            return [_time, L, K, fromNode, nE, sigmaK, [A1,A1F1,A1F2], [A2,A2F1,A2F2], [A3,A3F1,A3F2]]
        else:
            return [_time, L, K, fromNode, nE, sigmaK, [A1,A1F1,A1F2], [None,None,None], [A3,A3F1,A3F2]]
    
    '''
        Plots the graph
        provides AP WCN ID, AP graph ID, APs, cooperative (bold), intercell interference (bold black), \
            intracell interference (dotted red), intercell interference (dotted black), UEs, UE graph ID, UE WCN ID
    '''
    def plotNodeLevelGraph(self, gParams):
        [_time, L, K, fromNode, nE, sigmaK, [A1,A1F1,A1F2], [A2,A2F1,A2F2], [A3,A3F1,A3F2]] = gParams     
        plt.figure()
        fromNodeNew = np.arange(nE)
        mid = 5/2*(len(fromNodeNew) - K)
        # plt.text(Lx[i], Ly[i], str(i))
        _gKx = 5*np.ones(K)
        _gKy = mid + 5*np.arange(K)
        _gLx = np.zeros(len(fromNodeNew))
        _gLy = 5*np.arange(len(fromNodeNew))
        plt.scatter(_gKx, _gKy, marker='o')
        plt.scatter(_gLx, _gLy, marker='^')
        for i in range(K):
            plt.text(_gKx[i]+1, _gKy[i], str(i+nE))
            plt.text(_gKx[i]+2, _gKy[i], str(i))
        for i in range(len(fromNodeNew)):
            plt.text(_gLx[i]-1, _gLy[i], str(fromNodeNew[i]))
            plt.text(_gLx[i]-2, _gLy[i], str(fromNode[fromNodeNew[i]]))
        for i in range(len(A1.values)):
            iln = A1.indices[i][0].numpy()
            il = fromNode[iln]
            ik = A1.indices[i][1] - nE
            plt.arrow(_gLx[iln], _gLy[iln], _gKx[ik]-_gLx[iln], _gKy[ik]-_gLy[iln], length_includes_head=True, head_width=0, width=0.01);
        for i in range(len(A2.values)):
            iln = A2.indices[i][0].numpy()
            il = fromNode[iln]
            ik = A2.indices[i][1] - nE
            plt.arrow(_gLx[iln], _gLy[iln], _gKx[ik]-_gLx[iln], _gKy[ik]-_gLy[iln], length_includes_head=True, head_width=0, width=0.005, color='red', linestyle='-.', alpha=0.3);
        for i in range(len(A3.values)):
            iln = A3.indices[i][0].numpy()
            il = fromNode[iln]
            ik = A3.indices[i][1] - nE
            plt.arrow(_gLx[iln], _gLy[iln], _gKx[ik]-_gLx[iln], _gKy[ik]-_gLy[iln], length_includes_head=True, head_width=0, width=0.005, color='blue', linestyle='-.', alpha=0.3);
        [x1,x2,y1,y2] = plt.axis() 
        plt.axis([x1-2,x2+2,y1,y2]) 
        


class outer_genData(genData):
    '''
        Plots the Outer WCN
        bold lines for cooperative and dashed lines for interference links
    '''
    def outer_plotWCN(self, wcnParams):
        try:
            [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK] = wcnParams
            S_CA = dkl < self.params.threshold_coop  # super set cooperative adjacency
            S_IA = dkl < self.params.threshold_inf   # super set interference adjacency
        except:
            raise Exception('wcnParams shouldnt be processed by self::getSystemStatefromWCN()')
        plt.figure()
        for i in range(len(Kx)):
            plt.text(Kx[i], Ky[i], str(i))
        for i in range(len(Lx)):
            plt.text(Lx[i], Ly[i], str(i))
        for k in range(K):
            for l in range(L):
                if S_CA[k,l]:
                    plt.arrow(Lx[l],Ly[l],Kx[k]-Lx[l],Ky[k]-Ly[l], length_includes_head=True, head_width=0, width=0.01);
                if S_IA[k,l]:
                    plt.arrow(Lx[l],Ly[l],Kx[k]-Lx[l],Ky[k]-Ly[l], length_includes_head=True, head_width=0, width=0.005, linestyle='-.', alpha=0.2, color='blue');
        plt.scatter(Kx,Ky,marker='o', s=100)
        plt.scatter(Lx,Ly,marker='^', s=80)


    def outer_getNodeLevelGraphFromWCN(self, wcnParams, load_in_gpu=False):
        try:
            [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK] = wcnParams
            S_CA = dkl < self.params.threshold_coop  # super set cooperative adjacency
            S_IA = dkl < self.params.threshold_inf   # super set interference adjacency
        except:
            raise Exception('wcnParams shouldnt be processed by self::getSystemStatefromWCN()')
        tau_kl = self.get_tau_kl(dkl)
        numNodes = L + K
        numEdgetypes = 2
        _t = time.time()
        # make graph, C:Cooperative, I:Interference
        # 1. Creating Cooperative Edges
        # obtain from and to nodes from S_CA
        # give new identities to APs according to edge number\
        #               as we replicate AP if serving multiple Users
        # Rx id is changed to # Edges + idxtoNode, fromNode = np.where(D_l);
        # IDs do not signify anything and are just for accumulation purpose
        toNode_C, fromNode_C = np.where(S_CA);
        nE = len(fromNode_C)
        fromNodeNew_C = nAPid = np.arange(nE)
        toNodeNew_C = toNode_C + nE;
        dictOldtoNew = {l:np.where(fromNode_C==l)[0] for l in fromNode_C}
        F1_C = H_kl[:,toNode_C, fromNode_C,:,:]
        F2_C = tau_kl[toNodeNew_C-nE, fromNode_C[fromNodeNew_C]]
        # 2. Creating Interference Edges
        # Obtain from and to nodes from S_IA
        # say AP 3, interferes with UE 7, 
        # and AP 3 serves UE [1, 8] with ID _infidx=[1,17], np.where(fromNode==3)
        # Then add edges from [1,17] to AP 7
        # add edges from all [fromNodeNew]
        toNode_I, fromNode_I = np.where(S_IA)
        fromNodeNew_I = []
        toNodeNew_I = []
        for il, l in enumerate(fromNode_I):
            _infidx = np.where(fromNode_C==l)[0]
            fromNodeNew_I.extend(_infidx)
            toNodeNew_I.extend([toNode_I[il]+nE for _ in range(len(_infidx))]);
        fromNodeNew_I, toNodeNew_I = np.array(fromNodeNew_I), np.array(toNodeNew_I)
        F1_I = H_kl[:,toNodeNew_I-nE, fromNode_C[fromNodeNew_I],:,:]
        F2_I = tau_kl[toNodeNew_I-nE, fromNode_C[fromNodeNew_I]]
        if load_in_gpu:
            SSCA = tf.sparse.SparseTensor(indices=np.array([fromNodeNew_C, toNodeNew_C]).T, values=np.ones(nE), dense_shape=[nE+K,nE+K])
            SSIA = tf.sparse.SparseTensor(indices=np.array([fromNodeNew_I, toNodeNew_I]).T, values=np.ones(len(fromNodeNew_I)), dense_shape=[nE+K,nE+K])
        else:
            with tf.device('/cpu:0'):
                SSCA = tf.sparse.SparseTensor(indices=np.array([fromNodeNew_C, toNodeNew_C]).T, values=np.ones(nE), dense_shape=[nE+K,nE+K])
                SSIA = tf.sparse.SparseTensor(indices=np.array([fromNodeNew_I, toNodeNew_I]).T, values=np.ones(len(fromNodeNew_I)), dense_shape=[nE+K,nE+K])
        _time = _time + (time.time() - _t)
        return [_time, L, K, fromNode_C, nE, sigmaK, [SSCA,F1_C,F2_C], [SSIA,F1_I,F2_I]]
    
    def outer_plotNodeLevelGraph(self, gParams):
        [_time, L, K, fromNode_C, nE, sigmaK, [SSCA,F1_C,F2_C], [SSIA,F1_I,F2_I]] = gParams     
        plt.figure()
        fromNodeNew = np.arange(nE)
        mid = 5/2*(len(fromNodeNew) - K)
        # plt.text(Lx[i], Ly[i], str(i))
        _gKx = 5*np.ones(K)
        _gKy = mid + 5*np.arange(K)
        _gLx = np.zeros(len(fromNodeNew))
        _gLy = 5*np.arange(len(fromNodeNew))
        plt.scatter(_gKx, _gKy, marker='o')
        plt.scatter(_gLx, _gLy, marker='^')
        for i in range(K):
            plt.text(_gKx[i]+1, _gKy[i], str(i+nE))
            plt.text(_gKx[i]+2, _gKy[i], str(i))
        for i in range(len(fromNodeNew)):
            plt.text(_gLx[i]-1, _gLy[i], str(fromNodeNew[i]))
            plt.text(_gLx[i]-2, _gLy[i], str(fromNode_C[fromNodeNew[i]]))
        for i in range(len(SSCA.values)):
            iln = SSCA.indices[i][0].numpy()
            il = fromNode_C[iln]
            ik = SSCA.indices[i][1] - nE
            plt.arrow(_gLx[iln], _gLy[iln], _gKx[ik]-_gLx[iln], _gKy[ik]-_gLy[iln], length_includes_head=True, head_width=0, width=0.01);
        for i in range(len(SSIA.values)):
            iln = SSIA.indices[i][0].numpy()
            il = fromNode_C[iln]
            ik = SSIA.indices[i][1] - nE
            plt.arrow(_gLx[iln], _gLy[iln], _gKx[ik]-_gLx[iln], _gKy[ik]-_gLy[iln], length_includes_head=True, head_width=0, width=0.005, color='red', linestyle='-.', alpha=0.3);
        [x1,x2,y1,y2] = plt.axis() 
        plt.axis([x1-2,x2+2,y1,y2]) 

    def provideCooperativeClusters(self, wcnParams, gParams):
        [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK] = wcnParams
        [_time, L, K, fromNode_C, nE, sigmaK, [SSCA,F1_C,F2_C], [SSIA,F1_I,F2_I]] = gParams

        def getSparseDeltaVector(inputs):
            D_l, SSCA, fromNode_C = inputs
            _t, _f = np.where(D_l);
            nE = SSCA.indices.shape[0]
            delta = np.zeros(nE, dtype=ftype)
            _idxs = np.copy(SSCA.indices)
            _idxs[:,1] =  _idxs[:,1] - nE
            _idxs[:,0] =  fromNode_C
            for _ in range(len(_t)):
                _delidx = np.where(np.all([_f[_], _t[_]] == _idxs, axis=1))[0][0]
                delta[_delidx] = 1.
            return delta

        tau_kl = self.get_tau_kl(dkl)
        threshold_coop = self.params.threshold_coop;
        threshold_inf = self.params.threshold_inf;
        coopMax = self.params.coopMax
        D_l_master = dkl < threshold_coop
        D_l = np.zeros(shape=dkl.shape, dtype='bool')

        # 1.1 appoint Master AP for user k as argmax_l tau_kl, if l can accomodate, else choose second best etc and so on
        for k in range(K):
            sortedTauK = np.argsort(-tau_kl[k])[:np.sum(D_l_master[k,:])];
            _lidx = 0;
            l = sortedTauK[_lidx];
            while True:
                if np.sum(D_l[:,l]) < coopMax:
                    D_l[k, l] = True;
                    break;
                else:
                    _lidx += 1;
                    if _lidx >= np.sum(D_l_master[k,:]):
                        raise Exception('Configuration Error : No AP free')
                    else:                
                        l = sortedTauK[_lidx];

        _D_l = np.copy(D_l)
        Delta_Strongest = getSparseDeltaVector([D_l, SSCA, fromNode_C])
        assert np.allclose(np.sum(Delta_Strongest),K)

        # 1.2 Other APs within coop distance may want to cooperate randomly
        for l in range(L):
            idx_in = np.where(D_l_master[:,l])[0]
            if len(idx_in) != 0:
                _available_slots = coopMax - np.sum(D_l[:,l]);
                # find remaining users within coopDistance, D_l_master[:,l] == True and D_l[:,l] == False
                _availabe_Users = np.where(np.logical_and(np.logical_not(D_l[:,l]), D_l_master[:,l]))[0]
                if len(_availabe_Users) > 0:
                    _permidx = np.random.permutation(len(_availabe_Users))[:int(np.random.rand()*(_available_slots+1))]
                    D_l[:,l][_availabe_Users[_permidx]] = True

        Delta_Random = getSparseDeltaVector([D_l, SSCA, fromNode_C])

        # 1.3 All APs within coop distance may want to cooperate
        D_l = _D_l
        for l in range(L):
            idx_in = np.where(D_l_master[:,l])[0]
            if len(idx_in) != 0:
                _available_slots = coopMax - np.sum(D_l[:,l]);
                # find remaining users within coopDistance, D_l_master[:,l] == True and D_l[:,l] == False
                _availabe_Users = np.where(np.logical_and(np.logical_not(D_l[:,l]), D_l_master[:,l]))[0]
                if len(_availabe_Users) > 0:
                    # randomly choose as many users as you can
                    _permidx = np.random.permutation(len(_availabe_Users))[:_available_slots]
                    D_l[:,l][_availabe_Users[_permidx]] = True

        Delta_All = getSparseDeltaVector([D_l, SSCA, fromNode_C])
        return Delta_Strongest, Delta_Random, Delta_All

        


# if __name__ == '__main__':
#     g = genData()
#     g.params.sqside=100; g.params.Nsss=1
#     # g.params.rhoL = [0.5, 0.03];
#     # g.params.rhoK = [0.3, 0.03];
#     wcnParams0 = g.sampleWCN()
#     wcnParams1 = g.getSystemStatefromWCN(wcnParams0); 
#     gParams = g.getNodeLevelGraphFromWCN(wcnParams1)
#     [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK] = wcnParams0
#     [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK,  D_l,I_k,tau_kl] = wcnParams1
#     [_time, L, K, fromNode, nE, sigmaK, [A1,A1F1,A1F2], [A2,A2F1,A2F2], [A3,A3F1,A3F2]] = gParams

#     g.plotWCN(wcnParams1);
#     g.plotNodeLevelGraph(gParams);
#     plt.show()


if __name__ == '__main__':
    g = outer_genData()
    g.params.sqside=100;
    wcnParams = g.sampleWCN()
    gParams = g.outer_getNodeLevelGraphFromWCN(wcnParams)
    g.outer_plotWCN(wcnParams);
    g.outer_plotNodeLevelGraph(gParams);
    plt.show()
    [_time,L,K,Lx,Ly,Kx,Ky,dkl,H_kl,sigmaK] = wcnParams
    [_time, L, K, fromNode_C, nE, sigmaK, [SSCA,F1_C,F2_C], [SSIA,F1_I,F2_I]] = gParams
    self = g;
    Delta_Strongest, Delta_Random, Delta_All = g.provideCooperativeClusters(wcnParams, gParams)

# given a D_l get delta variable



