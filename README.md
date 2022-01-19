# Cell_free_mimo

Creates and plots a cell free MIMO with editable parameters as follows
The equivalent graph is created based on the (submitted) paper

'Two-Timescale Probabilistic Clustering and Distributed Beamforming for IoT Cell-free MIMO Systems using Graph Neural Networks'
Nilesh Kumar Jha, Graduate Student member, IEEE, Ye Xue, Graduate Student member, IEEE, and Vincent K. N. Lau, Fellow, IEEE

Editable Parameters
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
