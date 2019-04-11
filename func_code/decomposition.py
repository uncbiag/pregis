import numpy as np
import time
import gc

def pca_GPU(test_img, mean_brain, Beta, BetaT, _gamma, correction, verbose):
    import algo as algo
    t_begin = time.clock()
    l,m,n = test_img.shape
    D = test_img - mean_brain
    
    L_tmp, T, alpha = algo.decompose(D, Beta, BetaT, _gamma, verbose)
    for i in range(correction):
        sum_alpha_beta = np.dot(Beta, alpha)
        D_prime = D + L_tmp - sum_alpha_beta.reshape(l,m,n)
        L_tmp, T, alpha = algo.decompose(D_prime, Beta, BetaT , _gamma, verbose)

    gc.collect()
    t_end = time.clock()
    t_elapsed = t_end - t_begin
    print('Decomposition takes: %f seconds' %t_elapsed)

    L = D - T + mean_brain
    return (L, T, alpha)


def pca_CPU(test_img, mean_brain, Beta, _gamma, correction, verbose):
    import algo_cpu as algo
    t_begin = time.clock()
    l,m,n = test_img.shape
    D = test_img - mean_brain

    L_tmp, T, alpha = algo.decompose(D, Beta, _gamma, verbose)
    for i in range(correction):
        sum_alpha_beta = np.dot(Beta, alpha)
        D_prime = D + L_tmp - sum_alpha_beta.reshape(l,m,n)
        L_tmp, T, alpha = algo.decompose(D_prime, Beta, _gamma, verbose)

    gc.collect()
    t_end = time.clock()
    t_elapsed = t_end - t_begin
    print('Decomposition takes: %f seconds' %t_elapsed)

    L = D - T + mean_brain
    return (L, T, alpha)

  
