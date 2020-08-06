import numpy as np
import scipy.special
from math import log, floor

# Auxiliary functions 
#################################################################################################
def shift_polynomial(coeffs, shift):
    """
    Return coefficients of a shifted polynomial
    P(x) = p(x+shift)
    Args:
    coeffs: list of float
    coefficients of p
    shift: float
    right shift of polynomial
    Returns:
    coefficients of P
    """
    length = len(coeffs)
    coeffs_shift = np.zeros(length)
    for i in range(length):
        for j in range(i+1):
            coeffs_shift[j] += coeffs[i] * (shift**(i-j)) * scipy.special.binom(i, j)
    return coeffs_shift


def hist_to_fin(hist):
    """
    Return a fingerprint from histogram
    """
    fin = {}
    for freq in hist:
        if freq in fin:
            fin[freq] += 1
        else:
            fin[freq] = 1
    return fin.items()

def samples_to_histogram(samples, n):
    """
    Return a histogram from a list of smaples, n = universe size
    """
    histogram = [0]*n
    for s in samples:
        histogram[s] += 1
    return histogram

##################################################################################################


def poly_g(L, interval, num):
    
    """
    Returns coefficient of scaled and shifted chebyshev polynomial according to eq (21) in Wu Yang
    
    Args:
    L: degree of polynomial
    interval: interval to fit polynomial to. [l,r] in Wu Yang paper
    num: number of smaples
    
    """
    
    # get interval and chebyshev coeffs
    left_end, right_end = interval
    cheb = np.polynomial.chebyshev.Chebyshev.basis(L)
    cheb_coeffs = np.polynomial.chebyshev.cheb2poly(cheb.coef)
    
    # calculate shift and scaling
    shift = (right_end + left_end) / (right_end - left_end)
    scaling = 2. / (num*right_end-num*left_end)
    

    # shift polynomial
    a_coeffs = shift_polynomial(cheb_coeffs, -shift)
    g_coeffs = -a_coeffs / a_coeffs[0]
    g_coeffs[0] = 0
    
    # scale polynomial
    for j in range(1, L+1):
        for i in range(1, j+1):
            g_coeffs[j] *= (i*scaling)
        g_coeffs[j] += 1
        
    return g_coeffs
    


def estimator(L, hist, interval):
    """
    Returns estimate of support given args
    
    Args:
    L: degree of polynomial
    hist: histogram of samples
    interval: interval we want to approximate support in

    """
    
    # converge histogram to fingerprint
    fin = hist_to_fin(hist)
    num = sum(hist)
    if (num == 0):
        return 0
    
    # get coefficient of polynomial used for estimation
    g_coeffs = poly_g(L, interval, num)
    
    # get estimate by looping over fingerprint
    supp_estimate = 0.0
    for freq, cnt in fin:
        if freq > L: # plug-in
            supp_estimate += 1*cnt
        else: # polynomial
            supp_estimate += g_coeffs[freq]*cnt
    return supp_estimate

def WuYangEstimator(n, hist, L = None, left_end = None, right_end = None):
    """
    Returns estimate of support according to Wu Yang paper specifications
    
    Args:
    n: estimate of universe size
    hist: histogram of samples
    
    Optional Args: if not given set to Wu Yang paper specification
    L: degree of polynomial
    left_end: left end of interval we want to esimate support in

    """
    
    # set Wu Yang parameters
    num = sum(hist)
    if L == None:
        L = floor(0.45*log(n))
    if left_end == None:
        left_end = 1./n
    if right_end == None:
        right_end = 0.5*log(n)/num
    
    # call estimator
    return estimator(L, hist, [left_end, right_end])


def OracleEstimator(histograms, intervals, L):
    """
    Returns sum of supports for each interval
    
    Args:
    histograms: list of histograms, one for each interval
    intervals: list of intervals where each interval is of the form [left_end, right_end]
    L: degree of polynomial

    """
    supp_estimate = 0.0
    
    # loop over histogram and intervals and add up support 
    for i, hist in enumerate(histograms):
        curr_interval = intervals[i]
        supp_estimate += estimator(L, hist, curr_interval)
    return supp_estimate
    
# Test usage
if __name__ == '__main__':
	n = 10000
	actual_supp = 100
	num_samples = 65
	samples = np.random.randint(low=0, high=actual_supp, size = num_samples)
	hist = samples_to_histogram(samples, n)
	print(WuYangEstimator(n, hist))