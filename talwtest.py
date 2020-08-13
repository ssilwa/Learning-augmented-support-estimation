import numpy as np
import scipy.special

from math import log, floor
import sys
import warnings

import wuyang
import CollisionAlgo

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


def OracleEstimator(histograms, intervals, L, n, n_samples, base):
    """
    Returns sum of supports for each interval
    
    Args:
    histograms: list of histograms, one for each interval
    intervals: list of intervals where each interval is of the form [left_end, right_end]
    L: degree of polynomial

    """
    supp_estimate = 0.0
    supp_naive_estimate = 0.0
    # loop over histogram and intervals and add up support 
    for i, hist in enumerate(histograms):
        use_naive = True
        curr_interval = intervals[i]
        #L_curr = min(L, int(curr_interval[1]*n*1./(base/2.)))
        for L_curr in range(10, 1, -1):
            try:
                curr_estimate = estimator(L_curr, hist, curr_interval)
            except RuntimeWarning:
                continue
            if np.isnan(curr_estimate) or curr_estimate < 0 or curr_estimate > 1./curr_interval[0]:
                continue
            use_naive = False
            break

        midpoint = (intervals[i][0]+intervals[i][1])*1./2
        naive_estimate = sum(hist)*1./(midpoint * n_samples)

        print(L_curr, len(hist), sum(hist), curr_interval, use_naive, naive_estimate, curr_estimate, len([x for x in dist if x>=curr_interval[0] and x<=curr_interval[1]]))
        if use_naive:
            supp_estimate += naive_estimate
        else:
            supp_estimate += curr_estimate
        supp_naive_estimate += naive_estimate
    #print("Naive oracle estimate:", supp_naive_estimate)
    return supp_estimate, supp_naive_estimate


def callOracleEstimator(histogram, oracle_counts, base, Lo, n, n_samples):
    hist_dic = {}
    for sample in histogram:
        p = np.floor(np.log(max(1,oracle_counts[sample]))/np.log(base))
        if p not in hist_dic:
            hist_dic[p] = []
        hist_dic[p].append(histogram[sample])
    intervals = []
    histograms = []
    for p,h in hist_dic.items():
        eff_p = base**p
        intervals.append((eff_p*1./n, min(n,eff_p*base)*1./n))
        histograms.append(h)
    output = OracleEstimator(histograms, intervals, Lo, n, n_samples, base)
    print("Learned oracle:", output[0])
    print("Naive oracle:", output[1])

    
def hist_to_col(histogram, oracle, inputsize=0):
    output = []
    if inputsize == 0:
        poracle = np.array(oracle)*1./sum(oracle)
    else:
        poracle = np.array(oracle)*1./inputsize
    for sample in histogram:
        for i in range(histogram[sample]):
            output.append((sample, poracle[sample]))
    return output

# Test usage
if __name__ == '__main__':
#	n = 10000
#	actual_supp = 100
#	num_samples = 65
#	samples = np.random.randint(low=0, high=actual_supp, size = num_samples)
#	hist = samples_to_histogram(samples, n)
#	print(WuYangEstimator(n, hist))

    warnings.filterwarnings("error")

    n_samples = int(sys.argv[1])
    Lw = int(sys.argv[2])
    Lo = int(sys.argv[3])
    base = 2
    if len(sys.argv) > 4:
        base = int(sys.argv[4])
    print("Base:", base)

    histogram = {}

    # Load data and draw samples:
    if len(sys.argv)==5:
        # AOL day 50
        distribution = np.load("data/distribution.npy")
        dist = distribution # Nonsense to make it run without errors
        true_counts = np.load("data/query_counts_day_0050.npz")['counts']
        oracle_counts = np.load("data/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz")['test_output'][:,0]
        n = len(distribution)
        inputsize = n

        print("Truth:", len(true_counts))
        print("n:", n)

        for i in range(n_samples):
            sample = distribution[np.random.randint(n)]
            if sample not in histogram:
                histogram[sample] = 0
            histogram[sample] += 1

    else:
        # Synthetic Zipfian
        suppsize = 1000000
        dist = np.zeros(suppsize)
        accdist = np.zeros(suppsize+1)
        pow = 1
        nrml = sum([1./(i+1)**pow for i in range(suppsize)])
        sss = 0
        ttt = 0
        for i in range(suppsize):
            dist[i] = 1./(((i+1)**pow)*nrml)
            sss += 1./(((i+1)**pow)*nrml)
            ttt += ((1+1)**pow)*nrml
            accdist[i+1] = sss
        n = ((suppsize+1)**pow)*nrml
        inputsize = ttt

        print("Truth:", suppsize)
        print("n:", n)

        print("Drawing samples...")
        for i in range(n_samples):
            sample = np.random.uniform(0,1)
            bottom = 0
            top = len(accdist)-1
            while top-bottom>1:
                mid = int((top + bottom)/2)
                if sample <= accdist[mid]:
                    top = mid
                elif sample >= accdist[mid]:
                    bottom = mid
            if bottom not in histogram:
                histogram[bottom] = 0
            histogram[bottom] += 1
        true_counts = n * dist
        oracle_counts = np.array([x*np.random.uniform(0.5,2) for x in true_counts])

    # Wu-Yang estimate
    #print("Wu-Yang:", WuYangEstimator(n, list(histogram.values()), L=Lw))
    #print("Stupid count:", len(histogram))
    print("Original Wu-Yang:", wuyang.Support(1./n).estimate(hist_to_fin(list(histogram.values()))))

    # Perfect oracle estimate
    callOracleEstimator(histogram, true_counts, base, Lo, n, n_samples)

    # Perfect noisy estimate
    #callOracleEstimator(histogram, np.array([x*np.random.uniform(1,2) for x in true_counts]), base, Lo, n, n_samples)

    # Learned oracle estimate
    callOracleEstimator(histogram, oracle_counts, base, Lo, n, n_samples)

    # Collision algorithm
    print("Collision perfect oracle:", CollisionAlgo.collision_estimator(hist_to_col(histogram, true_counts, inputsize), 2, 2))
    print("Collision noisy oracle:", CollisionAlgo.collision_estimator(hist_to_col(histogram, np.array([x*np.random.uniform(0.5,1) for x in true_counts]), inputsize), 2, 2))
    print("Collision learned oracle:", CollisionAlgo.collision_estimator(hist_to_col(histogram, oracle_counts, inputsize), 2, 2))
