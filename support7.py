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
        #for L_curr in range(10, 1, -1):
        for L_curr in range(1):
            try:
                #curr_estimate = estimator(L_curr, hist, curr_interval) #* sum(hist)*1./n_samples
                curr_estimate = estimator(floor(0.45*log(n)), hist, curr_interval)
            except RuntimeWarning:
                continue
            if np.isnan(curr_estimate) or curr_estimate < 0 or curr_estimate > 1./curr_interval[0]:
                continue
            use_naive = False
            break

        midpoint = (intervals[i][0]+intervals[i][1])*1./2
        naive_estimate = sum(hist)*1./(midpoint * n_samples)
        #naive_estimate = sum(hist)*1./(intervals[i][0] * n_samples)

        #print(L_curr, len(hist), sum(hist), curr_interval, use_naive, naive_estimate, curr_estimate, len([x for x in dist if x>=curr_interval[0] and x<=curr_interval[1]]))
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
        intervals.append((eff_p*1./n, min(0.5*n*log(n)/n_samples, eff_p*base)*1./n)) # 0.5*log(n)/n_samples
        histograms.append(h)
    output = OracleEstimator(histograms, intervals, Lo, n, n_samples, base)
    return output
    #print("Learned oracle:", output[0])
    #print("Naive oracle:", output[1])

    
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

    warnings.filterwarnings("error")

    # Load data
    day = sys.argv[1]
    dayo = day
    if len(sys.argv) > 2:
        dayo = sys.argv[2]
    # Old data
    #true_counts = np.load("./data/actual/aol/aol_actual_counts_" + day + ".npz")['arr_0']
    #oracle_counts = np.load("./data/predictions/aol/aol_oracle_" + dayo + ".npz")['test_output'][:,0]
    # New data
    true_counts = np.load("./newdata/actual/aol_actual_counts_" + day + ".npz")['arr_0']
    oracle_counts = np.load("./newdata/predictions/aol_oracle_" + dayo + ".npz")['arr_0'][:,0]

    distribution = []
    for i in range(len(true_counts)):
        for j in range(true_counts[i]):
            distribution.append(i)

    noisy_counts = [x * np.random.uniform(4) for x in true_counts]

    n = len(distribution)
    true_count = len(true_counts)

    print()
    print("***~~~*** DAY " + day + " ***~~~***")
    print("True count:", true_count)
    print("n:", n)

    methods = ["ours+perfect", "naive+perfect", "ours+noisy", "naive+noisy", "ours+learned", "naive+learned"]
    prates = (5, 10, 15, 20)
    bases = (1.1, 1.5, 2, 3, 4, 8, 16, 32)
    reps = 10

    for prate in prates:
        print()
        print("Sample percentage (of n):", prate)
        # Run methods
        results = np.zeros((6, len(bases), reps)) # 6 methods
        n_samples = int(prate*n*1./100)
        for rep in range(reps):
            # Draw samples
            histogram = {}
            for i in range(n_samples):
                sample = distribution[np.random.randint(n)]
                if sample not in histogram:
                    histogram[sample] = 0
                histogram[sample] += 1
            for b,base in enumerate(bases):
                perfect_oracle_output = callOracleEstimator(histogram, true_counts, base, 0, n, n_samples)
                noisy_oracle_output = callOracleEstimator(histogram, noisy_counts, base, 0, n, n_samples)
                learned_oracle_output = callOracleEstimator(histogram, oracle_counts, base, 0, n, n_samples)
                results[0,b,rep] = perfect_oracle_output[0]
                results[1,b,rep] = perfect_oracle_output[1]
                results[2,b,rep] = noisy_oracle_output[0]
                results[3,b,rep] = noisy_oracle_output[1]
                results[4,b,rep] = learned_oracle_output[0]
                results[5,b,rep] = learned_oracle_output[1]
        errors = np.abs(results*1./true_count - np.ones(results.shape))
        median_errors = np.round(np.median(errors, axis=2), 2)
        std_errors = np.round(np.std(errors, axis=2), 2)
        print("Median errors:")
        print("---BASE:--- \t " + "\t".join([str(b) for b in bases]))
        for i,name in enumerate(methods):
            print(name + "\t" + "\t".join([str(x) for x in median_errors[i,:]]))
        print("Standard deviations:")
        print("---BASE:--- \t " + "\t".join([str(b) for b in bases]))
        for i,name in enumerate(methods):
            print(name + "\t" + "\t".join([str(x) for x in std_errors[i,:]]))
        #print(median_errors)
        print("Wu-Yang:", np.round(np.abs(1-WuYangEstimator(n, list(histogram.values()))*1./true_count), 2))



