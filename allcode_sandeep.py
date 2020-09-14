# imports
from math import log, floor
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

# WuYang

class WYSupport():
    """
    Support estimator
    """
    def __init__(self, pmin, L=None, M=None):
        """
        Args:
        pmin: float, required
        =1/k. Preset minimum non-zero mass
        L: int
        Polynoimal degree. Default c0*log(k)
        M: float
        M/n is the right-end of approximation interval. Default c1*log(k)
        """
        self.pmin = pmin
        self.degree = L if L != None else floor(0.45*log(1./pmin))
        self.ratio = M if M != None else 0.5*log(1./pmin)

    def estimate(self, fin):
        """
        Our rate-optimal estimator from a given fingerprint
        Args:
        fin: list of tuples (frequency, count)
        fingerprint of samples, fin[i] is the number of symbols that appeared exactly i times
        Return:
        the estimated entropy (bits)
        """
        # get total sample size
        num = get_sample_size(fin)

        if self.pmin < self.ratio / num:
            # get linear estimator coefficients
            cheb = np.polynomial.chebyshev.Chebyshev.basis(self.degree)
            cheb_coeffs = np.polynomial.chebyshev.cheb2poly(cheb.coef)
            shift = (self.ratio + num*self.pmin) / (self.ratio - num*self.pmin)
            a_coeffs = shift_polynomial(cheb_coeffs, -shift)
            g_coeffs = -a_coeffs / a_coeffs[0]
            g_coeffs[0] = 0
            scaling = 2. / (self.ratio-num*self.pmin)
            for j in range(1, self.degree+1):
                for i in range(1, j+1):
                    g_coeffs[j] *= (i*scaling)
                g_coeffs[j] += 1


            # get estimate
            s_estimate = 0
            for freq, cnt in fin:
                if freq > self.degree: # plug-in
                    s_estimate += 1*cnt
                else: # polynomial
                    s_estimate += g_coeffs[freq]*cnt
        else:
            s_estimate = self.estimate_plug(fin)

        return s_estimate

    def estimate_plug(self, fin):
        """
        Plug-in estimate of support size: number of seen symbols
        """
        s_estimate = 0
        for freq, cnt in fin:
            s_estimate += cnt
        return s_estimate

    def coverage_turing_good(self, fin):
        """
        Turing-Good coverage
        """
        num = get_sample_size(fin)
        fin1 = 0
        for freq, cnt in fin:
            if freq == 1:
                fin1 = cnt
                break
        return 1.-1.*fin1/num

    def estimate_turing_good(self, fin):
        """
        Turing-Good estimator
        Ref: "The Population Frequencies of Species and the Estimation of Population Parameters",
        I. J. Good, 1953
        """
        return 1.*self.estimate_plug(fin)/self.coverage_turing_good(fin)

    def estimate_jackknife1(self, fin):
        """
        First order Jackknife
        Ref: "Robust Estimation of Population Size When Capture Probabilities Vary Among Animals",
        K. P. Burnham and W. S. Overton, 1979
        """
        num = get_sample_size(fin)
        fin1 = 0
        for freq, cnt in fin:
            if freq == 1:
                fin1 = cnt
                break
        return self.estimate_plug(fin) + (num-1.0)/num*fin1
        
    def estimate_chao1(self, fin):
        """
        Chao 1 estimator
        Ref original Chao 1: "Nonparametric estimation of the number of classes in a population."
        A. Chao, 1984
        Ref this bias-corrected Chao 1: "Estimating species richness",
        N. J. Gotelli and R. K. Colwell, 2011
        """
        fin1 = 0
        fin2 = 0
        for freq, cnt in fin:
            if freq == 1:
                fin1 = cnt
                break
        for freq, cnt in fin:
            if freq == 2:
                fin2 = cnt
                break
        return self.estimate_plug(fin)+fin1*(fin1-1)/(2*(fin2+1))

    def estimate_chao_lee1(self, fin):
        """
        Chao-Lee 1, for small coefficient of variation
        Ref: "Estimating the Number of Classes via Sample Coverage", A. Chao and S. Lee, 1992
        """
        num = get_sample_size(fin)
        estimate1 = self.estimate_turing_good(fin)
        coverage = self.coverage_turing_good(fin)

        sums = 0.
        for freq, cnt in fin:
            sums += freq*(freq-1)*cnt
        gamma_sq = max(1.*estimate1*sums/num/(num-1)-1, 0)
        return estimate1 + num*(1-coverage)/coverage*gamma_sq

    def estimate_chao_lee2(self, fin):
        """
        Chao-Lee 2, for large coefficient of variation
        Ref: "Estimating the Number of Classes via Sample Coverage", A. Chao and S. Lee, 1992
        """
        num = get_sample_size(fin)
        estimate1 = self.estimate_turing_good(fin)
        coverage = self.coverage_turing_good(fin)

        sums = 0.
        for freq, cnt in fin:
            sums += freq*(freq-1)*cnt
        gamma_sq = max(1.*estimate1*sums/num/(num-1)-1, 0)
        gamma_sq2 = max(gamma_sq*(1+num*(1-coverage)*sums/num/(num-1)/coverage), 0)
        return estimate1 + num*(1-coverage)/coverage*gamma_sq2

def get_sample_size(fin):
    """
    get total sample size from a given fingerprint
    """
    num = 0
    for freq, cnt in fin:
        num += freq * cnt
    return num


def sample_to_fin(samples):
    """
    Return a fingerprint from samples
    """
    return hist_to_fin(sample_to_hist(samples))


def hist_to_fin(hist):
    """
    Return a fingerprint from histogram
    """
    fin = Counter()
    for freq in hist:
        fin[freq] += 1
    return fin.items()

def sample_to_hist(samples):
    """
    Return a histogram of samples
    """
    freq = Counter()
    for symbol in samples:
        freq[symbol] += 1
    return np.asarray(list(freq.values()))


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


class Counter(dict):
    """
    Class for counting items
    """
    def __missing__(self, key):
        return 0


# Oracle Estimator


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

def samples_to_histogram(samples):
    """
    Return a histogram from a list of smaples, n = universe size
    """
    histogram = {}
    for s in samples:
        if s in histogram:
            histogram[s] += 1
        else:
            histogram[s] = 1
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
    
    num_intervals = len(intervals)
    num_failures = 0.0
    supp_estimate = 0.0
    supp_naive_estimate = 0.0
    # loop over histogram and intervals and add up support 
    for i, hist in enumerate(histograms):
        use_naive = True
        curr_interval = intervals[i]

        for L_curr in range(1):
            try:
                curr_estimate = estimator(L, hist, curr_interval) #floor(0.45*log(n))
            except RuntimeWarning:
                continue
            if np.isnan(curr_estimate) or curr_estimate < 0 or curr_estimate > 1./curr_interval[0]:
                continue
            use_naive = False
            break

        midpoint = (intervals[i][0]+intervals[i][1])*1./2
        naive_estimate = sum(hist)*1./(midpoint * n_samples)

        if use_naive:
            supp_estimate += naive_estimate
            num_failures += 1
        else:
            supp_estimate += curr_estimate
        supp_naive_estimate += naive_estimate
    return [supp_estimate, supp_naive_estimate, num_failures/num_intervals, num_intervals]



def callOracleEstimator(histogram, oracle_counts, base, n, n_samples, L):
    if L == 0:
        L = int(floor(0.45*log(n)))

    hist_dic = {}
    outlier = []
    
    base_log = log(base)
    log_vals = np.floor(np.log(oracle_counts)/base_log)
    
    for sample, value in histogram.items():
        p = log_vals[sample] 
        if p not in hist_dic:
            hist_dic[p] = []
        hist_dic[p].append(value)
    
    
    intervals = []
    histograms = []
    
    for p,h in hist_dic.items():
        eff_p = base**p
        if eff_p < 0.5*n*log(n)/n_samples:
            intervals.append((eff_p*1./n, min(0.5*n*log(n)/n_samples, eff_p*base)*1./n)) 
            histograms.append(h)
        else:
            outlier += h
    
    max_p = int(np.floor(np.log(0.5*n*log(n)/n_samples)/np.log(base)))
    
    try:
        histograms[max_p] += outlier
    except:
        histograms[-1] += outlier

    output = OracleEstimator(histograms, intervals, L, n, n_samples, base)
    return output


    
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


def generate_zip_prob(n):
    output = np.arange(1.0, n+1.0)
    output = 1./output
    const = output.sum()
    output = (1./const)*output
    return output