"""
Code for the submission "Learning-based Support Estimation in Sublinear Time"

"""

# package imports
from math import log, floor
import numpy as np
import scipy.special


################################################ Auxiliary functions ################################################

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
	

def generate_zip_prob(n, alpha=0.5):
	"""
	Returns probability distribution of a zipfian distribution
	
	Args:
	n: Distribution over {1, ..., n}
	alpha: parameter of power law

	"""
	output = np.arange(1.0, n+1.0)
	output = output**(-alpha)
	const = output.sum()
	output = (1./const)*output
	return output




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



def upper_bound(n, perc):
	"""
	Computes the smallest base value that results in only 1 interval partition
	"""
	return int(np.ceil(.5*log(n)/perc))+1


def base_chooser(bases, failure, supp_estimates):
	"""
	Computes which base/estimate to use using our failure rule
	
	Args:
	bases: list of baes
	supp_estiamtes: estimates for each base
	failure: array that has the number of intervals that failed for each base

	"""

	# loop until base found that has no failures afterwards
	best_index = 0
	for indx, fail_perct in enumerate(failure):
		if sum(failure[indx:]) == 0.0:
			best_index = indx
			break
	
	# take twice that value for buffer
	best_index2 = int(min(best_index*2, len(bases)-1))
	return (bases[best_index2], supp_estimates[best_index2])


################################################################################################################################################


def OracleEstimator(histograms, intervals, L, n, n_samples, base):
	"""
	Returns sum of supports for each interval along with the number of intervals that failed sanity check
	
	Args:
	histograms: list of histograms, one for each interval
	intervals: list of intervals where each interval is of the form [left_end, right_end]
	L: degree of polynomial
	"""

	num_intervals = len(intervals)
	num_failures = 0.0
	supp_estimate = 0.0

	# loop over histogram and intervals and add up support 
	for i, hist in enumerate(histograms):
		use_fallback = True
		curr_interval = intervals[i]

		for L_curr in range(1):
			try:
				curr_estimate = estimator(L, hist, curr_interval) 
			except RuntimeWarning:
				continue
			if np.isnan(curr_estimate) or curr_estimate < 0 or curr_estimate > 1./curr_interval[0]:
				continue
			use_fallback = False
			break

		# fall back option is to use distinct elements
		# Note: we do not actually use this in our algorithm since we only pick a base that has does not fail
		distinct_estimate = len(hist)

		if use_fallback:
			supp_estimate += distinct_estimate
			num_failures += 1
		else:
			supp_estimate += curr_estimate
	return [supp_estimate, num_failures/num_intervals]

def callOracleEstimator(histogram, oracle_counts, base, n, n_samples, L = 0):
	"""
	Partition histogram of samples into intervals based on predictor
	
	Args:
	histogram: histogram of samples
	oracle_counts: array where the ith entry is the predicted count of the ith element
	base: value for the base parameter
	n: estimate for 1/p_min
	n_samples: number of samples
	L: degree of polynomial, 0 indicates use default value
	"""

	# use default value of polynomial
	if L == 0:
		L = int(floor(0.45*log(n)))
		

	# partition samples into different intervals
	hist_dic = {}
	outlier = []
	
	max_val = 0.5*n*log(n)/n_samples
	base_log = log(base)
	log_vals = np.floor(np.log(oracle_counts)/base_log)
	
	for sample, value in histogram.items():
		
		if oracle_counts[sample] <= max_val:
			p = log_vals[sample] 
			if p not in hist_dic:
				hist_dic[p] = []
			hist_dic[p].append(value)
		else:
			outlier.append(value)
	
	
	max_slots = int(max(hist_dic.keys()))+1
	intervals = [[] for k in range(max_slots)]
	histograms = [[] for k in range(max_slots)]
	
	for p,h in hist_dic.items():
		eff_p = base**p
		intervals[int(p)] = [eff_p*1./n, eff_p*base*1./n]
		histograms[int(p)] += h
		
	max_p = int(np.floor(np.log(0.5*n*log(n)/n_samples)/np.log(base)))

	if max_p > max_slots-1:
		eff_p = base**max_p
		intervals.append([eff_p*1./n, eff_p*base*1./n])
		histograms.append(outlier)
	else:
		histograms[max_p] += outlier
	
	histograms = [hists for hists in histograms if len(hists) > 0]
	intervals = [intvs for intvs in intervals if len(intvs) > 0]

	# call our algorithm on each interval
	output = OracleEstimator(histograms, intervals, L, n, n_samples, base)
	return output

################################################ Main function ################################################


def main_alg(samples, predicted_counts, n, sample_size):
	
	"""
	Given samples and predicted counts, returns estimate of support size
	
	Args:
	samples: array of samples
	predicted_counts: array where the ith entry is the predicted count of the ith element
	n: estimate for 1/p_min
	sample_size: number of samples
	"""

	perct = sample_size/n
	histogram = samples_to_histogram(samples)

	# get bases from 2 to the largest value that results in only one interval partition
	bases = range(2, upper_bound(n, perct)+1)
	base_results = []
	failure_tracker = []

	# loop over possible bases and get estimate for each base
	for base in bases:
		base_estimate, base_failure_rate = callOracleEstimator(histogram, predicted_counts, base, n, sample_size)
		base_results.append(base_estimate)
		failure_tracker.append(base_failure_rate)
	  
	# pick the base and estimate according to our failure rule  
	chosen_base, chosen_error = base_chooser(bases, failure_tracker, base_results)
	return chosen_error

