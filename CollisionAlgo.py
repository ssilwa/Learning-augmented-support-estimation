import numpy as np 
import scipy.special

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

def scale_polynomial(coeffs, scale):
    """
    Return coefficients of a scaled polynomial
    P(x) = p(scale*x)
    Args:
    coeffs: list of float coeffs of p
    Returns:
    coefficients of P
    """
    length = len(coeffs)
    factor = 1
    for i in range(length):
        coeffs[i] *= factor
        factor *= scale
    return coeffs
    
#################################################################################################


def get_poly(C, L):
    """
    Return coefficients of a scaled chebyshev polynomial in interval [1,C]
    that satisfies p(0) = 0 and roughly 1 in the interval [1,C]
    Args:
    C: integer that is right end point
    L: degree of polynomial
    Returns:
    coefficients of polynomial
    """
    cheb = np.polynomial.chebyshev.Chebyshev.basis(L)
    cheb_coeffs = np.polynomial.chebyshev.cheb2poly(cheb.coef)
    A = 2/(C-1)
    B = -1-A
    poly_coeffs = scale_polynomial(shift_polynomial(cheb_coeffs, B), A)
    poly_coeffs /= cheb(B)
    poly_coeffs[0] -= 1
    return -poly_coeffs

def collision_estimator(S, C=2, L=3):
    
    """
    Return estimate of support size using collision based algorithm
    Args:
    S: list of tuples of the form (sample, predicted probability)
    C: probability is a C > 1 approximation, default value 2
    L: degree of polynomial to use, uses L wise collisions, default value 3
    Returns:
    estimate of support according to collision based algorithm
    """
        
    # make a hash table to count frequency of each item
    counter = {}
    for sample in S:
        if sample[0] in counter:
            counter[sample[0]][0] += 1
        else:
            counter[sample[0]] = [1, sample[1]]
    
    # compute i wise collisions for i = 1,..., L
    ratios = np.zeros(L+1)
    counter_summary = list(counter.items())
    for i in range(1,L+1):
        curr_ratio = 0.0
        for pair in counter_summary:
            if pair[1][0] >= i:
                # i wise collision gets weighted by 1/(predicted probability)^i
                curr_ratio += scipy.special.binom(pair[1][0],i)/pair[1][1]**i
        curr_ratio /= scipy.special.binom(len(S), i)
        ratios[i] = curr_ratio
    
    # 0 wise collisions 
    ratios[0] = 1
    
    # get coefficients of scale and shifted chebyshev polynomial for interval [1,C]
    poly_coeffs = get_poly(C, L)
    
    # evaluate the polynomial to get support estimate
    support_estimate = poly_coeffs.dot(ratios)
        
    return support_estimate
    
    
# testing
if __name__ == '__main__':
	# load data 
	actual_data = np.load('data/by_day/query_counts_day_0050.npz') 
	# actual probability for testing
	actual_prob = actual_data['counts']/actual_data['counts'].sum()
	# get samples and respective probabilities
	S = np.random.choice(range(140725), size=25000, p=actual_prob)
	S_predicted_prob = actual_prob[S]
	# put it in requried format
	samples = list(zip(S, S_predicted_prob))
	# get estimate
	support = collision_estimator(samples, 2, 10)
	print(support)
