"""
Sample code usage for the submission "Learning-based Support Estimation in Sublinear Time"

"""

# import code
from code import *
import numpy as np 

if __name__ == "__main__":

	
	print("Loading data")
	# load actual data to simualte samples
	actual_file = np.load('sample_data/actual/aol_actual_counts_50.npz')
	actual_counts = actual_file['arr_0']
	actual_prob = actual_counts/actual_counts.sum()
	actual_support = len(actual_counts)
	n = actual_counts.sum()

	# use 5% of n number of samples
	sample_size = int(n*0.05)
	samples = np.random.choice(actual_support, size=sample_size, p=actual_prob)

	# get predicted counts from predictor
	predicted_file = np.load('sample_data/predictions/aol_oracle_50.npz')
	predicted_counts = predicted_file['test_output'][:,0]
	predicted_counts[ predicted_counts < 1.0] = 1.0

	print("Loaded data, getting estimate")
	print("Sample size: 5%")

	# run algorithm to get support estimate
	support_estimate = main_alg(samples, predicted_counts, n, sample_size)

	# print actual support size and our estimate
	print("actual support size:{}, estimated support size:{:.2f}, error:{:.2f}".format(actual_support, support_estimate, np.abs(1-support_estimate/actual_support)))