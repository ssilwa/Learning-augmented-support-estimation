import numpy as np 
from support import *


####### Loading data #######

print('Loading data')
actual_data = np.load('data/query_counts_day_0050.npz')
predicted_data = np.load('data/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz')

# Get actual query probabilities
actual_prob = actual_data['counts']/actual_data['counts'].sum()


####### Testing algorithm with actual probabilities #######

print('Preprocessing')
# Guess for the universe size
n = int(1e6)

# Get samples drawn according to actual probability
S = np.random.choice(range(actual_prob.size), size=50000, p=actual_prob)

# Get probabilities of samples
S_predicted_prob = actual_prob[S]

# Round probabilities to nearest multiple of 1/n
rounded_prob = [int(round(n*prob)) for prob in S_predicted_prob]
S_with_prob = list(zip(S, rounded_prob))

# Get intervals
# Interval from [1/n to 1/(eps*n)]
eps = 1e-6
intervals = []
left = 1
right = 4
while(right < 1/eps):
    intervals.append([left, right])
    left = right
    right *= 4
intervals.append([left, right])


# Partition samples into intervals
sample_partition = [[] for k in range(len(intervals))]
for k in S_with_prob:
    curr_prob = k[1]
    bucket = int(np.log(curr_prob)/np.log(4))-1
    try:
        sample_partition[bucket].append(k[0])
    except:
        pass
    
# Get histogram for each interval
histograms = [samples_to_histogram(samp, n) for samp in sample_partition]

# Feed to oracle
print('Running oracle algorithm')
support = OracleEstimator(histograms, intervals, 20)
print(support)

# WuYang algorithm
full_hist = samples_to_histogram(S,n)
print('Running WuYang algorithm')
support = WuYangEstimator(n, full_hist)
print(support)

####### Testing with fake data #######

print('Preprocessing fake data')
# Guess for the universe size
n = 10000

# Get samples drawn according to uniform on {0,..,99}
S = np.random.choice(range(100), size=300)

# Get probabilities of samples
S_predicted_prob = [1./100 for s in S]

# Round probabilities to nearest multiple of 1/n
rounded_prob = [int(round(n*prob)) for prob in S_predicted_prob]
S_with_prob = list(zip(S, rounded_prob))

# Get intervals
# Interval from [1/n to 1/(eps*n)]
eps = 1e-3
intervals = []
left = 1
right = 4
while(right < 1/eps):
    intervals.append([left, right])
    left = right
    right *= 4
intervals.append([left, right])

# Partition samples into intervals
sample_partition = [[] for k in range(len(intervals))]
for k in S_with_prob:
    curr_prob = k[1]
    bucket = int(np.log(curr_prob)/np.log(4))-1
    try:
        sample_partition[bucket].append(k[0])
    except:
        pass

# Get histogram for each interval
histograms = [samples_to_histogram(samp, n) for samp in sample_partition]

# Feed to oracle
print('Running oracle algorithm')
support = OracleEstimator(histograms, intervals, 5)
print(support)

# WuYang algorithm
full_hist = samples_to_histogram(S,n)
print('Running WuYang algorithm')
support = WuYangEstimator(n, full_hist)
print(support)
