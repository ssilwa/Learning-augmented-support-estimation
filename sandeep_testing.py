# Import all previous code
from allcode import *
import matplotlib.pyplot as plt



# Takes as input
# data which is ['aol', '06'] or ['ip', '1329'] etc
# bases: list of bases
# num_iters: number of iterations to run
# percent: somethng like 0.01 which is what to use as sample size
# print_int: print the number of intervals
# L: degree of polynomial, L = 0 means use default value

# Returns: oracle_output which is results for learned oracle (size is # bases), std_oracle_output is std of the values, wy_output is outout of WY algo
# naive_output is output of naive algorithm, failure_output is fraction of intervals that failed for each base, interval_output is # intervals per base

def base_tester(data, bases, num_iters, percent, print_int = False, L = 0):

    # load data
    dataset, data_type = data
    
    print("dataset: " + dataset + ", type:" + data_type)

    
    # actual data
    actual_file = np.load('data/actual/' + dataset + '/' + dataset + '_actual_counts_' +  data_type + '.npz')
    actual_counts = actual_file['arr_0']
    actual_prob = actual_counts/actual_counts.sum()
    
    # set parameters
    n = actual_counts.sum()
    act_supp = len(actual_counts)
    print("n: {}".format(n))
    print("actual support: {}".format(act_supp))
    print("percent: {}".format(percent))
    sample_size = int(n*percent)
    print("sample size: {}".format(sample_size))
    WY_estimator = WYSupport(pmin=1./n)
    
    
    
    # predicted data
    predicted_file = np.load('data/predictions/' + dataset + '/' + dataset + '_oracle_' + data_type + '.npz')
    try:
        predicted_counts = predicted_file['test_output'][:,0]
    except:
        predicted_counts = predicted_file['arr_0']
        
    # ip predictions are originally logs
    if dataset == "ip":
        predicted_counts = np.exp(predicted_counts)
    
    
    predicted_counts[ predicted_counts < 1.0] = 1.0
    predicted_prob = predicted_counts/n
    
    oracle_holder = np.zeros((num_iters, len(bases)))
    naive_holder = np.zeros((num_iters, len(bases)))
    WY_holder = np.zeros(num_iters)
    failure_holder = np.zeros((num_iters, len(bases)))
    interval_holder = np.zeros((num_iters, len(bases)))
    
    for i in range(num_iters):
        samples = np.random.choice(act_supp, size=sample_size, p=actual_prob)
        histogram = samples_to_histogram(samples)
        
        pred_array_oracle = []
        pred_array_naive = []
        failure_array = []
        interval_array = []
        
        for base in bases:
            oracle_estimate_pred, naive_oracle_estimate_pred, failure, num_intervals = callOracleEstimator(histogram, predicted_counts, base, n, sample_size, L)
            pred_array_oracle.append(oracle_estimate_pred)
            pred_array_naive.append(naive_oracle_estimate_pred)
            failure_array.append(failure)
            interval_array.append(num_intervals)
            
            if print_int:
                print(base, num_intervals, failure)
        
        oracle_holder[i,:] = pred_array_oracle
        naive_holder[i,:] = pred_array_naive
        failure_holder[i,:] = failure_array
        interval_holder[i,:] = interval_array
        
        
        fin = sample_to_fin(samples)
        WY_estimate = WY_estimator.estimate(fin)
        WY_holder[i] = WY_estimate
    
    # take median
    oracle_holder = np.abs(1-oracle_holder/act_supp)
    naive_holder = np.abs(1-naive_holder/act_supp)
    oracle_output = np.median(oracle_holder, axis = 0)
    
    naive_output = np.median(naive_holder, axis = 0)
    std_oracle_output = np.std(oracle_holder, axis = 0)
    WY_holder = np.abs(1-WY_holder/act_supp)
    wy_output = np.median(WY_holder)
    
    failure_output = np.median(failure_holder, axis = 0)
    interval_output = np.median(interval_holder, axis = 0)
            
    return oracle_output, std_oracle_output, wy_output, naive_output, failure_output, interval_output


def base_tester_zipf(bases, actual_prob, predicted_counts, sample_size, n, L, n_iter=5, print_int = False):
    output = np.zeros((n_iter, len(bases)))
    output_failure = np.zeros((n_iter, len(bases)))
    output_interval = np.zeros((n_iter, len(bases)))
    
    for i in range(n_iter):
        print(i)
        samples = np.random.choice(act_supp, size=sample_size, p=actual_prob)
        histogram = samples_to_histogram(samples)
        
        curr_result_oracle = []
        wy_estimates = []
        failure_oracle = []
        interval_oracle = []
        
        curr_wy_estimator = WYSupport(pmin)
        curr_wy_estimate = curr_wy_estimator.estimate(sample_to_fin(samples))
        
        for base in bases:
            oracle_estimate_pred, naive_oracle_estimate_pred, failure, num_intervals = callOracleEstimator(histogram, predicted_counts, base, n, sample_size, L)
            curr_result_oracle.append(oracle_estimate_pred)
            failure_oracle.append(failure)
            interval_oracle.append(num_intervals)
            
            
            if print_int:
                print(num_intervals, failure)
            
        output[i,:] = curr_result_oracle
        output_failure[i,:] = failure_oracle
        output_interval[i,:] = interval_oracle
        wy_estimates.append(curr_wy_estimate)
    output = np.abs(1-output/act_supp)
    output_oracle = np.median(output, axis=0)
    output_std = np.std(output, axis = 0)
    output_failure = np.median(output_failure, axis = 0)
    output_interval = np.median(output_interval, axis = 0)
    wy_estimates = [np.abs(1-k/act_supp) for k in wy_estimates]
        
    return output_oracle, output_std, np.median(wy_estimates), output_failure, output_interval

