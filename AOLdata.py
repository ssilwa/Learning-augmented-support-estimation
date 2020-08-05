import numpy as np 
from support import *


####### Testing Chen Yu data ######

# Load actual query data
actual_data = np.load('data/query_counts_day_0050.npz')
predicted_data = np.load('data/aol_inf_all_v05_t50_exp22_aol_5d_r1-h1_u256-32_eb64_bs128_ra_20180514-160509_ep190_res.npz')


# Get file contents
print(actual_data.files)
print(predicted_data.files)

# Make sure dimensions match
print(actual_data['queries'].shape)
print(predicted_data['test_output'].shape)


# Some of the predicitons are negative ... 
good_indices = np.where(predicted_data['test_output'][:,0] > 0)

# Reported loss
print(predicted_data['test_loss'])

# Actual loss
# Slicing predicted_data['test_output'][:,0] to get rid of extra (,1) dimension at the end

actual_loss = ((np.log(predicted_data['test_output'][:,0][good_indices])-np.log(actual_data['counts'][good_indices]))**2).sum()
print(actual_loss)