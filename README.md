Code for the submission "Learning-based Support Estimation in Sublinear Time." See https://openreview.net/forum?id=tilovEHA3YS

Files:

- code.py: contains implementation of our algorithm
- sample_use.py: contains a sample use case

If you wish to run our algorithm on all of AOL and CIADA data, please get the training data as specified here: https://github.com/chenyuhsu/learnedsketch (we use their RNN as our oracle/predictor but you don't need the model to use our algorithm, just the predictions of the model). 

If you wish to actually train/use the model, you can load the approapriate checkpoint from the folder 'pretrained' and use the appropriate loading file from the above linked github.

