import pandas as pd
import numpy as np
import pickle
import os

# get the targets and save them in a dictionary
df = pd.read_csv('train.csv')
targets = dict(zip(df.segment_id, df.time_to_eruption))

# function to get features from csv to numpy array with target
def csv_to_array_train(file_name):
    data = pd.read_csv(file_name)
    id = int(file_name.split('/')[-1][:-4])
    out = np.array(targets[id])
    des = data.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9])
    out = np.append(out, des[1:])
    return out

# function to get feature from all training data and pickle it
def all_csv_train(directory):
    files = os.listdir(directory)
    out = [csv_to_array_train(directory + files[0])]
    count = 2
    for f in files[1:]:
        arr = [csv_to_array_train(directory + f)]
        out = np.append(out, arr, axis=0)
        print('{} of {}'.format(count, len(files)))
        count += 1
    with open('train_features10.pck', 'wb') as f:
        pickle.dump(out, f)
    return out

# function to get features from csv to numpy array with segment_id
def csv_to_array_test(file_name):
    data = pd.read_csv(file_name)
    id = int(file_name.split('/')[-1][:-4])
    out = np.array(id)
    des = data.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9])
    out = np.append(out, des[1:])
    return out

# function to get features from all test data and pickle it
def all_csv_test(directory):
    files = os.listdir(directory)
    out = [csv_to_array_test(directory + files[0])]
    count = 2
    for f in files[1:]:
        arr = [csv_to_array_test(directory + f)]
        out = np.append(out, arr, axis=0)
        print('{} of {}'.format(count, len(files)))
        count += 1
    with open('test_features10.pck', 'wb') as f:
        pickle.dump(out, f)
    return out

all_csv_train('train/')
all_csv_test('test/')
