from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import numpy as np
from random import randint
import matplotlib.pyplot as plt

# function to load training data and train random forest
def train(trees=50, filename='rf.pck'):
    # load pickled features
    with open('train_features10.pck', 'rb') as f:
        arr = pickle.load(f)

    # replace NaN with column mean
    col_mean = np.nanmean(arr, axis=0)
    nan_inds = np.where(np.isnan(arr))
    arr[nan_inds] = np.take(col_mean, nan_inds[1])

    # split in to 3000 training samples and the rest validation
    train = arr[:3000, ]
    test = arr[3000:, ]

    # split into data and targets
    train_data = train[:, 1:]
    train_target = train[:, 0]
    test_data = test[:, 1:]
    test_target = test[:, 0]

    # create and train random forest
    reg = RandomForestRegressor(n_estimators=trees, criterion='mae', max_depth=None, 
                                max_features=None, random_state=randint(1, 1000), verbose=2,
                                n_jobs=-1)
    rf = reg.fit(train_data, train_target)

    # pickle random forest
    with open(filename, 'wb') as f:
        pickle.dump(rf, f)

    # evaluate on validation data
    test_preds = rf.predict(test_data)
    result = mean_absolute_error(test_target, test_preds)
    print(result)
    return result

# function to load pickled model, load test data, and write submission file
def predict(model):
    # load pickled features
    with open('test_features10.pck', 'rb') as f:
        arr = pickle.load(f)

    # replace NaNs with column mean
    col_mean = np.nanmean(arr, axis=0)
    nan_inds = np.where(np.isnan(arr))
    arr[nan_inds] = np.take(col_mean, nan_inds[1])
    
    # split into data and segment_id
    valid_data = arr[:, 1:]
    valid_id = arr[:, 0]

    # load pickled random forest
    with open(model, 'rb') as f:
        rf = pickle.load(f)
    
    # predict and write to file
    valid_preds = rf.predict(valid_data)
    with open('submission.csv', 'a') as f:
        for i in range(len(valid_id)):
            f.write('{},{}\n'.format(int(valid_id[i]), valid_preds[i]))

# function to create random forests of increasing size and plot their performance
def train_rfs():
    acc = train(trees=1, filename='rfs/rf_1.pck')
    sizes = [1]
    results = [acc]
    for i in range(5, 76, 5):
        acc = train(trees=i, filename='rfs/rf_{}.pck'.format(i))
        sizes.append(i)
        results.append(acc)
    print(results)
    figure = plt.figure()
    figure.suptitle('Random Forest Performance')
    plt.xlabel('Number of decision trees')
    plt.ylabel('Mean Absolute Error')
    plt.grid()
    plt.plot(sizes, results, c='blue')
    plt.show()

#predict('rfs/rf_50.pck')