from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import numpy as np

# function to load training data and train random forest
def train():
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
    reg = RandomForestRegressor(n_estimators=250, criterion='mae', max_depth=None, 
                                max_features=None, random_state=100, verbose=2,
                                n_jobs=-1)
    rf = reg.fit(train_data, train_target)

    # evaluate on validation data
    test_preds = rf.predict(test_data)
    print(mean_absolute_error(test_target, test_preds))

    # pickle random forest
    with open('rf_features2.pck', 'wb') as f:
        pickle.dump(rf, f)

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
