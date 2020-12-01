from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import numpy as np



def train():
    with open('train_features.pck', 'rb') as f:
        arr = pickle.load(f)

    col_mean = np.nanmean(arr, axis=0)
    nan_inds = np.where(np.isnan(arr))
    arr[nan_inds] = np.take(col_mean, nan_inds[1])

    train = arr[:3000, ]
    test = arr[3000:, ]
    train_data = train[:, 1:]
    train_target = train[:, 0]
    test_data = test[:, 1:]
    test_target = test[:, 0]

    reg = RandomForestRegressor(n_estimators=200, criterion='mae', max_depth=None, 
                                max_features=None, random_state=200, verbose=2,
                                n_jobs=-1)
    rf = reg.fit(train_data, train_target)
    test_preds = rf.predict(test_data)
    print(mean_absolute_error(test_target, test_preds))
    with open('rf_features2.pck', 'wb') as f:
        pickle.dump(rf, f)

def predict(model):
    with open('test_features.pck', 'rb') as f:
        arr = pickle.load(f)

    col_mean = np.nanmean(arr, axis=0)
    nan_inds = np.where(np.isnan(arr))
    arr[nan_inds] = np.take(col_mean, nan_inds[1])
    
    valid_data = arr[:, 1:]
    valid_id = arr[:, 0]

    with open(model, 'rb') as f:
        rf = pickle.load(f)
    valid_preds = rf.predict(valid_data)
    with open('submission.csv', 'a') as f:
        for i in range(len(valid_id)):
            f.write('{},{}\n'.format(int(valid_id[i]), valid_preds[i]))


predict('rf_features1.pck')
#train()
