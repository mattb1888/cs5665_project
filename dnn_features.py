import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from sklearn.metrics import mean_absolute_error

# function to make the neural network
def make_model1():
    input_layer = input_data(shape=[None, 130, 1, 1])
    conv_layer_1 = conv_2d(input_layer,
                           nb_filter=8,
                           filter_size=3,
                           activation='leaky_relu',
                           name='conv_layer_1')
    pool_layer_1 = max_pool_2d(conv_layer_1, 2, name='pool_layer_1')
    conv_layer_2 = conv_2d(pool_layer_1,
                           nb_filter=8,
                           filter_size=3,
                           activation='leaky_relu',
                           name='conv_layer_2')
    pool_layer_2 = max_pool_2d(conv_layer_2, 2, name='pool_layer_2')
    fc_layer_1 = fully_connected(pool_layer_2, 2000,
                                 activation='leaky_relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 1000,
                                 activation='leaky_relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 250,
                                 activation='leaky_relu',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 1,
                                 activation='linear',
                                 name='fc_layer_4')
    network = regression(fc_layer_4, optimizer='adam',
                         loss='mean_square',
                         learning_rate=0.001)
    model = tflearn.DNN(network)
    return model

# function to train a neural network
def train_model(model, train_X, train_Y, test_X, test_Y, num_epochs=10, batch_size=10):
  tf.reset_default_graph()
  model.fit(train_X, train_Y, n_epoch=num_epochs,
            shuffle=True,
            validation_set=(test_X, test_Y),
            show_metric=True,
            batch_size=batch_size,
            run_id='dnn_model')

# function to calculate MAE between model's prediction and ground truth
def test_model(network_model, valid_X, valid_Y):
    results = []
    for i in range(len(valid_X)):
        prediction = network_model.predict(valid_X[i].reshape([-1, 130, 1, 1]))
        results.append(prediction[0, 0])
    return mean_absolute_error(valid_Y, np.array(results))

# function to load trained model, load testing data, and create submission file
def predict():
    # load model
    tf.reset_default_graph()
    model = make_model1()
    model.load('dnn_features2.tfl')

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

    # normalize and reshape data 
    valid_data = (valid_data - np.mean(valid_data, axis=0)) / np.std(valid_data, axis=0)
    # some columns have standard deviation of 0 so replace the NaNs again
    valid_data = np.nan_to_num(valid_data)
    valid_data = valid_data.reshape([-1, 130, 1, 1])

    # predict using model and write to submission file
    with open('submission.csv', 'a') as f:
        for i in range(len(valid_data)):
            prediction = model.predict(valid_data[i].reshape([-1, 130, 1, 1]))
            f.write('{},{}\n'.format(int(valid_id[i]), prediction[0, 0]))

# method to load data, train model, and save model
def train():
    # load pickled features
    with open('train_features10.pck', 'rb') as f:
        arr = pickle.load(f)
    
    # replace NaNs with column means
    col_mean = np.nanmean(arr, axis=0)
    nan_inds = np.where(np.isnan(arr))
    arr[nan_inds] = np.take(col_mean, nan_inds[1])

    # split into 3000 training samples and the rest validation
    train = arr[:3000, ]
    test = arr[3000:, ]

    # split into data and targets
    train_data = train[:, 1:]
    train_target = train[:, 0]
    test_data = test[:, 1:]
    test_target = test[:, 0]

    # normalize data
    train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
    test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0)
    # some columns have standard deviation of zero so replace NaNs again
    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)

    # reshape data
    train_data = train_data.reshape([-1, 130, 1, 1])
    train_target = train_target.reshape([-1, 1])
    test_data = test_data.reshape([-1, 130, 1, 1])
    test_target = test_target.reshape([-1, 1])

    # make model, train it, save it, and evaluate it on validation data
    model = make_model1()
    train_model(model, train_data, train_target, test_data, test_target)
    model.save('dnn_features2.tfl')
    print(test_model(model, test_data, test_target))
