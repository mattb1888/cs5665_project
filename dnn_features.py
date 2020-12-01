import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from sklearn.metrics import mean_absolute_error


def make_model1():
    input_layer = input_data(shape=[None, 70, 1, 1])
    fc_layer_1 = fully_connected(input_layer, 1000,
                                 activation='leaky_relu',
                                 name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2000,
                                 activation='leaky_relu',
                                 name='fc_layer_2')
    fc_layer_3 = fully_connected(fc_layer_2, 500,
                                 activation='leaky_relu',
                                 name='fc_layer_3')
    fc_layer_4 = fully_connected(fc_layer_3, 100,
                                 activation='leaky_relu',
                                 name='fc_layer_4')
    fc_layer_5 = fully_connected(fc_layer_4, 1,
                                 activation='linear',
                                 name='fc_layer_3')
    network = regression(fc_layer_5, optimizer='adam',
                         loss='mean_square',
                         learning_rate=0.001)
    model = tflearn.DNN(network)
    return model



def train_model(model, train_X, train_Y, test_X, test_Y, num_epochs=10, batch_size=10):
  tf.reset_default_graph()
  model.fit(train_X, train_Y, n_epoch=num_epochs,
            shuffle=True,
            validation_set=(test_X, test_Y),
            show_metric=True,
            batch_size=batch_size,
            run_id='dnn_model')

def test_model(network_model, valid_X, valid_Y):
    results = []
    for i in range(len(valid_X)):
        prediction = network_model.predict(valid_X[i].reshape([-1, 70, 1, 1]))
        results.append(prediction[0, 0])
    return mean_absolute_error(valid_Y, np.array(results))

def predict():
    tf.reset_default_graph()
    model = make_model1()
    model.load('dnn2.tfl')
    with open('test_features.pck', 'rb') as f:
        arr = pickle.load(f)

    col_mean = np.nanmean(arr, axis=0)
    nan_inds = np.where(np.isnan(arr))
    arr[nan_inds] = np.take(col_mean, nan_inds[1])

    valid_data = arr[:, 1:]
    valid_id = arr[:, 0]
    valid_data = (valid_data - np.mean(valid_data, axis=0)) / np.std(valid_data, axis=0)
    valid_data = np.nan_to_num(valid_data)
    valid_data = valid_data.reshape([-1, 70, 1, 1])

    with open('submission.csv', 'a') as f:
        for i in range(len(valid_data)):
            prediction = model.predict(valid_data[i].reshape([-1, 70, 1, 1]))
            f.write('{},{}\n'.format(int(valid_id[i]), prediction[0, 0]))


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
    train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
    test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0)
    train_data = np.nan_to_num(train_data)
    test_data = np.nan_to_num(test_data)

    train_data = train_data.reshape([-1, 70, 1, 1])
    train_target = train_target.reshape([-1, 1])
    test_data = test_data.reshape([-1, 70, 1, 1])
    test_target = test_target.reshape([-1, 1])

    model = make_model1()
    train_model(model, train_data, train_target, test_data, test_target)
    model.save('dnn_features1.tfl')
    print(test_model(model, test_data, test_target))

train()