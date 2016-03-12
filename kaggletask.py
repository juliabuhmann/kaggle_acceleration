import pandas

import lasagne
import numpy as np
import h5py
import random
import multiprocessing




def build_mlp(input_var=None, num_classes=399):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(
        l_in_drop, num_units=800,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1_drop, num_units=800,
        nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hid2_drop, num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    return l_out


bin_size = 100



# train_data_arr = np.array(train_data[:10000].values)
# print train_data_arr

f = h5py.File('train.h5', 'r')
train_data_arr = f['num_arr'].value
train_data_arr2 = f['num_arr2'].value
f.close()

# train_data_arr = np.reshape(train_data_arr, [np.shape(train_data_arr)[1], np.shape(train_data_arr)[0]])



def get_sample(data, size=300, pos=None):
    data_size = data.shape[0]
    if pos is None:
        pos = random.randint(0, data_size-size)
    sample = data[pos:pos+size, :-1]
    return sample

def make_data_matrix(data, bin_size = 100):
    data_size = (data.shape[0] // bin_size)
    x_data = []
    y_data = []
    for ii in range(data_size):
        all_values = data[ii*bin_size:(ii+1)*bin_size, : 4]

        assert len(np.unique(data[ii*bin_size:(ii+1)*bin_size, 4])) == 1

        all_values_in_one = np.reshape(all_values[:, :5], ((data.shape[1]-1)*bin_size, 1))
        x_data.append(all_values_in_one)
        y_data.append(data[ii*bin_size:(ii+1)*bin_size, 4])

    return x_data, y_data

print train_data_arr.shape
x_target, y_target = make_data_matrix(train_data_arr)
x_target1, y_target1 = make_data_matrix(train_data_arr2)

x_target.extend(x_target1)
y_target.extend(y_target1)
print len(x_target)
#print x_target, y_target





"""
print train_data_arr[0,10]

# train_data_arr = np.swapaxes(train_data_arr, 0, 1)
print train_data_arr.shape
for column in range(4):
    data_col = train_data_arr[:, column]
    data_col -= np.mean(data_col)
    data_col /= np.std(data_col)


train_data_arr = (train_data_arr[1:-1]-np.mean(train_data_arr[1:-1]))/np.std(train_data_arr[1:-1])
print train_data_arr[0]

print np.unique(train_data_arr[:, -1])
print np.unique(train_data_arr2[:, -1])
print train_data_arr[:10, :]
"""