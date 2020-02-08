import time

import numpy as np
from Bio import Phylo
import os
import re
from tego.util import to_adjacency_matrix, to_distance_matrix, to_node_attributes
import gc
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from spektral.layers import EdgeConditionedConv, GlobalAvgPool
from keras.optimizers import Adam
from keras import regularizers
print("Imported packages.")


def nwkToNumpy(path) -> (np.ndarray, np.ndarray):
    tree = Phylo.read(path, "newick")
    A = to_adjacency_matrix(tree)
    X = to_node_attributes(path)
    E = to_distance_matrix(tree)
    return A, X, E


def getData(path="subtrees"):
    y = []
    adj = []
    nod = []
    edg = []
    files = os.listdir(path)
    successful = 0
    failed = 0
    for f in files:
        success = int(re.search("\((.*?)\)", f).group()[1:-1])
        if (successful < failed and success == 1) or (failed < successful and success == 0) or (failed == successful):
            # print("Get " + str(i) + " out of " + str(len(files) - 1))
            A, X, E = nwkToNumpy('subtrees/' + f)
            adj.append(A)
            nod.append(X)
            edg.append(E)
            y.append(success)
            if success == 1:
                successful += 1
            else:
                failed += 1
    print("Got " + str(successful) + " succesful trees and " + str(failed) + " failed trees.")
    print("Files read. Padding them for training.")
    k = max([_.shape[-1] for _ in adj])
    for i in range(len(adj)):
        matrix = adj[i]
        temp = np.zeros((k, k))
        temp[:matrix.shape[0], :matrix.shape[1]] = matrix
        adj[i] = temp
    print("Adj Padded.")
    for i in range(len(edg)):
        matrix = edg[i]
        temp = np.zeros((k, k))
        temp[:matrix.shape[0], :matrix.shape[1]] = matrix
        edg[i] = temp
    print("Edg Padded.")
    for i in range(len(nod)):
        matrix = nod[i]
        temp = np.full((k, matrix.shape[1]), -1)
        temp[:matrix.shape[0], :matrix.shape[1]] = matrix
        nod[i] = temp
    print("Nod Padded.")

    print("Stacking arrays")
    adj = np.stack(adj)
    gc.collect()
    print("Finish adj stack")
    edg = np.stack(edg)
    edg = edg.reshape((edg.shape[0], edg.shape[1], edg.shape[2], 1))
    print(edg.shape)
    gc.collect()
    print("Finish edg stack")
    nod = np.stack(nod)
    y = np.array(y)
    return adj, nod, edg, y


A, X, E, y = getData()
print("Data acquired")
# Parameters
N = X.shape[-2]  # Number of nodes in the graphs
F = X.shape[-1]  # Node features dimensionality
S = E.shape[-1]  # Edge features dimensionality
n_out = 2  # Dimensionality of the target
learning_rate = 1e-3  # Learning rate for SGD
epochs = 25  # Number of training epochs
batch_size = 8  # Batch size
es_patience = 5  # Patience fot early stopping

# Train/test split
A_train, A_test, \
X_train, X_test, \
E_train, E_test, \
y_train, y_test = train_test_split(A, X, E, y, test_size=0.1)

# A_train.tofile('a_train.np')
# X_train.tofile('x_train.np')
# E_train.tofile('e_train.np')
# y_train.tofile('y_train.np')
A_test.tofile('a_test' + str(A_test.shape) + '.np')
X_test.tofile('x_test' + str(X_test.shape) + '.np')
E_test.tofile('e_test' + str(E_test.shape) + '.np')
y_test.tofile('y_test' + str(y_test.shape) + '.np')

# Model definition
X_in = Input(shape=(N, F))
A_in = Input(shape=(N, N))
E_in = Input(shape=(N, N, S))

gc1 = EdgeConditionedConv(16, activation='relu')([X_in, A_in, E_in])
gc2 = EdgeConditionedConv(16, activation='relu')([gc1, A_in, E_in])
pool = GlobalAvgPool()(gc2)
output = Dense(n_out)(pool)

# Build model
model = Model(inputs=[X_in, A_in, E_in], outputs=output)
model.compile(optimizer=Adam(lr=.00004, clipnorm=1.), loss='sparse_categorical_crossentropy')
model.summary()

checkpoint = ModelCheckpoint('tego-' + str(int(time.time())) + '.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

# Train model
model.fit([X_train, A_train, E_train],
          y_train,
          batch_size=batch_size,
          validation_split=0.1,
          epochs=epochs,
          callbacks=[checkpoint])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([X_test, A_test, E_test],
                              y_test,
                              batch_size=batch_size)
print('Done.\n'
      'Test loss: {}'.format(eval_results))

correct = 0
total = 0
try:
    for ain, xin, ein, yout in zip(A_test, X_test, E_test, y_test):
        ain = ain.reshape((1, 215, 215))
        xin = xin.reshape((1, 215, 7))
        ein = ein.reshape((1, 215, 215, 1))
        prediction = model.predict(x=[xin, ain, ein])
        print("-------------------------")
        print("Prediction: " + str(prediction))
        print("Actual: " + str(yout))
        total += 1
        if (prediction[0][1] > prediction[0][0] and yout == 1) or (prediction[0][1] < prediction[0][0] and yout == 0):
            correct += 1
    print("Got " + str(correct) + " out of " + str(total))
except:
    print("Error testing acc.")
