import numpy as np
from Bio import Phylo
import os
import re
from tego.util import to_adjacency_matrix, to_distance_matrix, to_node_attributes
import gc
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from spektral.layers import EdgeConditionedConv, GlobalAvgPool

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
        success = float(re.search("\((.*?)\)", f).group()[1:-1])
        if (successful < failed and success == 1) or (failed < successful and success == 0) or (failed == successful):
            # print("Get " + str(i) + " out of " + str(len(files) - 1))
            A, X, E = nwkToNumpy('subtrees/' + f)
            adj.append(A)
            nod.append(X)
            edg.append(E)
            y.append(success)
            if success > 1:
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
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy')
model.summary()

# Train model
model.fit([X_train, A_train, E_train],
          y_train,
          batch_size=batch_size,
          validation_split=0.1,
          epochs=epochs,
          callbacks=[
              EarlyStopping(patience=es_patience, restore_best_weights=True)
          ])

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([X_test, A_test, E_test],
                              y_test,
                              batch_size=batch_size)
print('Done.\n'
      'Test loss: {}'.format(eval_results))

# Plot predictions
preds = model.predict([X_test, A_test, E_test])
