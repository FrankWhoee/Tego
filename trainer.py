import numpy as np
from Bio import Phylo
import os
import re
from tego.util import to_adjacency_matrix, to_distance_matrix
import numpy as np
import gc
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

from spektral.datasets import delaunay
from spektral.layers import GraphAttention, GlobalAttentionPool

print("Imported packages.")


def nwkToNumpy(path) -> (np.ndarray, np.ndarray):
    tree = Phylo.read(path, "newick")
    A = to_adjacency_matrix(tree)
    E = to_distance_matrix(tree)
    # E = np.reshape(E, (E.shape[0], E.shape[1], 1))
    return A, E


def getData(path="subtrees"):
    y = []
    adj = []
    edg = []
    i = 0
    files = os.listdir(path)
    for f in files:
        # print("Get " + str(i) + " out of " + str(len(files) - 1))
        A, E = nwkToNumpy('subtrees/' + f)
        adj.append(A)
        edg.append(E)
        y.append(re.search("\((.*?)\)", f).group()[1:-1])

        i += 1
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

    print("Stacking arrays")
    adj = np.stack(adj)
    gc.collect()
    print("Finish adj stack")
    edg = np.stack(edg)
    gc.collect()
    print("Finish edg stack")
    y = np.array(y)
    return adj, edg, y


A, E, y = getData()
print("Data acquired")
# Parameters
N = A.shape[1]  # Number of nodes in the graphs
S = N  # Edge features dimensionality
n_classes = 2  # Number of classes
l2_reg = 5e-4            # Regularization rate for l2
learning_rate = 1e-3     # Learning rate for Adam
epochs = 20000           # Number of training epochs
batch_size = 32          # Batch size
es_patience = 200        # Patience fot early stopping

# Train/test split
A_train, A_test, \
e_train, e_test, \
y_train, y_test = train_test_split(A, E, y, test_size=0.1)
print("Training/testing split.")

# Model definition
E_in = Input(shape=(N, S))
A_in = Input((N, N))

gc1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([E_in, A_in])
gc2 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([gc1, A_in])
pool = GlobalAttentionPool(128)(gc2)

output = Dense(n_classes, activation='softmax')(pool)

# Build model
model = Model(inputs=[E_in, A_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
model.summary()

# Train model
model.fit([e_train, A_train],
          y_train,
          batch_size=batch_size,
          validation_split=0.1,
          epochs=epochs,
          callbacks=[
              EarlyStopping(patience=es_patience, restore_best_weights=True)
          ],
          verbose=2)

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([e_test, A_test],
                              y_test,
                              batch_size=batch_size)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))