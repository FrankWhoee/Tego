import numpy as np
from Bio import Phylo
from spektral.utils.misc import pad_jagged_array
import os
import re
from tego.util import to_adjacency_matrix, to_distance_matrix
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from spektral.layers import EdgeConditionedConv, GlobalAvgPool
import gc

print("Imported packages.")


def nwkToNumpy(path) -> (np.ndarray, np.ndarray):
    tree = Phylo.read(path, "newick")
    A = to_adjacency_matrix(tree)
    E = to_distance_matrix(tree)
    E = np.reshape(E, (E.shape[0], E.shape[1], 1))
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
        if A.shape[0] <= 1600:
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
        temp = np.zeros((k, k, 1))
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
S = E.shape[3]  # Edge features dimensionality
n_out = y.shape[-1]  # Dimensionality of the target
learning_rate = 1e-3  # Learning rate for SGD
epochs = 25  # Number of training epochs
batch_size = 32  # Batch size
es_patience = 5  # Patience fot early stopping

# Train/test split
A_train, A_test, \
E_train, E_test, \
y_train, y_test = train_test_split(A, E, y, test_size=0.1)
print("Training/testing split.")
# Model definition
A_in = Input(shape=(N, N))
E_in = Input(shape=(N, N, S))
print("Model input defined")
gc1 = EdgeConditionedConv(32, activation='relu')([A_in, E_in])
gc2 = EdgeConditionedConv(32, activation='relu')([gc1, A_in, E_in])
pool = GlobalAvgPool()(gc2)
output = Dense(n_out)(pool)
print("Rest of model is defined.")
# Build model
model = Model(inputs=[A_in, E_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
model.summary()
print("Model built.")
# Train model
print("Training model...")
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

plt.figure()
plt.scatter(preds, y_test, alpha=0.3)
plt.plot(range(-6, 6), range(-6, 6))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('pred_v_true.png')
