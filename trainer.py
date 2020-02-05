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
    for f in os.listdir(path):
        y.append(re.search("\((.*?)\)", f).group()[1:-1])
        A, E = nwkToNumpy('subtrees/' + f)
        adj.append(A)
        edg.append(E)

    k = max([_.shape[-1] for _ in adj])
    adj = pad_jagged_array(adj, (k, k))
    edg = pad_jagged_array(edg, (k, k, -1))
    return adj, edg, y


A, E, y = getData()

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
X_train, X_test, \
E_train, E_test, \
y_train, y_test = train_test_split(A, E, y, test_size=0.1)

# Model definition
A_in = Input(shape=(N, N))
E_in = Input(shape=(N, N, S))

gc1 = EdgeConditionedConv(32, activation='relu')([A_in, E_in])
gc2 = EdgeConditionedConv(32, activation='relu')([gc1, A_in, E_in])
pool = GlobalAvgPool()(gc2)
output = Dense(n_out)(pool)

# Build model
model = Model(inputs=[A_in, E_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='mse')
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

plt.figure()
plt.scatter(preds, y_test, alpha=0.3)
plt.plot(range(-6, 6), range(-6, 6))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('pred_v_true.png')
