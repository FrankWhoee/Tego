from keras.models import load_model
import numpy as np
import spektral
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from spektral.layers import EdgeConditionedConv, GlobalAvgPool
# Model definition
from spektral.layers import EdgeConditionedConv

A = np.fromfile("a_test.np")
X = np.fromfile("a_test.np")
E = np.fromfile("a_test.np")
y = np.fromfile("a_test.np")


print(A.shape)

# Parameters
N = X.shape[-2]  # Number of nodes in the graphs
F = X.shape[-1]  # Node features dimensionality
S = E.shape[-1]  # Edge features dimensionality
n_out = 2  # Dimensionality of the target
learning_rate = 1e-3  # Learning rate for SGD
epochs = 25  # Number of training epochs
batch_size = 8  # Batch size
es_patience = 5  # Patience fot early stopping

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

model.load_weights('tego-1581113522.h5')

