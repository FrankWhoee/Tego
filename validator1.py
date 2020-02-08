import sys
from tego.util import getData, validate
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split
from spektral.layers import EdgeConditionedConv, GlobalAvgPool

print("Imported packages.")

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

model.load_weights(sys.argv[1])
correct, total = validate(A_test,X_test,E_test,y_test,model)
print("Got " + str(correct) + " out of " + str(total))