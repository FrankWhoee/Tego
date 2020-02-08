from keras.models import load_model
import numpy as np
import spektral
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from spektral.layers import EdgeConditionedConv, GlobalAvgPool
# Model definition
from spektral.layers import EdgeConditionedConv

A = np.fromfile("a_test(88, 215, 215).np").reshape((88, 215, 215))
X = np.fromfile("x_test(88, 215, 7).np").reshape((88, 215, 7))
E = np.fromfile("e_test(88, 215, 215, 1).np").reshape((88, 215, 215, 1))
y = np.fromfile("y_test(88,).np").reshape((88))

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

model.load_weights('tego-1581134625.h5')

try:
    for ain, xin, ein, yout in zip(A, X, E, y):
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
