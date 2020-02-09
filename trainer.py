import time
from tego.util import getData, validate
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from spektral.layers import EdgeConditionedConv, GlobalAvgPool
from keras.optimizers import Adam

print("Imported packages.")

A, X, E, y = getData()
print("Data acquired")
# Parameters
N = X.shape[-2]  # Number of nodes in the graphs
F = X.shape[-1]  # Node features dimensionality
S = E.shape[-1]  # Edge features dimensionality
n_out = 2  # Dimensionality of the target
learning_rate = 1e-3  # Learning rate for SGD
epochs = 50  # Number of training epochs
batch_size = 8  # Batch size
es_patience = 5  # Patience fot early stopping

# A,X,E,y = shuffle(A,X,E,y)

print(X)

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
# dense1 = Dense(64)(pool)
# dropout1 = Dropout(0.5)(dense1)
output = Dense(n_out)(pool)

from keras.optimizers import SGD

opt = SGD(lr=0.01, momentum=0.9, decay=0.1)

# Build model
model = Model(inputs=[X_in, A_in, E_in], outputs=output)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
model.summary()

checkpoint = ModelCheckpoint('tego-' + str(int(time.time())) + '.h5', monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

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

correct, total = validate(A_test,X_test,E_test,y_test,model)
print("Got " + str(correct) + " out of " + str(total))