import numpy as np
from Bio import Phylo
from spektral.utils import nx_to_numpy
from spektral.layers import GraphConv
from keras.models import Model
from keras.layers import Input, Dropout
from spektral import utils

tree = Phylo.read('h3n2-ha-12y.nwk', 'newick')

train = nx_to_numpy(Phylo.to_networkx(tree))

A, X, E = train
y = np.zeros(())

N = A.shape[0]
F = X.shape[-1]
n_classes = y.shape[-1]

# Model definition
X_in = Input(shape=(F, ))  # Input layer for X
A_in = Input((N, ), sparse=True)  # Input layer for A

graph_conv_1 = GraphConv(16, activation='relu')([X_in, A_in])
dropout = Dropout(0.5)(graph_conv_1)
graph_conv_2 = GraphConv(n_classes, activation='softmax')([dropout, A_in])

# Build model
model = Model(inputs=[X_in, A_in], outputs=graph_conv_2)
A = utils.localpooling_filter(A)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

validation_data = ([X, A], y)
model.fit([X, A],
          y,
          epochs=100,
          batch_size=N,
          validation_data=validation_data,
          shuffle=False)

# Evaluate model
eval_results = model.evaluate([X, A],
                              y,
                              batch_size=N)
print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))