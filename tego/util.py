import numpy as np
from Bio import Phylo
from ete3 import Tree
from meta import get as mget
import math
import os
import re
import gc


def to_distance_matrix(tree) -> np.ndarray:
    """Create a distance matrix (NumPy array) from clades/branches in tree.

    A cell (i,j) in the array is the length of the branch between allclades[i]
    and allclades[j], if a branch exists, otherwise infinity.

    Returns a tuple of (allclades, distance_matrix) where allclades is a list of
    clades and distance_matrix is a NumPy 2D array.
    """
    allclades = list(tree.find_clades(order='level'))
    lookup = {}
    for i, elem in enumerate(allclades):
        lookup[elem] = i
    distmat = np.repeat(-1, len(allclades) ** 2)
    distmat.shape = (len(allclades), len(allclades))
    for parent in tree.find_clades(terminal=False, order='level'):
        for child in parent.clades:
            if child.branch_length:
                distmat[lookup[parent], lookup[child]] = child.branch_length
    if not tree.rooted:
        distmat += distmat.transpose()
    return np.array(distmat)


def to_adjacency_matrix(tree) -> np.ndarray:
    """Create an adjacency matrix (NumPy array) from clades/branches in tree.

    Also returns a list of all clades in tree ("allclades"), where the position
    of each clade in the list corresponds to a row and column of the np
    array: a cell (i,j) in the array is 1 if there is a branch from allclades[i]
    to allclades[j], otherwise 0.

    Returns a tuple of (allclades, adjacency_matrix) where allclades is a list
    of clades and adjacency_matrix is a NumPy 2D array.
    """
    allclades = list(tree.find_clades(order='level'))
    lookup = {}
    for i, elem in enumerate(allclades):
        lookup[elem] = i
    adjmat = np.zeros((len(allclades), len(allclades)))
    for parent in tree.find_clades(terminal=False, order='level'):
        for child in parent.clades:
            adjmat[lookup[parent], lookup[child]] = 1
    if not tree.rooted:
        # Branches can go from "child" to "parent" in unrooted trees
        adjmat += adjmat.transpose()
    return np.array(adjmat)


def to_node_attributes(path) -> np.ndarray:
    result = []
    index_map = {
        "Antigenic advance (tree model)": 0,
        "Antigenic advance (sub model)": 1,
        "Epitope mutations": 2,
        "Local branching index": 3,
        "Non-epitope mutations": 4,
        "ne_star": 5,
        "RBS adjacent mutations": 6
    }
    node = Tree(path, format=3)
    node_arr = node.get_children().copy()
    node_arr.append(node)
    for child in node_arr:
        attributes = mget(child.name,
                          ["Antigenic advance (tree model)", "Antigenic advance (sub model)", "Epitope mutations",
                           "Local branching index", "Non-epitope mutations", "ne_star", "RBS adjacent mutations"])
        node_matrix = np.full(7, -1, dtype=np.dtype(np.float))
        for key in attributes.keys():
            if math.isnan(attributes[key]):
                node_matrix[index_map[key]] = -1
            else:
                node_matrix[index_map[key]] = attributes[key]
        result.append(node_matrix)
    return np.stack(result)


def resize(array: np.ndarray, new_size):
    new = np.zeros(new_size)
    new[:array.shape[0], :array.shape[1]] = array
    return new


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
        temp = np.full((k, matrix.shape[1]), -1, dtype=np.dtype(np.float))
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


def validate(A, X, E, y, model, verbose=0):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from matplotlib import pyplot as plt
    actual = []
    pred = []
    total = 0
    correct = 0
    for ain, xin, ein, yout in zip(A, X, E, y):
        ain = ain.reshape((1, ain.shape[0], ain.shape[1]))
        xin = xin.reshape((1, xin.shape[0], xin.shape[1]))
        ein = ein.reshape((1, ein.shape[0], ein.shape[1], ein.shape[2]))
        prediction = model.predict(x=[xin, ain, ein])
        if verbose == 1:
            print("-------------------------")
            print("Prediction: " + str(prediction))
            print("Actual: " + str(yout))
            print("Has nan: " + str((np.isnan(ain).any() or np.isnan(xin).any() or np.isnan(ein).any())))
        total += 1
        if (prediction[0][1] > prediction[0][0] and yout == 1) or (
                prediction[0][1] < prediction[0][0] and yout == 0):
            correct += 1
        actual.append(yout)
        pred.append(1 if prediction[0][1] > prediction[0][0] else 0)
    print(actual)
    print(pred)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(actual,pred)
    print("FPR: " + str(fpr_keras))
    print("TPR: " + str(tpr_keras))
    print(thresholds_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.plot(fpr_keras, tpr_keras)
    plt.savefig("auc-roc.png")
    print("AUC: " + str(auc_keras))
    return correct, total


def cross_validate(a,x,e,y):
    from keras.layers import Input, Dense, Dropout
    from keras.models import Model
    from spektral.layers import EdgeConditionedConv, GlobalAvgPool
    from keras.optimizers import Adam
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    cvscores = []
    # Parameters
    N = a.shape[-2]  # Number of nodes in the graphs
    F = x.shape[-1]  # Node features dimensionality
    S = e.shape[-1]  # Edge features dimensionality
    n_out = 2  # Dimensionality of the target
    epochs = 10  # Number of training epochs
    batch_size = 8  # Batch size
    k_folds = 6
    i = 0
    data_size = a.shape[0]
    print("{} points of data.".format(data_size))
    while i <= int(data_size/k_folds) * (k_folds - 1):
        # Model definition
        X_in = Input(shape=(N, F))
        A_in = Input(shape=(N, N))
        E_in = Input(shape=(N, N, S))

        gc1 = EdgeConditionedConv(30, activation='relu')([X_in, A_in, E_in])
        gc2 = EdgeConditionedConv(30, activation='relu')([gc1, A_in, E_in])
        pool = GlobalAvgPool()(gc2)
        output = Dense(n_out)(pool)

        # Build model
        model = Model(inputs=[X_in, A_in, E_in], outputs=output)
        model.compile(optimizer=Adam(lr=.00004, clipnorm=1.), loss='sparse_categorical_crossentropy')
        end = int(i + data_size/k_folds)
        print("Getting {} to {}".format(i, end))

        X = np.delete(x,range(i,end))
        A = np.delete(a, range(i, end))
        E = np.delete(e, range(i, end))
        Y = np.delete(y, range(i, end))

        X_test = x[i:end]
        A_test = a[i:end]
        E_test = e[i:end]
        y_test = y[i:end]

        print(x.shape)
        print(a.shape)
        print(e.shape)
        print(y.shape)

        # Train model
        model.fit([X,A,E],
                  Y,
                  batch_size=batch_size,
                  epochs=epochs)
        # evaluate the model
        correct, total = validate( A_test,X_test, E_test, y_test, model)
        print("%s: %.2f%%" % ("accuracy", (correct/total) * 100))
        cvscores.append((correct/total) * 100)
        i += int(data_size/k_folds)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
