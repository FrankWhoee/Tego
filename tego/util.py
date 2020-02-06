import numpy as np
from Bio import Phylo
from ete3 import Tree
from meta import get as mget
import math

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
    distmat = np.repeat(np.inf, len(allclades) ** 2)
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
        node_matrix = np.full(7, -1)
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
