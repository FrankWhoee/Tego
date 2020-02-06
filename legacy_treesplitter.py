from Bio import Phylo
from ete3 import Tree

def trimNode(node: Tree, prevLength=0):
    for child in node.get_children():
        if child.dist + prevLength >= 2:
            delDescendants(child)
        else:
            trimNode(child, prevLength=prevLength + child.dist)

def delDescendants(node:Tree):
    for desc in node.get_descendants():
        desc.delete()

def getSuccess(node:Tree, prevLength=0):
    trainNode = node.copy()
    trimNode(trainNode)
    initial = len(trainNode.get_descendants()) + 1
    future = node.copy()
    getSuccessHelper(future)
    final = len(future.get_descendants()) + 1
    return final/initial


def getSuccessHelper(node:Tree, prevLength=0):
    """
    Gets number of nodes within 3.4 years.
    :param node: The node that you want to get the success for
    :param prevLength:
    :return:
    """
    for child in node.get_children():
        if child.dist + prevLength > 20:
            delDescendants(child)
        else:
            trimNode(child, prevLength=prevLength + child.dist)

subtrees = []
tree = Tree("nextstrain_flu_seasonal_h3n2_ha_12y_timetree.nwk", format=3)
i = 0
avgSuccess = 0
fails = 0
for node in tree.traverse("preorder"):
    if node.is_leaf():
        continue
    temp = node.copy()
    trimNode(temp)
    subtrees.append(temp)
    success = getSuccess(node)
    if success == 1.0:
        fails += 1
    avgSuccess += success
    temp.write(outfile="subtrees/" + str(i) + "-("+str(success)+").tego")
    i += 1
print("Got " + str(i) + " subtrees")
print("Average success is " + str(avgSuccess/i))
print(fails, "failed trees.")