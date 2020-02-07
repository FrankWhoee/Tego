from ete3 import Tree


def trimNode(node: Tree, prevLength=0):
    for child in node.get_children():
        if child.dist + prevLength >= 1:
            delDescendants(child)
        else:
            trimNode(child, prevLength=prevLength + child.dist)


def delDescendants(node: Tree):
    for desc in node.get_descendants():
        desc.delete()


def getSuccess(node: Tree):
    trainNode = node.copy()
    trimNode(trainNode)
    initial = len(trainNode.get_descendants()) + 1
    future = node.copy()
    getSuccessHelper(future)
    final = len(future.get_descendants()) + 1
    # if final != initial:
    #     print(trainNode)
    #     print(future)
    return final / initial


def getSuccessHelper(node: Tree, prevLength=0):
    """
    Gets number of nodes within 3.4 years.
    :param node: The node that you want to get the success for
    :param prevLength:
    :return:
    """
    for child in node.get_children():
        if child.dist + prevLength >= 3.7:
            delDescendants(child)
        else:
            getSuccessHelper(child, prevLength=prevLength + child.dist)


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
    if success > 1:
        score = "-(1)"
    else:
        score = "-(0)"
    temp.write(outfile="subtrees/" + str(i) + score + ".tego")
    i += 1
print("Got " + str(i) + " subtrees")
print("Average success is " + str(avgSuccess / i))
print(fails, "failed trees.")
