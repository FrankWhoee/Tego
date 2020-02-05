from Bio import Phylo
from ete3 import Tree
from datetime import datetime
import meta

nextStrain_date_format = "%Y-%m-%d"


def convertNodeToDatetime(node: Tree):
    return datetime.strptime(meta.get(node.name, ["Collection Data"]), nextStrain_date_format)


def trimNode(node: Tree):
    start_date = convertNodeToDatetime(node)
    trimNodeHelper(node, start_date)


def trimNodeHelper(node: Tree, start_date: datetime):
    for child in node.get_children():
        if convertNodeToDatetime(child) - start_date >= datetime.timedelta(0, 4.415e7):
            delDescendants(child)
        else:
            trimNodeHelper(child, start_date)


def delDescendants(node: Tree):
    for desc in node.get_descendants():
        desc.delete()


def getSuccess(node: Tree, prevLength=0):
    print("-----------------------INPUT----------------------")
    trainNode = node.copy()
    trimNode(trainNode)
    print(trainNode)
    initial = len(trainNode.get_descendants()) + 1
    future = node.copy()
    getSuccessHelper(future)
    print("-----------------------FUTURE----------------------")
    print(future)
    final = len(future.get_descendants()) + 1
    return final / initial


def getSuccessHelper(node: Tree, start_date: datetime):
    """
    Gets number of nodes within 3.4 years.
    :param node: The node that you want to get the success for
    :param prevLength:
    :return:
    """
    for child in node.get_children():
        if convertNodeToDatetime(child) - start_date >= datetime.timedelta(0, 1.072e8):
            delDescendants(child)
        else:
            trimNode(child, start_date)


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
    temp.write(outfile="subtrees/" + str(i) + "-(" + str(success) + ").tego")
    i += 1
print("Got " + str(i) + " subtrees")
print("Average success is " + str(avgSuccess / i))
print(fails, "failed trees.")
