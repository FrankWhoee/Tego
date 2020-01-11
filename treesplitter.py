from Bio import Phylo
from ete3 import Tree

subtrees = []
tree = Tree("nextstrain_flu_seasonal_h3n2_ha_12y_timetree.nwk",format=3)

for node in tree.traverse("preorder"):
    temp = node.copy()
    for child in temp.get_children():
        for desc in child.get_descendants():
            desc.delete()
    subtrees.append(temp)
    print(temp)

print(tree)

def trimNode(node: Tree, prevLength=0):
    tree = node.copy()
    for child in tree.get_children():
        if child.dist > (1.4 + prevLength):
            delDescendants(child)
            child.delete()
        else:
            trimNode(child, prevLength=child.dist)

def delDescendants(node:Tree):
    for desc in node.get_descendants():
        desc.delete()