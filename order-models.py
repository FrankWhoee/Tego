import os
import re

files = os.listdir(".")
models = []
for f in files:
    if f.endswith(".h5"):
        models.append(f)


def compare(item1):
    return int(re.search("\-(.*?)\.", item1).group()[1:-1])


models.sort(key=compare)
i = 0
for model in models:
    os.rename(model,"model-" + str(i) + ".h5")
    i += 1
