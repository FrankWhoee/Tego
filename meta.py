import pandas as pd

meta = pd.read_csv("nextstrain_flu_seasonal_h3n2_ha_12y_metadata.tsv", sep='\t')
print("Tego[meta.py]: Metadata loaded.")
headers = list(meta.columns.values.tolist())
def get(strain: str, attributes: list = [], verbose = 9):
    data = {}
    if len(attributes) == 0:
        attributes = headers.copy()
    for attribute in attributes:
        if attribute not in headers:
            raise Exception(attribute + " not in headers.")
    for attribute in attributes:
        if attribute in meta:
            try:
                data[attribute] = meta.loc[meta["Strain"] == strain][attribute].tolist()[0]
            except IndexError:
                if verbose == 1:
                    print("Did not find " + strain)
    return data