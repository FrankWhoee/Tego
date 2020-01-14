import pandas as pd

meta = pd.read_csv("nextstrain_flu_seasonal_h3n2_ha_12y_metadata.tsv", sep='\t')
print("Metadata loaded.")
headers = list(meta.columns.values.tolist())


def get(strain: str, attributes: list = []):
    data = {}
    if len(attributes) == 0:
        attributes = headers.copy()
    for attribute in attributes:
        if attribute not in headers:
            raise Exception(attribute + " not in headers.")
    for attribute in attributes:
        if attribute in meta:
            print(attribute)
            print(meta.loc[meta[attribute] == strain])
            print("------------------------------------------------------------------")
            data[attribute] = meta.loc[meta[attribute] == strain]
    return data


print(get("A/Kenya/1599/2008", []))
