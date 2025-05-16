from rdkit import Chem
import pandas as pd


suppl = Chem.SDMolSupplier("enamine_bbs_costs.sdf")
smiles_list = []
data_dict = {}

for mol in suppl:

    if mol is None:
        continue

    smiles_list.append(Chem.MolToSmiles(mol))

    for prop in mol.GetPropNames():

        if prop not in data_dict:
            data_dict[prop] = [None] * (len(smiles_list)-1)

        data_dict[prop].append(mol.GetProp(prop))

    for prop in data_dict:
        if len(data_dict[prop]) < len(smiles_list):
            data_dict[prop].append(None)

df = pd.DataFrame(data_dict)
df["smiles"] = smiles_list

with open("enamine_bbs.txt", "r") as file:
    building_blocks = file.read().splitlines()

df_subset = df[df["smiles"].isin(building_blocks)]
df_subset["Price_EUR_100mg"] = df_subset["Price_EUR_100mg"].astype(float)
df_subset["Price_EUR_100mg"] = df_subset["Price_EUR_100mg"].fillna(df_subset["Price_EUR_100mg"].median())
df_subset["score"] = df_subset["Price_EUR_100mg"] / df_subset["Price_EUR_100mg"].max()

df_subset[["smiles", "score"]].to_csv("building_blocks_costs.csv", index=False)
