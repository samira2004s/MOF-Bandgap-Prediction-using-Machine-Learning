import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from pymatgen.core import Structure
import matplotlib.pyplot as plt



import os
from pymatgen.core import Structure


mof_data = pd.read_csv(
    "qmof.csv",
    usecols=["info.formula", "outputs.pbe.bandgap"],
    low_memory=False
)
#for further manipulation and calculation only the chemical formula and bandgap are needed

mof_data = mof_data.dropna()
#renaming the columns to make the dataset more readable
mof_data.columns = ["formula", "bandgap"]
#print(mof_data.head())

#print(mof_data['bandgap'].describe()) # most values are around 2eV indicating that many MOFs are semiconductors.
#The dataset includes the metals and insulating materials as well
#print((mof_data["bandgap"] < 0.1).mean()) #around 1% of MOFs are metals
def classify(bg):
    if bg < 0.1:
        return "metal"
    elif bg < 3:
        return "semiconductor"
    else:
        return "insulator"

mof_data["class"] = mof_data["bandgap"].apply(classify)
#print(mof_data.head())

# Create composition column
mof_data["composition"] = mof_data["formula"].apply(Composition)

# Initialize featurizer
featurizer = ElementProperty.from_preset("magpie")

# Get feature names
feature_labels = featurizer.feature_labels()

# Featurize safely
features = mof_data["composition"].apply(
    lambda x: featurizer.featurize(x)
)

# Convert to dataframe
features_df = pd.DataFrame(
    features.tolist(),
    columns=feature_labels,
    index=mof_data.index
)

# Combine
mof_data = pd.concat([mof_data, features_df], axis=1)

print(mof_data.head())
print(mof_data.columns)

X = mof_data.drop(columns=["formula", "bandgap", "composition", "class"], errors="ignore")
y = mof_data["bandgap"]
print(X.isna().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

importance = pd.Series(model.feature_importances_, index=X.columns)
top_features = importance.sort_values(ascending=False).head(10)

print(top_features)

plt.figure(figsize=(8,5))

plt.hist(y, bins=60)
plt.xlabel("Band Gap (eV)")
plt.ylabel("Count")
plt.title("Distribution of MOF Band Gaps")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,6))

plt.scatter(y_test, y_pred, alpha=0.2)

# Perfect prediction line
plt.plot([0, 6], [0, 6], linestyle="--")

plt.xlabel("Actual Band Gap (eV)")
plt.ylabel("Predicted Band Gap (eV)")
plt.title("Predicted vs Actual Band Gap")

plt.xlim(0, 6)
plt.ylim(0, 6)

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


cif_folder = r"C:\Users\Samira\PycharmProjects\PYTHON2\relaxed_structures"
structures = {}

for file in os.listdir(cif_folder):
    if file.endswith('.cif'):
        mof_id = file.replace(".cif", "")
        structure = Structure.from_file(os.path.join(cif_folder, file))
        structures[mof_id] = structure

print(len(structures))
