import os
import pandas as pd
from pymatgen.core import Structure
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor


def load_structure(path):
    try:
        return Structure.from_file(path)
    except Exception as e:
        print(f"Failed: {path} -> {e}")
        return None


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("qmof.csv", low_memory=False)

    # Build CIF paths
    df["cif_path"] = df["qmof_id"].apply(
        lambda x: os.path.join("relaxed_structures", f"{x}.cif")
    )

    # Check files
    df["cif_exists"] = df["cif_path"].apply(os.path.exists)
    print(df["cif_exists"].value_counts())

    target = "outputs.pbe.bandgap"

    # Keep valid rows
    df = df[df["cif_exists"]].copy()
    df = df[df[target].notna()].copy()

    # Small test subset

    # Load structures
    df["structure"] = df["cif_path"].apply(load_structure)
    df = df[df["structure"].notnull()].copy()

    print("Loaded structures:", len(df))

    # Structure features
    df = DensityFeatures().featurize_dataframe(
        df, "structure", ignore_errors=True
    )

    df = GlobalSymmetryFeatures().featurize_dataframe(
        df, "structure", ignore_errors=True
    )

    print("After featurization:", df.shape)

    # Build X and y
    drop_cols = [
        "qmof_id", "name", "cif_path", "cif_exists", "structure", target
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include="number")

    # Remove leakage if present
    leakage_cols = [
        "outputs.pbe.cbm",
        "outputs.pbe.vbm",
        "outputs.pbe.directgap",
        "outputs.pbe.energy_total",
        "outputs.pbe.energy_vdw",
        "outputs.pbe.energy_elec",
    ]
    X = X.drop(columns=leakage_cols, errors="ignore")

    y = df[target]

    print("Feature matrix shape:", X.shape)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, pred))
    print('RMS', root_mean_squared_error(y_test, pred))
    print("Number of features:", X.shape[1])

    import matplotlib.pyplot as plt

    plt.scatter(y_test, pred)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.show()
