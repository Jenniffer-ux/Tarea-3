import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def cargar_y_preprocesar(path="data/synthetic_data.csv"):
    df = pd.read_csv(path)
    X = df.drop("demanda", axis=1)
    y = df["demanda"].values.reshape(-1, 1)

    preprocessor = ColumnTransformer([
        ("scale", StandardScaler(), ["precio", "descuento", "stock", "mes"]),
        ("onehot", OneHotEncoder(sparse_output=False), ["categoria"])
    ])

    X_pre = preprocessor.fit_transform(X)
    y = (y - y.mean()) / y.std()

    return train_test_split(X_pre, y, test_size=0.2, random_state=42)
