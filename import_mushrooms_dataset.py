import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.impute import SimpleImputer
def import_mushrooms_dataset(nrows):
    #wczytanie datasetu
    data_p = pd.read_csv("datasets/mushrooms.csv", index_col=False, header=0, delimiter=";", nrows=nrows)

    #rozdzielenie danych numerycznych od reszty, w celu późniejszej konwersji
    numeric_features = data_p.select_dtypes(include=[np.number]).columns
    categorical_features = data_p.select_dtypes(include=[object]).columns

    #uzupełnienie pustych wartości
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    numeric_imputer = SimpleImputer(strategy='mean')

    # Dopasowanie i transformacja danych numerycznych i kategorycznych
    data_p[categorical_features] = categorical_imputer.fit_transform(data_p[categorical_features])
    data_p[numeric_features] = numeric_imputer.fit_transform(data_p[numeric_features])

    #przeniesienie etykiet na koniec
    data_p['class'] = data_p.pop('class')

    #zapisanie i otworzenie przez numpy
    data_p.to_csv("mushrooms_formatted.csv", index=False)
    data_mushrooms = np.loadtxt("mushrooms_formatted.csv", delimiter=",", dtype=object)

    #rozdzielenie etykiet od atrybutów
    X_mushrooms = data_mushrooms[1:, :-1]
    y_mushrooms = data_mushrooms[1:, -1]

    #kategoryzcne transformacja na numeryczne
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_mushrooms = encoder.fit_transform(X_mushrooms)

    #zakodowanie etykiet
    y_mushrooms[y_mushrooms == 'p'] = 0
    y_mushrooms[y_mushrooms == 'e'] = 1
    #konwersja do integer
    y_mushrooms = y_mushrooms.astype(int)
    return X_mushrooms, y_mushrooms