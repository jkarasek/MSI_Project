import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.impute import SimpleImputer

def import_penguins_dataset():
        # Wczytanie datasetu z pingwinami (z użyciem pandy łatwiej obrabiać)
        data_p = pd.read_csv("datasets/penguins.csv", sep=",", skipinitialspace=True, quotechar='"', quoting=1,
                             index_col=False, header=0)

        ################ Rozwiązanie problemu brakujących danych #################
        # Sprawdzenie brakujących danych przed poprawkami
        #     Tools.missingChecker(data_p)
        #     data_p.to_csv("b_f_data_p.csv", index=False)      #data_p przed formatowaniem (before format data_p)

        # Usunięcie kolumn "studyName", "Sample Number", "Stage", "Individual ID", "Clutch Completion", "Date Egg", "Comments" bo są useless
        data_p = data_p.drop(
            columns=["studyName", "Sample Number", "Stage", "Individual ID", "Clutch Completion", "Date Egg",
                     "Comments"])

        # Podział danych na numeryczne i kategoryczne
        numeric_features = data_p.select_dtypes(include=[np.number]).columns
        categorical_features = data_p.select_dtypes(include=[object]).columns

        # Imputer dla danych numerycznych i kategorycznych
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        numeric_imputer = SimpleImputer(strategy='mean')

        # Dopasowanie i transformacja danych numerycznych i kategorycznych
        data_p[numeric_features] = numeric_imputer.fit_transform(data_p[numeric_features])
        data_p[categorical_features] = categorical_imputer.fit_transform(data_p[categorical_features])

        # Przesunięcie kolumny z etykietami ("Species") na ostatnią pozycję
        data_p['Species'] = data_p.pop('Species')

        # Sprawdzenie danych po poprawkach, oraz zapisanie ich w formacie numpy (numpy nie sypie błędów przy kodowaniu etykiet z jakiegoś powodu)
        #     Tools.missingChecker(data_p)
        data_p.to_csv("penguins_formatted.csv", index=False)
        data_penguins = np.loadtxt("penguins_formatted.csv", delimiter=",", dtype=object)

        # Podział na nazwy kolumn, cechy i etykiety
        p_column_names = data_penguins[0, :]
        X_penguins = data_penguins[1:, :-1]
        y_penguins = data_penguins[1:, -1]

        # Konwersja danych kategorycznych na numeryczne
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        X_penguins = encoder.fit_transform(X_penguins)

        # Zakodowanie etykiet klas
        y_penguins[y_penguins == 'Adelie Penguin (Pygoscelis adeliae)'] = 0
        y_penguins[y_penguins == 'Gentoo penguin (Pygoscelis papua)'] = 1
        y_penguins[y_penguins == 'Chinstrap penguin (Pygoscelis antarctica)'] = 2

        y_penguins = y_penguins.astype(int)
        return X_penguins, y_penguins