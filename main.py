import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.base import clone, BaseEstimator, ClassifierMixin


class Adaboost(BaseEstimator, ClassifierMixin):
    def __init__(self, T):
        self.T = T
        self.alphas = []
        self.errors = []
        self.weak_learn = []

    def calculate_error(self, y, y_pred, weight_t):
        # Obliczanie błędu jako suma wag dla błędnych klasyfikacji
        return np.sum(weight_t * (y != y_pred))

    def calculate_alpha(self, error):
        # Obliczanie alpha z uwzględnieniem unikania dzielenia przez zero
        return 0.5 * np.log((1 - error) / max(error, 1e-10))

    def new_weights_t(self, alpha, weight_t, y, y_pred):
        # Nowe wagi z uwzględnieniem normalizacji
        new_weights = weight_t * np.exp(-alpha * y * y_pred)
        return new_weights / new_weights.sum()

    def fit(self, X, y):
        n_samples = len(y)
        weight_t = np.ones(n_samples) / n_samples

        for t in range(self.T):
            weak_learn = DecisionTreeClassifier(max_depth=1)
            weak_learn.fit(X, y, sample_weight=weight_t)
            y_pred = weak_learn.predict(X)
            self.weak_learn.append(weak_learn)

            error = self.calculate_error(y, y_pred, weight_t)
            self.errors.append(error)
            alpha = self.calculate_alpha(error)
            self.alphas.append(alpha)

            weight_t = self.new_weights_t(alpha, weight_t, y, y_pred)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for t in range(self.T):
            y_pred_t = self.weak_learn[t].predict(X) * self.alphas[t]
            y_pred += y_pred_t
        return np.sign(y_pred).astype(int)

class Tools:
    @staticmethod
    def missingChecker(data):
        # Sprawdzenie liczby brakujących wartości w każdej kolumnie
        missing_values = data.isnull().sum()
        print("Liczba brakujących wartości w każdej kolumnie:")
        print(missing_values)

        # Wyświetlenie dokładnych pozycji brakujących wartości
        missing_data_positions = data.isnull().stack()
        print("\nDokładne pozycje brakujących wartości (wiersz, kolumna):")
        print(missing_data_positions[missing_data_positions == True])


if __name__ == '__main__':

# Wczytanie datasetów (można to póżniej zamknąć w jakiejś metodzie, najlepiej na każdy dataset oddzielna)

# #Chwilowo przydatne do przeformatowanie pliku jak cos###################
#     with open("datasets/penguins.csv", "r") as file:
#         data = file.read()
#
#     data = data.replace('"', '')
#
#     with open("datasets/penguins.csv", "w") as file:
#         file.write(data)
# #######################################################################


# Wczytanie datasetu z pingwinami (z użyciem pandy łatwiej obrabiać)
    data_p = pd.read_csv("datasets/penguins.csv", sep=",", skipinitialspace=True, quotechar='"', quoting=1, index_col=False, header=0)


################ Rozwiązanie problemu brakujących danych #################
# Sprawdzenie brakujących danych przed poprawkami
#     Tools.missingChecker(data_p)
#     data_p.to_csv("b_f_data_p.csv", index=False)      #data_p przed formatowaniem (before format data_p)

# Usunięcie kolumn "studyName", "Sample Number", "Stage", "Individual ID", "Clutch Completion", "Date Egg", "Comments" bo są useless
    data_p = data_p.drop(columns=["studyName", "Sample Number", "Stage", "Individual ID", "Clutch Completion", "Date Egg", "Comments"])

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


# # Dataset z mushroomami
#     # X_mushrooms, y_mushrooms
#     # exit(1)


# Dane syntetyczne
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)


# Słownik datasetów
    d_set = {'Synthetic Data (make_classification)': (X, y),
             'Penguins data set': (X_penguins, y_penguins),
             # 'mushrooms_data_set': (X_mushrooms, y_mushrooms),
             }

# Słownik klasyfikatorów
    clfs = {
        'AdaBoost_50': Adaboost(T=50),
        'AdaBoost_100': Adaboost(T=100),
        'Random_Forrest': RandomForestClassifier(max_depth=2),
        'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=1),
    }

# Walidacja krzyżowa
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

#Zadeklarowanie macierzy wyników
    results = np.zeros(shape=[len(clfs), rskf.get_n_splits()])

# Cała magia
    for i, dset in enumerate(d_set):
        X, y = d_set[dset]
        print("\nDataset: ", dset)

        for fold, (train, test) in enumerate(rskf.split(X, y)):
            print(f"\nFold {fold}:")

            for clf_idx, clfn in enumerate(clfs):
                clf = clone(clfs[clfn])

                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])

                results[clf_idx, fold] = accuracy_score(y_true=y[test], y_pred=y_pred)
                # results[clf_idx, fold] = ballanced_accuracy_score(y_true=y[test], y_pred=y_pred)  #Jak coś to może ta metryka sie nada
                print(f"{list(clfs)[clf_idx]}: {results[clf_idx, fold]:.3f}")

        mean_results = np.mean(results, axis=1)

        print(f"\nAverage mean score vector: {mean_results}")

# Zapis wyników do pliku (w celu użycia później przy testach statystycznych)
    np.save("results.npy", results)

