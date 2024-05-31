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
from import_penguins_dataset import import_penguins_dataset
from import_mushrooms_dataset import import_mushrooms_dataset

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

#zaimportowanie pingwinów i grzybów
    X_penguins, y_penguins = import_penguins_dataset()
#nrows w grzybach służy do ograniczenia ilość obserwacji, dane są tam przetasowane, a obserwacji jest w sumie ok. 61000
    X_mushrooms, y_mushrooms = import_mushrooms_dataset(nrows=300)



# Dane syntetyczne
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)


# Słownik datasetów
    d_set = {'Synthetic Data (make_classification)': (X, y),
             'Penguins data set': (X_penguins, y_penguins),
             'Mushrooms data set': (X_mushrooms, y_mushrooms),
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

