import numpy as np
import pandas as pd
#from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from import_penguins_dataset import import_penguins_dataset
from import_mushrooms_dataset import import_mushrooms_dataset
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

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
            weak_learn = DecisionTreeClassifier(max_depth=1, random_state=random_state)
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

    random_state = 2 #mozna sobie ustawiac

#zaimportowanie pingwinów i grzybów
    X_penguins, y_penguins = import_penguins_dataset()
#nrows w grzybach służy do ograniczenia ilość obserwacji, dane są tam przetasowane, a obserwacji jest w sumie ok. 61000
    X_mushrooms, y_mushrooms = import_mushrooms_dataset(nrows=300)



# Dane syntetyczne
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=random_state)


# Słownik datasetów
    d_set = {'Synthetic Data (make_classification)': (X, y),
             'Penguins data set': (X_penguins, y_penguins),
             'Mushrooms data set': (X_mushrooms, y_mushrooms),
             }

# Słownik klasyfikatorów
    clfs = {
        'AdaBoost_50': Adaboost(T=50),
        'AdaBoost_100': Adaboost(T=100),
        'Random_Forest': RandomForestClassifier(max_depth=2, random_state=random_state),
        'Gradient_Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=1, random_state=random_state),
        'Bagging': BaggingClassifier(KNeighborsClassifier(n_neighbors=3),n_estimators=100)
    }

# Walidacja krzyżowa
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=random_state)

#Zadeklarowanie macierzy wyników
    results = np.zeros(shape=[len(d_set), len(clfs), rskf.get_n_splits()])

# Cała magia
    for dset_id, dset in enumerate(d_set):
        X, y = d_set[dset]
        print("\nDataset: ", dset)

        for fold, (train, test) in enumerate(rskf.split(X, y)):
            print(f"\nFold {fold}:")

            for clf_idx, clfn in enumerate(clfs):
                clf = clone(clfs[clfn])

                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])

                results[dset_id, clf_idx, fold] = accuracy_score(y_true=y[test], y_pred=y_pred)
                # results[dset_id, clf_idx, fold] = ballanced_accuracy_score(y_true=y[test], y_pred=y_pred)  #Jak coś to może ta metryka sie nada
                print(f"{list(clfs)[clf_idx]}: {results[dset_id, clf_idx, fold]:.3f}")

        mean_results = np.mean(results[dset_id], axis=1)
        std_results = np.std(results[dset_id], axis=1)

        # print(f"\nAverage mean score vector: {mean_results}") # Wyświetlane poźniej razem z std

# Zapis wyników do pliku (w celu użycia później przy testach statystycznych)
    np.save("results.npy", results)

    ''' Prezentowane w tabeli na sam koniec
    # Prezentacja jakości
    for clf_name, mean_score, std in zip(clfs.keys(), mean_results, std_results):
        print(f"{clf_name}: Średnia={mean_score:.3f}, Odchylenie standardowe={std:.3f}")
    '''

    # Przeprowadzenie testów statystycznych
    for dset_id, dset in enumerate(d_set):
        X, y = d_set[dset]

        mean_results = np.mean(results[dset_id], axis=1)
        std_results = np.std(results[dset_id], axis=1)

        best_clf_idx = np.argmax(mean_results)
        best_clf_name = list(clfs.keys())[best_clf_idx]

        print(f"\nNajlepszy klasyfikator dla {dset}: {best_clf_name} z wynikiem: {mean_results[best_clf_idx]:.3f}\n")

        for clf_name, clf_scores in zip(clfs.keys(), results[dset_id]):
            if clf_name != best_clf_name:
                t_statistic, p_value = ttest_rel(results[dset_id, best_clf_idx], clf_scores)
                if p_value < 0.05:
                    print(
                        f"Istnieje istotna statystycznie różnica między {clf_name} a najlepszym klasyfikatorem dla {dset}.")
                else:
                    print(
                        f"Brak istotnej statystycznie różnicy między {clf_name} a najlepszym klasyfikatorem dla {dset}.")

    # Prezentacja wyników w tabeli
    results_df = pd.DataFrame({
        'Klasyfikator': list(clfs.keys()),
        'Średnia wyniku': mean_results,
        'Odchylenie standardowe': std_results
    })
    print("\nTabela wyników:")
    print(results_df)

    # Rysowanie wykresów
    for i, dataset in enumerate(d_set.keys()):
        means = np.mean(results[i], axis=1)
        stds = np.std(results[i], axis=1)

        plt.figure(figsize=(10, 6))
        plt.bar(clfs.keys(), means, yerr=stds, capsize=5, color=['blue', 'green', 'red', 'cyan'])
        plt.xlabel('Classifiers')
        plt.ylabel('Mean Accuracy')
        plt.title(f'Performance on {dataset}')
        plt.ylim(0, 1)
        plt.show()
