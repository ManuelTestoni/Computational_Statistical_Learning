from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # --- Caricamento del dataset ---
    iris = datasets.load_iris()
    X, y = iris.data[:, :2], iris.target
    print(f"[1] Dataset Iris caricato: {X.shape[0]} campioni, {X.shape[1]} feature usate (sepal length, sepal width)")
    print(f"    Classi: {list(iris.target_names)}")

    # --- Split train/test ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
    print(f"\n[2] Split train/test (75%/25%):")
    print(f"    Training set: {X_train.shape[0]} campioni")
    print(f"    Test set:     {X_test.shape[0]} campioni")

    # --- Normalizzazione ---
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"\n[3] Normalizzazione con StandardScaler applicata")
    print(f"    Media (train): {scaler.mean_.round(4)}")
    print(f"    Std   (train): {scaler.scale_.round(4)}")

    # --- Addestramento KNN ---
    k = 5
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    print(f"\n[4] Addestramento KNN con k={k}...")
    knn.fit(X_train, y_train)
    print(f"    Modello addestrato.")

    # --- Predizione ---
    y_pred = knn.predict(X_test)
    print(f"\n[5] Predizioni sul test set completate.")
    print(f"    Valori reali:    {y_test}")
    print(f"    Valori predetti: {y_pred}")

    # --- Valutazione ---
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[6] Accuratezza del modello: {acc * 100:.2f}%")
    correct = (y_test == y_pred).sum()
    print(f"    Predizioni corrette: {correct}/{len(y_test)}")

if __name__ == "__main__":
    main()