import cuml
import cudf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Générer des données de classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Convertir les données en format GPU (cudf DataFrame)
X_cudf = cudf.DataFrame.from_records(X)
y_cudf = cudf.Series(y)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_cudf, y_cudf, test_size=0.2, random_state=42)

# Créer et entraîner le modèle SVM avec cuML
svm_model = cuml.svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred = svm_model.predict(X_test)

# Évaluer la performance
accuracy = accuracy_score(y_test.to_array(), y_pred.to_array())

print(f"Accuracy of the SVM model on the test data: {accuracy * 100:.2f}%")
