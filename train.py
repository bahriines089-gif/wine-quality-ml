import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Reproductibilité (Seed)
# C'est important pour que le jury obtienne le même résultat que toi
np.random.seed(42)

# 2. Chargement des données
data = load_wine()
X = data.data
y = data.target

# 3. Séparation Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Normalisation (La correction Pro !)
# On met toutes les données à la même échelle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entraînement du modèle
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Prédiction
y_pred = model.predict(X_test_scaled)

# 7. Évaluation
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(cm)