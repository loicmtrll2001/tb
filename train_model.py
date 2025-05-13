import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

DATASET_PATH = "../models/dataset_sequences.csv"
MODEL_PATH = "../models/model_gestes.h5"
ENCODER_PATH = "../models/label_encoder.pkl"

print("Chargement du dataset")
df = pd.read_csv(DATASET_PATH)

# Nettoyage : conversion en float et suppression des lignes corrompues
df = df.dropna()
X = df.drop("label", axis=1).astype(np.float32).values
y = df["label"].values

print(f"Dimensions d'entrée détectées : {X.shape[1]} features par séquence")

print("Préparation des données")
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
joblib.dump(encoder, ENCODER_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

print("Création du modèle")
model = Sequential()
model.add(Dense(256, activation="relu", input_shape=(X.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(len(np.unique(y_encoded)), activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Entraînement en cours")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)

print("Sauvegarde du modèle")
model.save(MODEL_PATH)

print("Modèle entraîné et sauvegardé avec succès !")
