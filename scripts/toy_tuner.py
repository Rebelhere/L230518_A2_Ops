import keras_tuner as kt, pandas as pd
from tensorflow import keras

(x, y), (xv, yv) = keras.datasets.fashion_mnist.load_data()
x, y = x[:10000] / 255.0, y[:10000]
xv, yv = xv[:2000] / 255.0, yv[:2000]

def build(hp):
    m = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                          keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
                          keras.layers.Dense(10, activation="softmax")])
    m.compile(keras.optimizers.Adam(hp.Float("lr", 1e-4, 1e-2, sampling="log")),
              "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

keras.utils.set_random_seed(42)
t = kt.RandomSearch(build, objective="val_accuracy", max_trials=8, seed=42,
                    directory="tuner_logs", project_name="toy_fashion", overwrite=True)
t.search(x, y, validation_data=(xv, yv), epochs=5, verbose=2)
df = pd.DataFrame([{**tr.hyperparameters.values, "val_accuracy": round(tr.score, 4)}
                   for tr in t.oracle.get_best_trials(8)])
print(df.to_string(index=False))
df.to_csv("reports/toy_tuner.csv", index=False)
print("best settings:", t.get_best_hyperparameters(1)[0].values)