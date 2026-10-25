import os, sys
import matplotlib.pyplot as plt
import mlflow
from dotenv import load_dotenv
from tensorflow import keras

load_dotenv()
target = sys.argv[1]                       # "local" or "dagshub"
if target == "local":
    mlflow.set_tracking_uri("sqlite:///toy.db")
else:
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("toy-fashion-mnist")

(x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
x, y = x[:10000] / 255.0, y[:10000]
xt, yt = xt[:2000] / 255.0, yt[:2000]

class EpochLogger(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)

for lr in (0.01, 0.0001):
    keras.utils.set_random_seed(42)
    with mlflow.start_run(run_name=f"toy-lr-{lr}"):
        mlflow.set_tags({"stage": "toy", "target": target})
        mlflow.log_params({"lr": lr, "units": 64, "epochs": 5})
        model = keras.Sequential([
            keras.Input((28, 28)), keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax")])
        model.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(x, y, validation_data=(xt, yt), epochs=5, batch_size=64,
                      callbacks=[EpochLogger()], verbose=2)
        fig = plt.figure()
        plt.plot(h.history["loss"], label="train"); plt.plot(h.history["val_loss"], label="val")
        plt.legend(); plt.title(f"loss, lr={lr}")
        mlflow.log_figure(fig, "loss_curve.png"); plt.close(fig)
        mlflow.keras.log_model(model, "model")
print("done ->", target)