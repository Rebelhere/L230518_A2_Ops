import keras_tuner as kt, pandas as pd
from tensorflow import keras
from src.config import SEED
from src.data import make_ds
from src.models import build_mobilenet, METRICS

train, val = make_ds("train", augment=True, shuffle=True), make_ds("val")

class VisionGuardHM(kt.HyperModel):
    def build(self, hp):
        return build_mobilenet(hp.Float("dropout", 0.2, 0.5, step=0.1),
                               hp.Float("lr", 1e-4, 1e-3, sampling="log"))
    def fit(self, hp, model, *args, **kwargs):
        n = hp.Choice("unfrozen_layers", [0, 10, 30, 60])
        stop = lambda: keras.callbacks.EarlyStopping("val_loss", patience=3, restore_best_weights=True)
        h = model.fit(train, validation_data=val, epochs=6, callbacks=[stop()], verbose=0)
        if n > 0:
            base = model.get_layer("mobilenetv2_1.00_224")
            base.trainable = True
            for l in base.layers[:-n]: l.trainable = False
            for l in base.layers:
                if isinstance(l, keras.layers.BatchNormalization): l.trainable = False
            model.compile(keras.optimizers.Adam(hp.get("lr") / 10), "binary_crossentropy", metrics=METRICS())
            h = model.fit(train, validation_data=val, epochs=4, callbacks=[stop()], verbose=0)
        return h

keras.utils.set_random_seed(SEED)
t = kt.RandomSearch(VisionGuardHM(), objective=kt.Objective("val_auc", "max"), max_trials=8, seed=SEED,
                    directory="tuner_logs", project_name="visionguard", overwrite=True)
t.search()
df = pd.DataFrame([{**tr.hyperparameters.values, "val_auc": round(tr.score, 4)}
                   for tr in t.oracle.get_best_trials(8)])
print(df.to_string(index=False)); df.to_csv("reports/vg_tuner.csv", index=False)
print("best settings:", t.get_best_hyperparameters(1)[0].values)
for c in ("dropout", "unfrozen_layers"):
    print(f"\nmean val_auc by {c}:\n", df.groupby(c)["val_auc"].agg(["mean", "count"]).round(4))
print("\nSpearman(lr, val_auc):", round(df["lr"].corr(df["val_auc"], method="spearman"), 3))