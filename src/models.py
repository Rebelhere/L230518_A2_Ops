from tensorflow import keras
from tensorflow.keras import layers
from src.config import IMG_SIZE

METRICS = lambda: [keras.metrics.BinaryAccuracy(name="accuracy"),
                   keras.metrics.Precision(name="precision"),
                   keras.metrics.Recall(name="recall"),
                   keras.metrics.AUC(name="auc")]

def build_cnn(dropout=0.4, lr=1e-3):
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Rescaling(1 / 255.0)(inp)
    for f in (32, 64, 128, 128):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out, name="baseline_cnn")
    m.compile(keras.optimizers.Adam(lr), "binary_crossentropy", metrics=METRICS())
    return m

def build_mobilenet(dropout=0.3, lr=1e-3):
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inp)   # scales to [-1, 1]
    x = base(x, training=False)                                  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    m = keras.Model(inp, out, name="mobilenetv2_transfer")
    m.compile(keras.optimizers.Adam(lr), "binary_crossentropy", metrics=METRICS())
    return m

def unfreeze_last_block(model, lr=1e-5):
    base = model.get_layer("mobilenetv2_1.00_224")
    base.trainable = True
    for l in base.layers:
        l.trainable = l.name.startswith("block_16") or l.name == "Conv_1" or l.name == "Conv_1_bn"
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False                                   # keep BN statistics frozen
    model.compile(keras.optimizers.Adam(lr), "binary_crossentropy", metrics=METRICS())
    return model