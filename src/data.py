import pandas as pd, tensorflow as tf
from tensorflow import keras
from src.config import BATCH, SEED, SPLITS
from src.preprocess import decode_resize

AUG = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.08),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomContrast(0.15),
], name="augmentation")

def _load(path, label):
    return decode_resize(tf.io.read_file(path)), tf.cast(label, tf.float32)

def make_ds(split, augment=False, shuffle=False):
    df = pd.read_csv(SPLITS[split])
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).cache()   # decode+resize once
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    if augment:                                                         # training data only
        ds = ds.map(lambda x, y: (AUG(tf.expand_dims(x, 0), training=True)[0], y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)