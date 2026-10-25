import tensorflow as tf
from src.config import IMG_SIZE

def decode_resize(raw):
    """Bytes of a jpg/png -> float32 (IMG_SIZE, IMG_SIZE, 3) in [0, 255]. Shared by training and serving."""
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)   # drops alpha, greyscale -> 3 channels
    return tf.image.resize(img, (IMG_SIZE, IMG_SIZE))                    # bilinear, aspect ratio NOT kept (stretch)
