import hashlib, io
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

ALLOWED = {"jpg", "jpeg", "png"}          # the formats the model was trained on (audit: 831 JPEG, 143 PNG)
PIL_FORMATS = {"JPEG", "PNG"}

class InvalidImage(ValueError):
    """Raised with a message that is safe to show to the user."""

def check_upload(filename, raw):
    """Validate an upload held in memory. Returns (safe_name, sha256). Nothing is written to disk."""
    name = secure_filename(filename or "") or "upload"
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED:
        raise InvalidImage(f"Unsupported file type '.{ext or '?'}'. Please upload a JPG or PNG image.")
    if not raw:
        raise InvalidImage("The file is empty.")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt = im.format
            im.load()                      # full decode, catches truncated / fake files
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise InvalidImage("This file has an image extension but could not be decoded as an image.")
    if fmt not in PIL_FORMATS:
        raise InvalidImage(f"The file content is {fmt}, not JPG or PNG.")
    return name, hashlib.sha256(raw).hexdigest()
