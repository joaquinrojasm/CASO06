from flask import Flask, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

MODEL_DIR = "model"

# --- Cargar modelos ---
encoder = load_model(os.path.join(MODEL_DIR, "encoder.h5"))
decoder = load_model(os.path.join(MODEL_DIR, "decoder.h5"))

LATENT_DIM = 128  # Debe coincidir con el modelo

# --- Preparar vectores latentes promedio por dígito ---
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1)

digit_latent_avg = {}
for digit in range(10):
    idx = np.where(y_test == digit)[0]
    x_digit = x_test[idx]
    z = encoder.predict(x_digit, verbose=0)
    digit_latent_avg[digit] = np.mean(z, axis=0, keepdims=True)

# --- Función para generar imagen de un número (multi-dígito) ---
def generar_imagen(numero_str):
    imgs = []
    for ch in numero_str:
        if ch.isdigit():
            d = int(ch) % 10
            z = digit_latent_avg[d]
            img_array = decoder.predict(z, verbose=0)[0]
            img_array = (img_array.squeeze() * 255).astype("uint8")
            pil_img = Image.fromarray(img_array, mode="L").resize((100,100))  # tamaño fijo por dígito
            imgs.append(pil_img)
    # Combinar los dígitos horizontalmente
    if not imgs:
        return None
    total_width = sum(img.width for img in imgs)
    max_height = max(img.height for img in imgs)
    combined = Image.new("L", (total_width, max_height), color=255)
    x_offset = 0
    for img in imgs:
        combined.paste(img, (x_offset,0))
        x_offset += img.width
    return combined

# --- Ruta principal ---
@app.route("/", methods=["GET", "POST"])
def index():
    img_data = None
    from flask import request
    if request.method == "POST":
        numero = request.form.get("numero")
        if numero is not None and numero.isdigit():
            pil_img = generar_imagen(numero)
            if pil_img:
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                img_data = "data:image/png;base64," + base64.b64encode(buf.read()).decode('utf-8')
    return render_template("index.html", img_data=img_data)

if __name__ == "__main__":
    app.run(debug=True)