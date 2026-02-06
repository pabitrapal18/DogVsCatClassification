from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load Keras 3 model
model = tf.keras.models.load_model("dog_cat_model_fixed.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            img = Image.open(file).convert("RGB")
            img = img.resize((128, 128))
            img = np.asarray(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = float(model.predict(img).squeeze())

            if pred >= 0.5:
                prediction = "Dog 🐶"
                confidence = round(pred * 100, 2)
            else:
                prediction = "Cat 🐱"
                confidence = round((1 - pred) * 100, 2)

            print("Prediction value:", pred)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
