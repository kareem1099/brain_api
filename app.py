from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os
import tensorflow as tf

app = FastAPI()

model = load_model("Models/model_image_(Brain).h5")
brain_classes = ['brain_menin', 'brain_glioma', 'brain_pituitary', 'no_tumor']

def preprocess_brain_image(path):
    img = tf.keras.utils.load_img(path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict_Brain")
async def predict_brain(file: UploadFile = File(...)):
    try:
        temp_path = "temp_brain.jpg"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = preprocess_brain_image(temp_path)
        prediction = model.predict(img)
        predicted_class = brain_classes[np.argmax(prediction[0])]
        os.remove(temp_path)

        return {"result": predicted_class}
    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
