from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import shutil
import os

app = FastAPI()

model = load_model("models/model_image_(Brain).h5")
brain_classes = ['brain_menin', 'brain_glioma', 'brain_pituitary', 'no_tumor']

def preprocess_brain_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128)) / 255.0
    return np.expand_dims(img, axis=0)

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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
