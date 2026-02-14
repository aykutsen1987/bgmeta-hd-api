import os
import io
import gc
import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import tflite_runtime.interpreter as tflite

app = FastAPI()

# Doğrudan indirilebilir Hugging Face linki
MODEL_URL = "https://huggingface.co/aykutsen1987/bgmeta-u2net/resolve/main/u2net.tflite"
MODEL_PATH = "u2net.tflite"

interpreter = None
input_details = None
output_details = None

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model downloaded.")

def load_model():
    global interpreter, input_details, output_details
    if interpreter is None:
        download_model()
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

@app.post("/remove-bg")
async def remove_bg(image: UploadFile = File(...)):
    load_model()

    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((512, 512))

    input_data = np.array(img, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    mask = (output_data * 255).astype(np.uint8)

    mask_img = Image.fromarray(mask).resize(img.size)
    img.putalpha(mask_img)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    gc.collect()
    return StreamingResponse(buf, media_type="image/png")

# main.py dosyasının en altına ekle
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
