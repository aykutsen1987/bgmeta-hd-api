import os
import io
import gc
import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import tflite_runtime.interpreter as tflite

app = FastAPI()

MODEL_URL = "https://huggingface.co/aykutsen1987/bgmeta-u2net/resolve/main/u2net.tflite"
MODEL_PATH = "u2net.tflite"

interpreter = None
input_details = None
output_details = None

def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            r = requests.get(MODEL_URL, stream=True, timeout=60)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
            print("Model indirildi.")
        except Exception as e:
            print(f"Indirme hatasi: {e}")
            return False
    return True

def load_model_globally():
    global interpreter, input_details, output_details
    if interpreter is None:
        if download_model():
            try:
                # num_threads=1 Render RAM'i icin zorunludur
                interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=1)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print("Model RAM'e yuklendi.")
            except Exception as e:
                print(f"Model yukleme hatasi: {e}")

# Uygulama acilista modeli bir kez yukler
load_model_globally()

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"status": "alive", "model": "loaded" if interpreter else "failed"}

@app.post("/remove-bg")
async def remove_bg(image: UploadFile = File(...)):
    global interpreter, input_details, output_details
    
    if interpreter is None:
        load_model_globally()
        if interpreter is None:
            raise HTTPException(status_code=500, detail="Model hazir degil.")

    try:
        # 1. Resmi oku ve RAM tasarrufu icin hemen kucult
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = img.size
        
        # OOM hatasini onlemek icin isleme boyutunu 256x256'ya dusurduk
        img_resized = img.resize((256, 256), Image.LANCZOS)

        # 2. Veriyi hazirla
        input_data = np.array(img_resized, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        # 3. Tahmin (Inference)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 4. Maskeyi al ve temizle
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        mask = (output_data * 255).astype(np.uint8)
        
        # Bellek temizligi (Tahmin sonrasi ilk temizlik)
        del input_data
        gc.collect()

        # 5. Maskeyi orijinal boyuta getir ve uygula
        mask_img = Image.fromarray(mask).resize(original_size, Image.LANCZOS)
        img.putalpha(mask_img)

        # 6. Sonucu gonder
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        buf.seek(0)

        # Agresif temizlik
        del contents, img_resized, mask, mask_img, output_data
        gc.collect()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
