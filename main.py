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

# Yapılandırma
MODEL_URL = "https://huggingface.co/aykutsen1987/bgmeta-u2net/resolve/main/u2net.tflite"
MODEL_PATH = "u2net.tflite"

# Global Değişkenler
interpreter = None
input_details = None
output_details = None

def download_model():
    """Modeli Hugging Face'den indirir."""
    if not os.path.exists(MODEL_PATH):
        print("Model indiriliyor...")
        try:
            r = requests.get(MODEL_URL, stream=True, timeout=60)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: f.write(chunk)
            print("Model başarıyla indirildi.")
        except Exception as e:
            print(f"İndirme hatası: {e}")
            return False
    return True

def load_model():
    """Modeli RAM'e yükler ve detayları ayarlar."""
    global interpreter, input_details, output_details
    if interpreter is None:
        if download_model():
            try:
                # num_threads=1: Render Free RAM (512MB) için hayati önem taşır
                interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=1)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                print("Model başarıyla RAM'e yüklendi.")
            except Exception as e:
                print(f"Model yükleme hatası: {e}")
                interpreter = None

# Uygulama ayağa kalkarken modeli yükle
load_model()

@app.get("/")
async def root():
    """Uygulamanın çalıştığını ve model durumunu kontrol eder."""
    return {
        "status": "online",
        "service": "BG-Meta Background Remover",
        "api_endpoint": "/remove-bg",
        "model_loaded": interpreter is not None
    }

@app.post("/remove-bg")
async def remove_bg(image: UploadFile = File(...)):
    global interpreter, input_details, output_details
    
    # Model yüklü değilse tekrar yüklemeyi dene
    if interpreter is None:
        load_model()
        if interpreter is None:
            raise HTTPException(status_code=500, detail="Model sunucuda başlatılamadı.")

    try:
        # Resmi oku
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = img.size
        
        # RAM tasarrufu için işlem boyutunu 320x320 yapalım
        img_resized = img.resize((320, 320))

        input_data = np.array(img_resized, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        # Tahmin işlemi
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Sonucu al ve maske oluştur
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        mask = (output_data * 255).astype(np.uint8)

        # Maskeyi orijinal boyuta büyüt ve orijinal resme uygula
        mask_img = Image.fromarray(mask).resize(original_size)
        img.putalpha(mask_img)

        # PNG formatında belleğe kaydet
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # [cite_start]Bellek temizliği (Render Free RAM koruması) [cite: 4]
        del input_data
        del output_data
        gc.collect()

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print(f"İşlem hatası detayı: {e}")
        raise HTTPException(status_code=500, detail=f"İşlem sırasında hata: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Render için port 10000 varsayılandır
    uvicorn.run(app, host="0.0.0.0", port=10000)
