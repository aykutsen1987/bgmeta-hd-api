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

# Küresel değişkenler (Uygulama başladığında bir kez dolacak)
interpreter = None
input_details = None
output_details = None

def download_model():
    """Model dosyasını Hugging Face'den güvenli bir şekilde indirir."""
    if not os.path.exists(MODEL_PATH):
        print("Model indiriliyor...")
        try:
            r = requests.get(MODEL_URL, stream=True, timeout=30)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Model başarıyla indirildi.")
        except Exception as e:
            print(f"İndirme hatası: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)

def load_model_globally():
    """Modeli RAM'e yükler. Render'da OOM hatasını önlemek için optimize edilmiştir."""
    global interpreter, input_details, output_details
    if interpreter is None:
        download_model()
        try:
            # Bellek verimliliği için num_threads eklenmiştir
            interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=1)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details() [cite: 2]
            output_details = interpreter.get_output_details() [cite: 3]
            print("Model başarıyla yüklendi.")
        except Exception as e:
            print(f"Model yükleme hatası: {e}")

# Uygulama ayağa kalkarken modeli yükle (İstek gelmesini bekleme)
load_model_globally()

@app.get("/")
async def health_check():
    return {"status": "online", "model_loaded": interpreter is not None}

@app.post("/remove-bg")
async def remove_bg(image: UploadFile = File(...)):
    # Güvenlik: Dosya tipi kontrolü
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Sadece resim dosyası yüklenebilir.")
    
    if interpreter is None:
        load_model_globally()
        if interpreter is None:
            raise HTTPException(status_code=500, detail="Model yüklenemedi.")

    try:
        # Resmi oku
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = img.size

        # Render RAM koruması: İşleme boyutunu 320x320'ye düşürelim (Hız ve stabilite sağlar)
        process_size = (320, 320)
        img_resized = img.resize(process_size)

        # Giriş verisini hazırla
        input_data = np.array(img_resized, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        # Tahmin (Inference)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Çıkış verisini al ve maske oluştur
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        mask = (output_data * 255).astype(np.uint8)

        # Maskeyi orijinal boyuta geri getir ve resme uygula
        mask_img = Image.fromarray(mask).resize(original_size)
        img.putalpha(mask_img)

        # Sonucu belleğe yaz
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0) [cite: 4]

        # Bellek temizliği
        del input_data
        del output_data
        gc.collect() [cite: 4]

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print(f"İşlem hatası: {e}")
        raise HTTPException(status_code=500, detail="Resim işlenirken bir hata oluştu.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
