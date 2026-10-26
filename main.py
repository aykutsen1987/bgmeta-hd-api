import gc
import io
import os
import threading

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import tflite_runtime.interpreter as tflite
except ImportError:  # yerel test için
    from ai_edge_litert import interpreter as tflite

MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://huggingface.co/aykutsen1987/bgmeta-u2net/resolve/main/u2net.tflite",
)
MODEL_PATH = os.getenv("MODEL_PATH", "u2net.tflite")
MIN_MODEL_BYTES = 1_000_000          # bundan küçük dosya bozuk indirmedir
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_SIDE = 2048                      # bundan büyük görüntüler küçültülür (RAM ve süre)

app = FastAPI()

_interpreter = None
_in = None
_out = None
_load_lock = threading.Lock()
_run_lock = threading.Lock()         # TFLite yorumlayıcısı thread-safe değildir
_state = "idle"                      # idle | loading | ready | failed


def _download_model() -> bool:
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= MIN_MODEL_BYTES:
        return True
    tmp = MODEL_PATH + ".part"
    try:
        with requests.get(MODEL_URL, stream=True, timeout=(15, 120)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        if os.path.getsize(tmp) < MIN_MODEL_BYTES:
            raise IOError("indirilen model çok küçük")
        os.replace(tmp, MODEL_PATH)   # yarım dosya asla MODEL_PATH olmaz
        print("Model indirildi.")
        return True
    except Exception as e:
        print(f"Indirme hatasi: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def load_model() -> bool:
    global _interpreter, _in, _out, _state
    with _load_lock:
        if _interpreter is not None:
            return True
        _state = "loading"
        if not _download_model():
            _state = "failed"
            return False
        try:
            # num_threads=1: Render ücretsiz katmanı RAM/CPU için
            it = tflite.Interpreter(model_path=MODEL_PATH, num_threads=1)
            it.allocate_tensors()
            _in = it.get_input_details()[0]
            _out = it.get_output_details()[0]
            _interpreter = it
            _state = "ready"
            print(f"Model hazir. giris={_in['shape'].tolist()} cikis={_out['shape'].tolist()}")
            return True
        except Exception as e:
            print(f"Model yukleme hatasi: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)  # bozuksa sonraki istekte yeniden indirilsin
            _state = "failed"
            return False


@app.on_event("startup")
def _warmup():
    # Port hemen açılsın (Render dağıtım denetimi); model arka planda yüklenir.
    threading.Thread(target=load_model, daemon=True).start()


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "alive", "model": _state}


def _predict(img: Image.Image) -> np.ndarray:
    """0..1 aralığında, modelin çıkış boyutunda maske döndürür."""
    shape = [int(x) for x in _in["shape"]]
    nchw = len(shape) == 4 and shape[1] == 3
    h, w = (shape[2], shape[3]) if nchw else (shape[1], shape[2])

    arr = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    if nchw:
        arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)
    if _in["dtype"] == np.uint8:
        arr = (arr * 255.0).astype(np.uint8)

    with _run_lock:
        _interpreter.set_tensor(_in["index"], arr)
        _interpreter.invoke()
        out = _interpreter.get_tensor(_out["index"])

    m = np.squeeze(out).astype(np.float32)
    if m.ndim == 3:                       # [C,H,W] veya [H,W,C] → ilk kanal
        m = m[0] if m.shape[0] <= 4 else m[..., 0]

    lo, hi = float(m.min()), float(m.max())
    m = (m - lo) / max(hi - lo, 1e-6)     # min-max normalizasyon
    t = np.clip((m - 0.25) / 0.5, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)        # yumuşak kenar


@app.post("/remove-bg")
def remove_bg(image: UploadFile = File(...)):
    if _interpreter is None and not load_model():
        raise HTTPException(status_code=503, detail="Model hazir degil, tekrar deneyin.")

    contents = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Dosya cok buyuk (en fazla 15 MB).")

    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
        del contents
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

        mask = _predict(img)
        mask_img = Image.fromarray((mask * 255.0 + 0.5).astype(np.uint8)).resize(img.size, Image.BICUBIC)
        img.putalpha(mask_img)

        buf = io.BytesIO()
        img.save(buf, format="PNG", compress_level=3)   # optimize=True ücretsiz CPU'da çok yavaş
        return Response(content=buf.getvalue(), media_type="image/png")
    except HTTPException:
        raise
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Gecersiz goruntu dosyasi.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        gc.collect()
