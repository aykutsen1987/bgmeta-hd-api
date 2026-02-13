import io
import gc
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
import tflite_runtime.interpreter as tflite

app = FastAPI()

interpreter = None
input_details = None
output_details = None


def load_model():
    global interpreter, input_details, output_details
    if interpreter is None:
        interpreter = tflite.Interpreter(model_path="u2net.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()


@app.post("/remove-bg")
async def remove_bg(image: UploadFile = File(...)):

    load_model()

    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # RAM için resize limit
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
