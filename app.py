
from pathlib import Path
import base64
import io
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from enhancer import Enhancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lumafusionnet")

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_IMAGE_SIDE = 1600

app = FastAPI(title="LumaFusionNet")
app.mount(
    "/static",
    StaticFiles(directory=ROOT / "static"),
    name="static",
)

enhancer = Enhancer()


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    return {
        "model_loaded": enhancer.model_loaded,
        "device": str(enhancer.device),
        "mode": "neural" if enhancer.model_loaded else "demo",
    }


@app.post("/api/enhance")
async def enhance(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Choose an image first.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Please upload a JPG, PNG, or other supported image.",
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 12 MB upload limit.",
        )

    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large. Try a smaller image.",
                )

            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")

        # Limit processing memory and response size.
        image.thumbnail(
            (MAX_IMAGE_SIDE, MAX_IMAGE_SIDE),
            Image.Resampling.LANCZOS,
        )

        result, mode = enhancer.enhance(image)

        buffer = io.BytesIO()
        result.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        logger.info(
            "Image processed successfully; mode=%s size=%sx%s",
            mode,
            result.width,
            result.height,
        )

        return JSONResponse({
            "success": True,
            "mode": mode,
            "image": "data:image/png;base64," + encoded,
        })

    except HTTPException:
        raise

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="This file is not a supported image. Try a JPG or PNG.",
        )

    except Exception:
        logger.exception("Image enhancement failed")
        raise HTTPException(
            status_code=500,
            detail="Image processing failed. Please try a smaller JPG or PNG.",
        )
