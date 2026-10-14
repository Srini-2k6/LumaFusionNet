from pathlib import Path
import base64,io
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from enhancer import Enhancer
ROOT=Path(__file__).resolve().parent; app=FastAPI(title='LumaFusionNet'); app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static'); enhancer=Enhancer()
@app.get('/')
def home(): return FileResponse(ROOT/'static'/'index.html')
@app.get('/api/status')
def status(): return {'model_loaded':enhancer.model_loaded,'device':str(enhancer.device),'mode':'neural' if enhancer.model_loaded else 'demo'}
@app.post('/api/enhance')
async def enhance(file:UploadFile=File(...)):
 if not file.content_type or not file.content_type.startswith('image/'): raise HTTPException(400,'Please upload an image file.')
 data=await file.read()
 if len(data)>12*1024*1024: raise HTTPException(413,'Image is larger than 12 MB.')
 try:
  result,mode=enhancer.enhance(Image.open(io.BytesIO(data))); b=io.BytesIO(); result.save(b,'PNG',optimize=True); enc=base64.b64encode(b.getvalue()).decode(); return JSONResponse({'success':True,'mode':mode,'image':'data:image/png;base64,'+enc})
 except Exception as e: raise HTTPException(500,f'Processing failed: {e}')
