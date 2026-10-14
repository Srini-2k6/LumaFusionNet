from pathlib import Path
from PIL import Image,ImageEnhance,ImageOps
import numpy as np, torch
from model import LumaFusionNet
ROOT=Path(__file__).resolve().parent; WEIGHTS=ROOT/'weights'/'lumafusionnet_best.pth'
class Enhancer:
 def __init__(self):
  self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); self.model=LumaFusionNet().to(self.device).eval(); self.model_loaded=False
  if WEIGHTS.exists():
   try:
    c=torch.load(WEIGHTS,map_location=self.device); self.model.load_state_dict(c.get('model',c) if isinstance(c,dict) else c,strict=False); self.model_loaded=True
   except Exception as e: print('Checkpoint load failed:',e)
 def demo(self,img):
  img=ImageOps.exif_transpose(img).convert('RGB'); a=np.asarray(img).astype(np.float32)/255; a=np.power(np.clip(a,0,1),.62); out=Image.fromarray(np.uint8(a*255)); return ImageEnhance.Color(ImageEnhance.Contrast(out).enhance(1.08)).enhance(1.03)
 @torch.inference_mode()
 def enhance(self,img):
  if not self.model_loaded:return self.demo(img),'Demo enhancement — trained weights not found'
  img=ImageOps.exif_transpose(img).convert('RGB'); size=img.size; scale=min(1,768/max(size));
  if scale<1: img=img.resize((round(size[0]*scale),round(size[1]*scale)),Image.Resampling.LANCZOS)
  a=np.asarray(img).astype(np.float32)/255; x=torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(self.device); y=self.model(x)[0].permute(1,2,0).cpu().numpy(); out=Image.fromarray(np.uint8(np.clip(y*255,0,255))); return (out.resize(size,Image.Resampling.LANCZOS) if out.size!=size else out),'LumaFusionNet neural inference'
