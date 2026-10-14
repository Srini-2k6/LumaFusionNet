from pathlib import Path
import argparse,numpy as np,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from model import LumaFusionNet
class Pair(Dataset):
 def __init__(self,a,b,size=256):
  self.a,self.b=Path(a),Path(b); self.names=sorted(p.name for p in self.a.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'} and (self.b/p.name).exists()); self.size=size
  if not self.names: raise RuntimeError('No matching pairs found.')
 def __len__(self): return len(self.names)
 def __getitem__(self,i):
  def read(p): return torch.from_numpy(np.asarray(Image.open(p).convert('RGB').resize((self.size,self.size))).copy()).permute(2,0,1).float()/255
  return read(self.a/self.names[i]),read(self.b/self.names[i])
p=argparse.ArgumentParser();p.add_argument('--low',default='data/train/low');p.add_argument('--normal',default='data/train/normal');p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=2);p.add_argument('--lr',type=float,default=2e-4);a=p.parse_args();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ds=Pair(a.low,a.normal);dl=DataLoader(ds,a.batch,shuffle=True);m=LumaFusionNet().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=a.lr);best=1e9;Path('weights').mkdir(exist_ok=True)
for ep in range(1,a.epochs+1):
 m.train();total=0
 for x,y in dl:
  x,y=x.to(dev),y.to(dev);loss=F.l1_loss(m(x),y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();total+=loss.item()
 loss=total/len(dl);print(f'Epoch {ep:03d} | L1 {loss:.6f}')
 if loss<best: best=loss;torch.save({'model':m.state_dict(),'epoch':ep,'loss':loss},'weights/lumafusionnet_best.pth')
