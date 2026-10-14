import torch
import torch.nn as nn
import torch.nn.functional as F

class SFB(nn.Module):
    def __init__(self,c):
        super().__init__(); self.s=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,c,3,padding=1)); self.a=nn.Sequential(nn.Conv2d(c,c,1),nn.GELU(),nn.Conv2d(c,c,1)); self.p=nn.Sequential(nn.Conv2d(c,c,1),nn.GELU(),nn.Conv2d(c,c,1)); self.fo=nn.Conv2d(c,c,3,padding=1); self.f=nn.Sequential(nn.Conv2d(2*c,c,1),nn.GELU(),nn.Conv2d(c,c,3,padding=1)); self.n=nn.GroupNorm(8 if c>=8 else 1,c)
    def forward(self,x):
        s=self.s(x); z=torch.fft.fft2(x,dim=(-2,-1),norm='ortho'); a=F.softplus(self.a(torch.abs(z)))+1e-6; p=self.p(torch.angle(z)); q=torch.polar(a,p); fr=self.fo(torch.fft.ifft2(q,dim=(-2,-1),norm='ortho').real); return self.n(x+self.f(torch.cat([s,fr],1)))

class CA(nn.Module):
    def __init__(self,c,r=8):
        super().__init__(); h=max(c//r,4); self.net=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,h,1),nn.ReLU(True),nn.Conv2d(h,c,1),nn.Sigmoid())
    def forward(self,x): return x*self.net(x)

class Stack(nn.Module):
    def __init__(self,c,n=2): super().__init__(); self.b=nn.Sequential(*[SFB(c) for _ in range(n)])
    def forward(self,x): return self.b(x)

class LumaFusionNet(nn.Module):
    def __init__(self):
        super().__init__(); self.i=nn.Conv2d(3,32,3,padding=1); self.e0=Stack(32); self.d0=nn.Conv2d(32,64,3,2,1); self.e1=Stack(64); self.d1=nn.Conv2d(64,128,3,2,1); self.e2=Stack(128); self.d2=nn.Conv2d(128,256,3,2,1); self.b=nn.Sequential(SFB(256),SFB(256),CA(256),SFB(256)); self.u2=nn.ConvTranspose2d(256,128,2,2); self.x2=Stack(128); self.u1=nn.ConvTranspose2d(128,64,2,2); self.x1=Stack(64); self.u0=nn.ConvTranspose2d(64,32,2,2); self.x0=Stack(32); self.o=nn.Conv2d(32,3,3,padding=1)
    def forward(self,x):
        e0=self.e0(self.i(x)); e1=self.e1(self.d0(e0)); e2=self.e2(self.d1(e1)); b=self.b(self.d2(e2)); y=self.u2(b); y=F.interpolate(y,size=e2.shape[-2:],mode='bilinear',align_corners=False); y=self.x2(y+e2); y=self.u1(y); y=F.interpolate(y,size=e1.shape[-2:],mode='bilinear',align_corners=False); y=self.x1(y+e1); y=self.u0(y); y=F.interpolate(y,size=e0.shape[-2:],mode='bilinear',align_corners=False); y=self.x0(y+e0); return torch.clamp(x+self.o(y),0,1)
