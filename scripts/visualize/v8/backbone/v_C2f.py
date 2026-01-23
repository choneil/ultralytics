from ultralytics.nn.modules import C2f
import torch 
import subprocess
import shutil


x = torch.ones(1, 128, 20, 20)
#input [B, C, H, W]
m = C2f(c1=128, c2=128, n=1, shortcut=True)
m.eval()

print(m)
traced_m = torch.jit.trace(m,x)

f = f"{m._get_name()}.onnx"
torch.onnx.export(m,x,f)
subprocess.run(f"netron {f}", shell=True)

