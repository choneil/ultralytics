from ultralytics.nn.modules import Conv
import torch 
import subprocess
import shutil


x = torch.rand(1, 3, 640, 640)
m = Conv(3,64, 3, 2)
m.eval()

print(m)
traced_m = torch.jit.trace(m,x)

f = f"{m._get_name()}.onnx"
torch.onnx.export(m,x,f)
subprocess.run(f"netron {f}", shell=True)

