from ultralytics.nn.modules import Conv
import torch 
import subprocess
import shutil


x = torch.rand(1, 3, 640, 640)
conv1 = Conv(3,64, 3, 2)
conv2 = Conv(64, 128, 3, 2)

model = torch.nn.Sequential(conv1, conv2)

model.eval()

print(model)
traced_model = torch.jit.trace(model,x)

f = f"{model._get_name()}.onnx"
torch.onnx.export(model,x,f)
subprocess.run(f"netron {f}", shell=True)

