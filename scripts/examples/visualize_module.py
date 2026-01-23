from ultralytics.nn.modules import ConvTranspose
import torch 
import subprocess

x = torch.ones(1, 128, 40, 40)
m = ConvTranspose(128, 128)

m.eval()
traced_m = torch.jit.trace(m,x)

f = f"{m._get_name()}.onnx"
torch.onnx.export(m,x,f)
subprocess.run(f"onnxslim {f} {f} && open {f}", shell=True, check=True)
