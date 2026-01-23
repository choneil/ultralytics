from ultralytics.nn.modules import Conv
import torch
import subprocess
import shutil

# 1. Setup Input and Model
x = torch.rand(1, 3, 640, 640)
m = Conv(3, 64, 3, 2) # c1=3, c2=64, k=3, s=2

# 2. Set to Training Mode (so layers don't merge)
m.eval()

print(m)

# 3. Trace the model (Fixes the naming bug)
traced_m = torch.jit.trace(m, x)

f = f"{m._get_name()}.onnx"

# 4. Export the TRACED model
torch.onnx.export(
    traced_m,   # <--- IMPORTANT: Pass traced_m, not m
    x, 
    f, 
    do_constant_folding=False, 
    training=torch.onnx.TrainingMode.TRAINING,
    opset_version=12
)

# 5. Open Netron
if shutil.which("netron"):
    subprocess.run(f"netron {f}", shell=True)
else:
    print(f"Saved to {f}. Upload to netron.app to view.")
