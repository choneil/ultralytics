from ultralytics import YOLO
import torch

def export_fused(input_weights, output_weights):
    print(f"Loading un-fused model: {input_weights}")
    model = YOLO(input_weights)
    
    # 1. Fuse Batch Norm into Convolution
    # Since cv1a and cv1b are standard 'Conv' objects, they fuse automatically!
    print("Fusing Batch Normalization...")
    model.fuse()
    
    # 2. Extract the underlying PyTorch model
    fused_model = model.model
    
    # 3. Save as a standard PyTorch checkpoint
    # This file will have NO 'bn' layers; they are baked into the 'conv' weights.
    print(f"Saving fused model to: {output_weights}")
    torch.save(fused_model.state_dict(), output_weights)
    
    print("✅ Done! You can now load this .pt directly into vai_q_pytorch.")

if __name__ == "__main__":
    # Update these paths to match your training result
    INPUT = "./weights/yolo11/tuned/unfused/11nunfused.pt"
    OUTPUT = "./weights/yolo11/tuned/fused/yolo11n_zcu102.pt"
    
    export_fused(INPUT, OUTPUT)
