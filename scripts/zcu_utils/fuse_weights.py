from ultralytics import YOLO
import torch
import argparse
import os

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
    os.makedirs(os.path.dirname(output_weights), exist_ok=True)
    print(f"Saving fused model to: {output_weights}")
    torch.save(fused_model.state_dict(), output_weights)
    
    print("✅ Done! You can now load this .pt directly into vai_q_pytorch.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse BatchNorm into Conv layers")
    parser.add_argument("input", help="Path to unfused .pt weights")
    args = parser.parse_args()
    
    input_dir = os.path.dirname(args.input)
    parent_dir = os.path.dirname(input_dir)
    filename = os.path.basename(args.input).replace("unfused", "fused")
    output = os.path.join(parent_dir, "fused", filename)
    
    export_fused(args.input, output)
