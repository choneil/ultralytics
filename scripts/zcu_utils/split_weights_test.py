#!/usr/bin/env python3
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO

def split_weights(model_name):
    # Derive paths from model name (e.g., "yolov8n")
    # Family is everything except the size suffix (n/s/m/l/x)
    model_family = model_name[:-1]  # e.g., "yolov8n" -> "yolov8"
    
    weights = f"{model_name}.pt"
    model_cfg = f"{model_name}.yaml"
    
    # Output directories
    base_dir = Path(f"weights/{model_family}")
    pretrained_dir = base_dir / "pretrained"
    split_dir = base_dir / "split"
    
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    
    pretrained_output = pretrained_dir / f"{model_name}.pt"
    split_output = split_dir / f"{model_name}_split.pt"

    print(f"Loading source weights from: {weights}")
    pretrained = YOLO(weights)
    pretrained_sd = pretrained.model.state_dict()
    
    # Save original pretrained weights
    pretrained.save(pretrained_output)
    print(f"Saved original pretrained weights to: {pretrained_output.absolute()}")

    print(f"Building target architecture from: {model_cfg}")
    target_model = YOLO(model_cfg)
    target_sd = target_model.model.state_dict()

    new_sd = {}
    converted_count = 0

    print("Starting weight conversion...")
    
    for key, val in pretrained_sd.items():
        if '.cv1.' in key:
            base, suffix = key.rsplit('.cv1.', 1)
            cv1a_key = f"{base}.cv1a.{suffix}"
            cv1b_key = f"{base}.cv1b.{suffix}"

            if cv1a_key in target_sd and cv1b_key in target_sd:
                if val.dim() == 0:
                    new_sd[cv1a_key] = val.clone()
                    new_sd[cv1b_key] = val.clone()
                else:
                    if val.shape[0] % 2 != 0:
                        print(f"⚠️ Warning: {key} has odd channels {val.shape[0]}, cannot split evenly!")
                        continue
                        
                    half = val.shape[0] // 2
                    new_sd[cv1a_key] = val[:half].clone()
                    new_sd[cv1b_key] = val[half:].clone()
                
                converted_count += 1
            else:
                new_sd[key] = val
        else:
            new_sd[key] = val

    if converted_count == 0:
        print("\n❌ Error: No layers were converted. Did you update block.py to define cv1a/cv1b?")
        return

    print(f"Successfully split {converted_count} tensors.")

    missing, unexpected = target_model.model.load_state_dict(new_sd, strict=False)
    
    if len(missing) > 0:
        print(f"⚠️ Warning: Missing keys: {missing}")
    if len(unexpected) > 0:
        print(f"⚠️ Warning: Unexpected keys in source: {unexpected}")

    target_model.save(split_output)
    print(f"\n✅ Saved DPU-ready model to: {split_output.absolute()}")

def parse_opt():
    parser = argparse.ArgumentParser(description='Split YOLO weights for DPU compatibility (cv1 -> cv1a/cv1b)')
    parser.add_argument('model', type=str, help='Model name (e.g., yolov8n, yolo11n)')
    return parser.parse_args()

def main(opt):
    split_weights(opt.model)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
