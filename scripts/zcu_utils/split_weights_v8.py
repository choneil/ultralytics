#!/usr/bin/env python3
import torch
import argparse
from pathlib import Path
from copy import deepcopy
from ultralytics import YOLO

def split_weights(weights, model_cfg, output):
    print(f"Loading source weights from: {weights}")
    # Load the standard pretrained model (e.g., yolo11n.pt)
    # Note: This loads the serialized object. Even if your block.py is modified, 
    # the keys in this object still correspond to the old 'cv1' structure.
    pretrained = YOLO(weights)
    pretrained_sd = pretrained.model.state_dict()

    print(f"Building target architecture from: {model_cfg}")
    # Build the new model structure based on your modified block.py
    # This will generate a model with 'cv1a' and 'cv1b' layers initialized randomly
    target_model = YOLO(model_cfg)
    target_sd = target_model.model.state_dict()

    new_sd = {}
    converted_count = 0

    print("Starting weight conversion...")
    
    # Iterate through the Standard (Source) keys
    for key, val in pretrained_sd.items():
        # Check if this key belongs to a C2f 'cv1' layer
        if '.cv1.' in key:
            # Construct the corresponding keys for the Split architecture
            # Example: model.2.cv1.conv.weight -> model.2.cv1a.conv.weight
            base, suffix = key.rsplit('.cv1.', 1)
            cv1a_key = f"{base}.cv1a.{suffix}"
            cv1b_key = f"{base}.cv1b.{suffix}"

            # Check if these target keys actually exist in our new architecture
            if cv1a_key in target_sd and cv1b_key in target_sd:
                # Case 1: 0-dim scalars (e.g., num_batches_tracked) -> Copy to both
                if val.dim() == 0:
                    new_sd[cv1a_key] = val.clone()
                    new_sd[cv1b_key] = val.clone()
                
                # Case 2: Tensors (Weights, Bias, BN Stats) -> Split in half
                else:
                    # Validate shape
                    if val.shape[0] % 2 != 0:
                        print(f"⚠️ Warning: {key} has odd channels {val.shape[0]}, cannot split evenly!")
                        continue
                        
                    half = val.shape[0] // 2
                    new_sd[cv1a_key] = val[:half].clone()
                    new_sd[cv1b_key] = val[half:].clone()
                
                converted_count += 1
            else:
                # If we found a .cv1. key but the target doesn't have a/b (maybe not a C2f?), keep original
                new_sd[key] = val
        else:
            # Standard copy for all non-C2f layers
            new_sd[key] = val

    # Verify we actually did something
    if converted_count == 0:
        print("\n❌ Error: No layers were converted. Did you update block.py to define cv1a/cv1b?")
        return

    print(f"Successfully split {converted_count} tensors.")

    # Load the new dictionary into the target model
    # strict=False allows us to ignore minor mismatches if any (though usually strict=True is better)
    missing, unexpected = target_model.model.load_state_dict(new_sd, strict=False)
    
    if len(missing) > 0:
        print(f"⚠️ Warning: Missing keys: {missing}")
    if len(unexpected) > 0:
        print(f"⚠️ Warning: Unexpected keys in source: {unexpected}")

    # Save
    save_path = Path(output)
    target_model.save(save_path)
    print(f"\n✅ Saved DPU-ready model to: {save_path.absolute()}")

def parse_opt():
    parser = argparse.ArgumentParser(description='Split YOLO weights for DPU compatibility (cv1 -> cv1a/cv1b)')
    parser.add_argument('-w', '--weights', type=str, default='yolo11n.pt', help='Path to standard pretrained weights (e.g., yolo11n.pt)')
    parser.add_argument('-m', '--model', type=str, default='yolo11n.yaml', help='Path to model YAML config (e.g., yolo11n.yaml)')
    parser.add_argument('-o', '--output', type=str, default='yolo11n_split.pt', help='Path to save the split checkpoint')
    return parser.parse_args()

def main(opt):
    split_weights(opt.weights, opt.model, opt.output)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
