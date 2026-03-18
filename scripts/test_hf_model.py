#!/usr/bin/env python3
"""
Quick inline test of HuggingFace SmolVLA LIBERO checkpoint.
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, "external/lerobot/src")

print("Attempting to import and load HuggingFace checkpoint...")
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import torch
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Try to load from HuggingFace
    print("\n1. Trying: HuggingFaceVLA/smolvla_libero")
    try:
        model = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero")
        print("✅ LOADED: HuggingFaceVLA/smolvla_libero")
    except Exception as e:
        print(f"❌ Failed: {e}")
        
        print("\n2. Trying: lerobot/smolvla_libero")
        try:
            model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_libero")
            print("✅ LOADED: lerobot/smolvla_libero")
        except Exception as e:
            print(f"❌ Failed: {e}")
            
            print("\n3. Trying: tinyrobotics/smolvla_libero")
            try:
                model = SmolVLAPolicy.from_pretrained("tinyrobotics/smolvla_libero")
                print("✅ LOADED: tinyrobotics/smolvla_libero")
            except Exception as e:
                print(f"❌ Failed: {e}")
                print("\n" + "="*80)
                print("No HuggingFace LIBERO checkpoint found under tested names.")
                print("="*80)
                sys.exit(1)
    
    # If we got here, we have a model
    model.to(torch.device(device)).eval()
    print(f"✓ Model loaded to {device}")
    print(f"✓ Config: {model.config.model_name}")
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
