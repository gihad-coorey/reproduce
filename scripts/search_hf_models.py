#!/usr/bin/env python3
"""
Search for available SmolVLA and LIBERO-related checkpoints on HuggingFace.
"""
import sys

try:
    from huggingface_hub import list_models
    print("Searching HuggingFace for SmolVLA LIBERO checkpoints...")
    print("(This may take a moment)\n")
    
    # Search for LIBERO-related SmolVLA models
    search_queries = [
        "smolvla libero",
        "libero smolvla",
        " smolvla_libero"
    ]
    
    found_models = set()
    
    for query in search_queries:
        print(f"Searching for: '{query}'")
        try:
            models = list_models(search=query, limit=20)
            for model in models:
                if "libero" in model.id.lower() and ("smolvla" in model.id.lower() or "vla" in model.id.lower()):
                    found_models.add(model.id)
                    print(f"  ✓ Found: {model.id}")
        except Exception as e:
            print(f"  Error: {e}")
    
    if found_models:
        print(f"\n{'='*80}")
        print(f"Available LIBERO SmolVLA checkpoints ({len(found_models)}):")
        print(f"{'='*80}")
        for model_id in sorted(found_models):
            print(f"  {model_id}")
        
        print(f"\nTo test any of these, run:")
        print(f"  python scripts/test_checkpoint.py --model-path <model_id> --steps 100")
    else:
        print("\n❌ No LIBERO SmolVLA checkpoints found on HuggingFace.")
        print("\nTrying broader search for any LIBERO robotics models...")
        
        models = list_models(search="libero robotics", limit=30)
        libero_models = [m.id for m in models]
        
        if libero_models:
            print(f"\nFound {len(libero_models)} LIBERO-related models:")
            for model_id in sorted(libero_models)[:15]:
                print(f"  {model_id}")
        else:
            print("No LIBERO models found either.")
            print("\nRecommendation: Consider fine-tuning from scratch using:")
            print("  lerobot-train --dataset.repo_id=lerobot/libero_combined --policy.type=smolvla")

except ImportError:
    print("❌ huggingface_hub not installed. Install with: pip install huggingface-hub")
    sys.exit(1)
