#!/usr/bin/env python3
"""
Download Pretrained PPE Models from Kaggle
Automated download of community PPE detection models
"""

import os
import sys
from pathlib import Path

print("="*70)
print("KAGGLE PPE MODEL DOWNLOADER")
print("="*70)

# Check if Kaggle API is configured
kaggle_dir = Path.home() / '.kaggle'
kaggle_json = kaggle_dir / 'kaggle.json'

print("\n[INFO] Checking Kaggle API configuration...")

if not kaggle_json.exists():
    print("\n" + "="*70)
    print("[WARNING] KAGGLE API NOT CONFIGURED")
    print("="*70)
    print("\nTo download from Kaggle, you need to set up your API key:")
    print("\nSTEPS:")
    print("1. Go to: https://www.kaggle.com/")
    print("2. Sign in (or create free account)")
    print("3. Click your profile picture > Settings")
    print("4. Scroll to 'API' section")
    print("5. Click 'Create New Token'")
    print("6. This downloads 'kaggle.json'")
    print("\n7. Place kaggle.json in:")
    print(f"   {kaggle_dir}")
    print("\n8. Then run this script again!")
    
    print("\n" + "="*70)
    print("ALTERNATIVE: Download Manually")
    print("="*70)
    print("\nYou can also manually download from these direct links:")
    print("\n1. Hard Hat Detection:")
    print("   https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection")
    print("   Click 'Download' button")
    
    print("\n2. Construction Safety Dataset:")
    print("   https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow")
    print("   Click 'Download' button")
    
    print("\n3. PPE Detection Dataset:")
    print("   https://www.kaggle.com/datasets/andrewmvd/ppe-detection-dataset")
    print("   Click 'Download' button")
    
    sys.exit(1)

else:
    print("[OK] Kaggle API configured!")
    
    # Import kaggle
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[OK] Kaggle authentication successful!")
    except Exception as e:
        print(f"[ERROR] Kaggle authentication failed: {e}")
        sys.exit(1)
    
    # Download datasets
    print("\n" + "="*70)
    print("DOWNLOADING PPE DATASETS")
    print("="*70)
    
    datasets = [
        {
            'name': 'hard-hat-detection',
            'owner': 'andrewmvd',
            'folder': 'hard_hat_dataset',
            'description': 'Hard Hat Detection Dataset'
        },
        {
            'name': 'construction-site-safety-image-dataset-roboflow',
            'owner': 'snehilsanyal',
            'folder': 'construction_safety',
            'description': 'Construction Site Safety'
        },
    ]
    
    downloaded = []
    
    for ds in datasets:
        print(f"\n[INFO] Downloading: {ds['description']}...")
        try:
            dataset_path = f"{ds['owner']}/{ds['name']}"
            output_dir = ds['folder']
            
            # Download dataset
            api.dataset_download_files(dataset_path, path=output_dir, unzip=True)
            
            print(f"[OK] Downloaded to: {output_dir}/")
            downloaded.append(ds)
            
        except Exception as e:
            print(f"[ERROR] Failed to download {ds['description']}: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    
    if downloaded:
        print(f"\n[OK] Successfully downloaded {len(downloaded)} datasets:")
        for ds in downloaded:
            print(f"  - {ds['description']}: {ds['folder']}/")
        
        print("\n[INFO] Next steps:")
        print("  1. Check the downloaded folders")
        print("  2. Look for .pt model files")
        print("  3. Or use the datasets to supplement your training")
    else:
        print("\n[ERROR] No datasets downloaded")
        print("[INFO] Please check your internet connection")

if __name__ == "__main__":
    pass

