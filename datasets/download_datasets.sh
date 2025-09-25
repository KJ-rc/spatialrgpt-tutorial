#!/bin/bash
# Download SpatialRGPT datasets from Google Drive
# Usage: bash download_datasets.sh

echo "📥 Downloading SpatialRGPT Tutorial Datasets..."
echo "=============================================="

echo "📋 Datasets to download:"
echo "  1. spatial_category_subsets.zip (includes full dataset + 12 categories)"
echo "  2. overlay_images.zip"
echo ""

# Install gdown if not available
pip install -q gdown

# Download spatial_category_subsets.zip
echo "⬇️  Downloading spatial_category_subsets.zip..."
gdown --fuzzy 'https://drive.google.com/file/d/1kVjG00YHcB3RQ88WX4TCGNQ5SPwy6lUw/view?usp=sharing' -O spatial_category_subsets.zip

# Download overlay_images.zip  
echo "⬇️  Downloading overlay_images.zip..."
gdown --fuzzy 'https://drive.google.com/file/d/1JeqbQ8b6-k-czLGZBejR560TPz_FOSei/view?usp=sharing' -O overlay_images.zip

# Extract files
echo "📦 Extracting files..."
unzip -o -q spatial_category_subsets.zip
unzip -o -q overlay_images.zip

echo "✅ Datasets downloaded and extracted!"
echo "📁 Final structure:"
echo "   datasets/spatial_category_subsets/SpatialRGPT-Bench_v1_with_overlay.json (full dataset)"
echo "   datasets/spatial_category_subsets/*.jsonl (12 category subsets)"
echo "   datasets/overlay_images/*.jpg (overlay images)"

echo "🎉 Ready to use! Run: python eval_spatial_tutorial.py"