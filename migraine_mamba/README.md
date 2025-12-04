# MigraineMamba - Setup Guide for MacBook Air M3

## Quick Start

### 1. Create Conda Environment
```bash
# Create new environment with Python 3.10
conda create -n mamba_env python=3.10 -y
conda activate mamba_env
```

### 2. Install PyTorch with MPS Support
```bash
# Install PyTorch with Apple Silicon support
pip install torch torchvision torchaudio
```

### 3. Install Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Test PyTorch MPS
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Test Mamba implementation
python src/mamba_ssm.py

# Test full model
python src/model.py
```

## Project Structure

```
migraine_mamba/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── config.json          # Clinical parameters
│   └── migraine_synthetic.csv  # Synthetic data (from generator)
├── processed/
│   ├── train_tensors.pt     # Training data
│   ├── val_tensors.pt       # Validation data
│   ├── test_tensors.pt      # Test data
│   └── scaler_config.json   # Normalization params
├── models/
│   ├── mamba_ssl.pth        # Pre-trained weights
│   └── mamba_finetuned.pth  # Fine-tuned weights
└── src/
    ├── mamba_ssm.py         # Pure PyTorch Mamba
    ├── model.py             # MigraineMamba architecture
    ├── make_tensors.py      # Data preprocessing
    ├── train_ssl.py         # SSL pre-training (Phase 2.4)
    └── train_finetune.py    # Fine-tuning (Phase 2.5)
```

## Workflow

### Step 1: Generate Synthetic Data (Already Done)
You should have `migraine_synthetic_data.csv` from the generator.
Copy it to `data/migraine_synthetic_data.csv`

### Step 2: Preprocess Data
```bash
python src/make_tensors.py --data data/migraine_synthetic_data.csv --output processed
```

### Step 3: SSL Pre-training (Phase 2.4)
```bash
python src/train_ssl.py --epochs 50 --batch-size 64
```

### Step 4: Fine-tuning (Phase 2.5)
```bash
python src/train_finetune.py --epochs 30 --batch-size 64
```

## Important Notes for M3 Mac

1. **No CUDA Mamba**: The official `mamba-ssm` requires CUDA. We use a pure PyTorch implementation that works on MPS.

2. **MPS Backend**: PyTorch 2.1+ supports Apple Silicon via MPS (Metal Performance Shaders).

3. **num_workers=0**: DataLoader workers should be 0 on MPS to avoid multiprocessing issues.

4. **Memory**: M3 with 8GB unified memory can handle batch_size=64 comfortably.

## Expected Performance

| Metric | Expected | Notes |
|--------|----------|-------|
| Parameters | ~800K | Trainable on laptop |
| Training Time (SSL) | ~2-3 hours | 50 epochs |
| Inference | <50ms | Per prediction |
| AUC (synthetic) | 0.75-0.80 | Without data leakage |

## Troubleshooting

### "MPS not available"
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"
# Should be 2.1+ for full MPS support
```

### Out of Memory
- Reduce batch_size to 32 or 16
- Reduce d_model to 32

### Slow Training
- MPS is ~2-3x faster than CPU but slower than CUDA
- This is normal for Apple Silicon