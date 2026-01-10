# Binary segmentation model for coronary angiography

## Usage

1. Setup

```
uv sync
```

2. Ensure dataset is available. If not, you can download it from s3

```
./scripts/download_dataset.sh
```

3. Train binary segmentation model

```
uv run scripts/train_binary.py
```

4. You can make prediction on chosen image and save result

```
uv run scripts/test_prediction_binary.py
```