# Selfie to Art
Bot for creating personalized images

## Installation
```
git clone https://github.com/dslvt/shrex
cd shrex

pip install -r requirements.txt
```

## Download checkpoints
```
mkdir models
wget -O ./models/inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx

```

## Inference
```
touch .env
python bot.py
```