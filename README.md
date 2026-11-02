# OpenCV + PyTorch Image Processing and Classification

This project processes still images and video frames with OpenCV, then trains a
ResNet-18 image classifier with PyTorch. The processing pipeline applies
contrast enhancement, Gaussian denoising, and sharpening. Video files are
sampled into frames and processed in place.

> **Important:** The current dataset-preparation script assigns the labels
> `effective` and `ineffective` by alternating through filenames. These are not
> ground-truth labels. As a result, model training, evaluation, and predictions
> are only a pipeline demonstration and should not be treated as meaningful
> classification results until the dataset labels are corrected.

## Features

- CLAHE contrast enhancement in LAB color space.
- Gaussian-blur denoising and kernel-based sharpening.
- Extraction of video frames for downstream image processing.
- Transfer learning with a pretrained ResNet-18; only its final layer is
  trainable.
- Optional validation evaluation and prediction with a confidence threshold.

## Requirements

- Python 3.10 or later is recommended.
- Dependencies listed in `requirements.txt`.
- A CUDA-capable GPU is optional; the scripts select CUDA when PyTorch reports
  it available and otherwise use the CPU.

Install the dependencies from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, use the Python
interpreter in `.venv\Scripts\python.exe` directly or adjust the execution
policy for your environment.

## Input data

Run commands from the project root because scripts use paths relative to the
current working directory.

- Put still images (`.jpg`, `.jpeg`, or `.png`) in `data/raw/images/`.
- Put videos (`.mp4`, `.avi`, or `.mov`) in `data/raw/videos/`.
- Put images to classify in `data/test_images/`.
- The existing `data/raw/damaged/` folder is not used by the current scripts.

The directories can be created as needed. Keep your original inputs safe: the
frame processing scripts modify extracted frame files in place.

## Run the pipeline

From the activated environment and project root:

```powershell
python main.py
```

The orchestrator runs these steps in order:

1. Enhance images from `data/raw/images/` into `data/processed/enhanced/`.
2. Denoise those enhanced images into `data/processed/denoised/`.
3. Sharpen denoised images into `data/processed/restored/`.
4. Extract every 30th frame from videos in `data/raw/videos/`, then enhance,
   denoise, and sharpen the extracted frames in place in
   `data/processed/frames/`.
5. Build the classification folders from restored images and processed frames.
6. Train the classifier and save its state dictionary to
   `models/resnet18_classification.pth`.
7. Ask whether to evaluate on the validation split.
8. Ask whether to predict labels for images in `data/test_images/`.

The script launches each stage as a subprocess. If a stage fails, check its
output and resolve the error before relying on later-stage results.

You can also run individual stages from the project root:

```powershell
python src/image_processing/enhancement.py
python src/image_processing/denoising.py
python src/image_processing/restoration.py
python src/image_processing/frame_extractor.py
python src/image_processing/frame_enhance_inplace.py
python src/image_processing/frame_denoise_inplace.py
python src/image_processing/frame_restore_inplace.py
python src/image_processing/utils.py
python src/classification/train.py
python src/classification/evaluate.py
python src/classification/predict.py
```

Run the preprocessing steps before dataset preparation and training. Evaluation
expects a trained checkpoint at `models/resnet18_classification.pth`. Prediction
reads images from `data/test_images/` and reports `Uncertain` when the top
softmax confidence is below `0.70`.

## Dataset and model notes

- Dataset preparation collects images from `data/processed/restored/` and
  `data/processed/frames/`, shuffles them, and splits them 80/20 into training
  and validation sets.
- The script copies images into the class folders but does not clear existing
  files. Re-running it can therefore leave stale or duplicate training data;
  clean or recreate `data/classification/` before preparing a fresh split.
- The class assignment currently alternates by index within each split. Replace
  this with verified labels before training a model intended for real use.
- Training runs for five epochs with batch size eight and saves the checkpoint
  to `models/resnet18_classification.pth`.
- Evaluation prints accuracy and a confusion matrix. Prediction prints one
  label and confidence per test image; the current scripts do not save these
  results to the `results/` or `report/` directories.

## Project layout

```text
main.py                         Run the end-to-end pipeline
requirements.txt                Python dependencies
src/
  image_processing/              Image and video-frame processing scripts
  classification/                Dataset loader, model, training and inference
data/
  raw/images/                    Still-image inputs
  raw/videos/                    Video inputs
  processed/                     Intermediate and processed images/frames
  classification/                Generated train/validation folders
  test_images/                   Images used by prediction
models/                           Saved model checkpoint
results/                          Reserved for outputs (not currently written)
report/                           Reserved for reports (not currently written)
```