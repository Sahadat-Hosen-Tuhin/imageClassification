import subprocess
import sys

PYTHON = sys.executable   

print("🚀 Starting Full Image Processing & Classification Pipeline")


print("\n[1/7] Image Enhancement")
subprocess.run([PYTHON, "src/image_processing/enhancement.py"])

# STEP 2: Image Denoising
print("\n[2/7] Image Denoising")
subprocess.run([PYTHON, "src/image_processing/denoising.py"])

# STEP 3: Image Restoration
print("\n[3/7] Image Restoration")
subprocess.run([PYTHON, "src/image_processing/restoration.py"])

# STEP 4: Frame Extraction + Processing
print("\n[4/7] Frame Extraction & Processing")
subprocess.run([PYTHON, "src/image_processing/frame_extractor.py"])
subprocess.run([PYTHON, "src/image_processing/frame_enhance_inplace.py"])
subprocess.run([PYTHON, "src/image_processing/frame_denoise_inplace.py"])
subprocess.run([PYTHON, "src/image_processing/frame_restore_inplace.py"])

# STEP 5: Dataset Preparation
print("\n[5/7] Preparing Classification Dataset")
subprocess.run([PYTHON, "src/image_processing/utils.py"])

# STEP 6: Model Training
print("\n[6/7] Training Classification Model")
subprocess.run([PYTHON, "src/classification/train.py"])

# STEP 7: Evaluation (Optional)
choice = input("\n❓ Do you want to evaluate the model? (yes/no): ").strip().lower()

if choice == "yes":
    print("\n📊 Running Model Evaluation")
    subprocess.run([PYTHON, "src/classification/evaluate.py"])
else:
    print("\n⏹️ Evaluation skipped by user")

# STEP 8: Prediction (Optional)
predict_choice = input("\n🔮 Do you want to run prediction on test images? (yes/no): ").strip().lower()

if predict_choice == "yes":
    print("\n🧠 Running Prediction on Test Images")
    subprocess.run([PYTHON, "src/classification/predict.py"])
else:
    print("\n⏭️ Prediction skipped by user")

print("\n✅ Pipeline Execution Completed Successfully")
