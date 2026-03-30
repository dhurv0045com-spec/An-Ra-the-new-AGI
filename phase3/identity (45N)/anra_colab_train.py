"""
AN-RA v4 FLUENT TRAINING - GOOGLE COLAB QUICK START
=====================================================
Upload these files to Colab:
  1. anra_identity_v4_fluent.txt
  2. train_identity.py
  3. test_identity.py

Then run this script OR paste these commands in Colab cells:

Cell 1:
  !pip install transformers peft datasets accelerate bitsandbytes torch

Cell 2:
  !python train_identity.py

Cell 3 (after training):
  !python test_identity.py

Cell 4 (download model):
  from google.colab import files
  !zip -r anra_model_v4.zip ./anra_model_v4/
  files.download('anra_model_v4.zip')

Training takes ~3-5 hours on Colab T4 GPU.
Loss should drop below 0.3 by epoch 50-60.
Target test score: 8/10 or higher.
"""

print("=" * 60)
print("  AN-RA v4 FLUENT TRAINING - COLAB LAUNCHER")
print("=" * 60)
print()
print("Step 1: Installing dependencies...")
import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "torch"])
print()
print("Step 2: Starting training...")
print("This will take 3-5 hours on T4 GPU.")
print()

# Run training
exec(open("train_identity.py").read())

print()
print("Step 3: Running identity tests...")
exec(open("test_identity.py").read())

print()
print("=" * 60)
print("  TRAINING COMPLETE!")
print("  Download your model: zip and download ./anra_model_v4/")
print("=" * 60)
