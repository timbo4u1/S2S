import kagglehub

print("Downloading WESAD (auto-resumes if interrupted)...")
print("This will take 10-20 minutes with slow connection.")

path = kagglehub.dataset_download("orvile/wesad-wearable-stress-affect-detection-dataset")

print(f"\n✅ Downloaded to: {path}")
