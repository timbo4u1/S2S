#!/bin/bash
mkdir -p ~/wesad_data
cd ~/wesad_data

ATTEMPT=1
while true; do
    echo "Attempt $ATTEMPT: Downloading WESAD..."
    
    kaggle datasets download -d orvile/wesad-wearable-stress-affect-detection-dataset
    
    if [ $? -eq 0 ]; then
        echo "✅ Download complete!"
        unzip -q wesad-wearable-stress-affect-detection-dataset.zip
        echo "✅ Extracted!"
        ls -lh
        break
    else
        echo "❌ Failed. Retry in 10s..."
        sleep 10
        ATTEMPT=$((ATTEMPT + 1))
    fi
done
