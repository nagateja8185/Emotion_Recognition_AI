"""
Complete Training Pipeline for Emotion Detection
Trains both image and text models sequentially
"""

import subprocess
import sys
import os

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def main():
    print_header("🚀 COMPLETE EMOTION DETECTION TRAINING PIPELINE")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("📂 Script directory:", script_dir)
    print("\nThis script will train both emotion detection models:")
    print("1. 📸 Image Emotion Detection (CNN)")
    print("2. 💬 Text Emotion Detection (NLP)")
    print("\n⚠️  This process may take 30-60 minutes depending on your hardware.\n")
    
    response = input("Continue? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Training cancelled.")
        return
    
    # Train image model
    print_header("📸 STEP 1: Training Image Emotion Detection Model")
    image_script = os.path.join(script_dir, "train_image.py")
    
    if not os.path.exists(image_script):
        print(f"❌ Error: {image_script} not found!")
        return
    
    try:
        result = subprocess.run([sys.executable, image_script], check=True)
        print("\n✅ Image model training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Image model training failed: {e}")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Train text model
    print_header("💬 STEP 2: Training Text Emotion Detection Model")
    text_script = os.path.join(script_dir, "train_text.py")
    
    if not os.path.exists(text_script):
        print(f"❌ Error: {text_script} not found!")
        return
    
    try:
        result = subprocess.run([sys.executable, text_script], check=True)
        print("\n✅ Text model training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Text model training failed: {e}")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    print_header("🎉 TRAINING COMPLETE!")
    print("\n✅ Both models have been trained successfully!")
    print("\n📁 Model locations:")
    print("   - Image model: models/image_emotion.h5")
    print("   - Text model: models/text_emotion/pipeline.joblib")
    print("\n🚀 Next steps:")
    print("   1. Run: python src/multimodal_server.py")
    print("   2. Open browser: http://localhost:8000")
    print("   3. Test both text and facial emotion detection!\n")

if __name__ == "__main__":
    main()
