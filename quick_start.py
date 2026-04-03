"""
Quick Start Script for Emotion Detection System
Checks dependencies, validates setup, and provides easy startup
"""

import os
import sys
import subprocess

def print_banner():
    print("\n" + "="*70)
    print("  🧠 EMOTION RECOGNITION AI - QUICK START")
    print("="*70 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print("📌 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n📦 Checking dependencies...")
    required = [
        'numpy', 'pandas', 'cv2', 'matplotlib', 'sklearn', 
        'joblib', 'tensorflow', 'PIL'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg} - MISSING")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\n💡 Install with:")
        print("   pip install -r requirements.txt\n")
        return False
    
    print("\n   ✅ All dependencies installed!")
    return True

def check_models():
    """Check if trained models exist"""
    print("\n🤖 Checking model files...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models = {
        'Image Model': os.path.join(base_dir, 'models', 'image_emotion.h5'),
        'Text Model': os.path.join(base_dir, 'models', 'text_emotion', 'pipeline.joblib')
    }
    
    all_exist = True
    for name, path in models.items():
        if os.path.exists(path):
            print(f"   ✅ {name} - Found")
        else:
            print(f"   ❌ {name} - Not found")
            all_exist = False
    
    if not all_exist:
        print("\n💡 Train models with:")
        print("   python train_all_models.py")
        print("   OR train individually:")
        print("   python src/train_image.py")
        print("   python src/train_text.py\n")
    
    return all_exist

def check_data():
    """Check if data directories exist"""
    print("\n📊 Checking data directories...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = {
        'Image Training Data': os.path.join(base_dir, 'data', 'images', 'fer2013', 'train'),
        'Text Training Data': os.path.join(base_dir, 'data', 'text')
    }
    
    all_exist = True
    for name, path in dirs.items():
        if os.path.exists(path):
            print(f"   ✅ {name} - Found")
        else:
            print(f"   ❌ {name} - Not found")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Please ensure your data is properly organized.")
        print("   See README.md for dataset structure.\n")
    
    return all_exist

def start_server():
    """Start the emotion detection server"""
    print("\n🚀 Starting Emotion Detection Server...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(base_dir, 'src', 'multimodal_server.py')
    
    if not os.path.exists(server_script):
        print(f"   ❌ Server script not found: {server_script}")
        return
    
    print("\n✅ Server starting...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([sys.executable, server_script], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def main():
    print_banner()
    
    # Run checks
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    models_ok = check_models()
    data_ok = check_data()
    
    print("\n" + "="*70)
    
    if not python_ok:
        print("\n❌ Python version check failed. Please upgrade to Python 3.8+")
        return
    
    if not deps_ok:
        print("\n❌ Dependency check failed. Please install requirements.")
        response = input("\nInstall now? (y/n): ").strip().lower()
        if response == 'y':
            print("\n📦 Installing dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            deps_ok = check_dependencies()
        else:
            return
    
    if not models_ok:
        print("\n❌ Models not found. Training required.")
        response = input("\nTrain models now? (y/n): ").strip().lower()
        if response == 'y':
            train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_all_models.py')
            if os.path.exists(train_script):
                subprocess.run([sys.executable, train_script])
                models_ok = check_models()
            else:
                print("❌ Training script not found")
                return
        else:
            print("\n⚠️  You can train models later with: python train_all_models.py")
    
    if deps_ok and models_ok:
        print("\n✅ All checks passed! Ready to start.\n")
        start_server()
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.\n")

if __name__ == "__main__":
    main()
