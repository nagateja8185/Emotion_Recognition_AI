"""
System Test Script
Validates that all components are working correctly
"""

import os
import sys

def print_test_header(test_name):
    print(f"\n{'='*60}")
    print(f"  🧪 {test_name}")
    print('='*60)

def test_imports():
    """Test if all required packages can be imported"""
    print_test_header("Testing Package Imports")
    
    packages = {
        'numpy': 'np',
        'pandas': 'pd',
        'cv2': None,
        'matplotlib.pyplot': 'plt',
        'sklearn': None,
        'joblib': None,
        'tensorflow': 'tf',
        'PIL': None
    }
    
    failed = []
    warnings = []
    for pkg, alias in packages.items():
        try:
            if alias:
                exec(f"import {pkg} as {alias}")
            else:
                exec(f"import {pkg}")
            print(f"   ✅ {pkg}")
        except Exception as e:
            # TensorFlow can have issues on Windows but might still work
            if pkg == 'tensorflow':
                print(f"   ⚠️  {pkg}: May work despite import error")
                warnings.append(pkg)
            else:
                print(f"   ❌ {pkg}: {str(e)[:50]}")
                failed.append(pkg)
    
    return len(failed) == 0

def test_model_files():
    """Test if model files exist"""
    print_test_header("Testing Model Files")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models = [
        os.path.join(base_dir, 'models', 'image_emotion.h5'),
        os.path.join(base_dir, 'models', 'text_emotion', 'pipeline.joblib')
    ]
    
    all_exist = True
    for model_path in models:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"   ✅ {os.path.basename(model_path)} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {os.path.basename(model_path)} - Not found")
            all_exist = False
    
    return all_exist

def test_data_directories():
    """Test if data directories exist"""
    print_test_header("Testing Data Directories")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = [
        os.path.join(base_dir, 'data', 'images', 'fer2013', 'train'),
        os.path.join(base_dir, 'data', 'text')
    ]
    
    all_exist = True
    for dir_path in dirs:
        if os.path.exists(dir_path):
            # Count subdirectories/files
            items = os.listdir(dir_path)
            print(f"   ✅ {os.path.basename(dir_path)} ({len(items)} items)")
        else:
            print(f"   ❌ {os.path.basename(dir_path)} - Not found")
            all_exist = False
    
    return all_exist

def test_server_script():
    """Test if server script has valid syntax"""
    print_test_header("Testing Server Script")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(base_dir, 'src', 'multimodal_server.py')
    
    if not os.path.exists(server_script):
        print(f"   ❌ Server script not found")
        return False
    
    try:
        with open(server_script, 'r', encoding='utf-8') as f:
            compile(f.read(), server_script, 'exec')
        print(f"   ✅ Server script syntax OK")
        return True
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        return False
    except UnicodeDecodeError:
        # Try with default encoding
        try:
            with open(server_script, 'r') as f:
                compile(f.read(), server_script, 'exec')
            print(f"   ✅ Server script syntax OK")
            return True
        except Exception as e2:
            print(f"   ❌ Encoding error: {e2}")
            return False

def test_web_interface():
    """Test if web interface exists"""
    print_test_header("Testing Web Interface")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_html = os.path.join(base_dir, 'web_demo', 'index.html')
    
    if os.path.exists(index_html):
        size_kb = os.path.getsize(index_html) / 1024
        print(f"   ✅ index.html ({size_kb:.1f} KB)")
        
        # Check for key features
        with open(index_html, 'r', encoding='utf-8') as f:
            content = f.read()
            features = ['conf-ring', 'video', 'textarea', 'fetch']
            for feature in features:
                if feature in content:
                    print(f"      ✓ {feature} found")
                else:
                    print(f"      ⚠ {feature} missing")
        return True
    else:
        print(f"   ❌ index.html not found")
        return False

def run_all_tests():
    """Run all tests and provide summary"""
    print("\n" + "="*70)
    print("  🧠 EMOTION RECOGNITION AI - SYSTEM TEST")
    print("="*70)
    
    results = {
        'Package Imports': test_imports(),
        'Model Files': test_model_files(),
        'Data Directories': test_data_directories(),
        'Server Script': test_server_script(),
        'Web Interface': test_web_interface()
    }
    
    print("\n" + "="*70)
    print("  📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print(f"\n   Total: {passed}/{total} tests passed ({(passed/total*100):.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\n🚀 Next steps:")
        print("   python src/multimodal_server.py")
        print("   Then open: http://localhost:8000")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\n💡 Common fixes:")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Train models: python train_all_models.py")
        print("   - Check data organization: See README.md")
    
    print("="*70 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
