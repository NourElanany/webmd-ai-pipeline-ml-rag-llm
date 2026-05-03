# Troubleshooting Guide

## Common Issues and Solutions

---

## 🔴 Issue: "Could not import module 'PreTrainedModel'" or transformers errors

### **Cause:**
Python 3.13 compatibility issue with `transformers` and `sentence-transformers` libraries.

### **Solution 1: Use Python 3.10 or 3.11 (Recommended)**

**Using Conda:**
```bash
# Create new environment with Python 3.11
conda create -n webmd python=3.11
conda activate webmd
pip install -r requirements.txt
```

**Using pyenv:**
```bash
pyenv install 3.11.0
pyenv local 3.11.0
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **Solution 2: Update to Latest Versions**
```bash
pip install --upgrade pip
pip install --upgrade transformers sentence-transformers torch
```

### **Solution 3: Skip RAG Phase Temporarily**
If you just want to test other phases:
```bash
python analysis.py    # ✅ Works
python ml_model.py    # ✅ Works
python dl_nlp.py      # ✅ Works
# Skip rag_system.py and gradio_demo.py for now
```

---

## 🔴 Issue: "ModuleNotFoundError: No module named 'X'"

### **Solution:**
Install the missing package:
```bash
pip install <package_name>
```

Or reinstall all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔴 Issue: "FileNotFoundError: webmd.csv not found"

### **Solution:**
1. Download the WebMD Drug Reviews dataset from Kaggle
2. Place `webmd.csv` in the project root directory
3. Run Phase 1 first: `python analysis.py`

---

## 🔴 Issue: "Out of memory" during training

### **Solution 1: Reduce Sample Size**

**For dl_nlp.py:**
```python
# Line 103: Change from 80000 to 40000
def load_nlp_data(path="webmd_cleaned.csv", sample_size=40000):
```

**For rag_system.py:**
```python
# Line 23: Change from 50000 to 20000
SAMPLE_SIZE = 20000
```

### **Solution 2: Close Other Applications**
Free up RAM by closing unnecessary programs.

### **Solution 3: Use Smaller Batch Size**

**For dl_nlp.py:**
```python
# Line 60: Change batch size
BATCH_SIZE = 128 if gpus else 64  # Reduced from 1024/256
```

---

## 🔴 Issue: TensorFlow GPU not detected

### **Check GPU:**
```bash
nvidia-smi
```

### **Install CUDA-compatible TensorFlow:**
```bash
pip install tensorflow[and-cuda]
```

### **Verify GPU in Python:**
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### **If still not working:**
- Check CUDA version compatibility
- Install correct cuDNN version
- Use CPU version (slower but works): `pip install tensorflow-cpu`

---

## 🔴 Issue: ChromaDB build fails

### **Windows:**
Install Visual C++ Build Tools:
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Install "Desktop development with C++"

### **macOS:**
```bash
xcode-select --install
```

### **Linux:**
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

---

## 🔴 Issue: Tkinter not found

### **Windows:**
Tkinter is included with Python. Reinstall Python if missing.

### **macOS:**
```bash
brew install python-tk@3.11
```

### **Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3-tk
```

### **Linux (Fedora):**
```bash
sudo dnf install python3-tkinter
```

---

## 🔴 Issue: NLTK data not found

### **Solution:**
```python
import nltk
nltk.download('stopwords')
```

Or download all NLTK data:
```python
nltk.download('all')
```

---

## 🔴 Issue: "RuntimeError: CUDA out of memory"

### **Solution 1: Reduce Batch Size**
```python
# In dl_nlp.py, line 60
BATCH_SIZE = 64  # Reduce from 256 or 1024
```

### **Solution 2: Use CPU Instead**
```python
# Add at the top of dl_nlp.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### **Solution 3: Clear GPU Memory**
```python
import tensorflow as tf
tf.keras.backend.clear_session()
```

---

## 🔴 Issue: Gradio demo not loading models

### **Solution:**
Make sure you've run the training phases first:
```bash
python analysis.py    # Creates webmd_cleaned.csv
python ml_model.py    # Creates rf_effectiveness_model.pkl
python dl_nlp.py      # Creates lstm_sentiment_model.keras
python rag_system.py  # Creates chroma_db/
```

Then run:
```bash
python gradio_demo.py
```

---

## 🔴 Issue: "PermissionError" when saving files

### **Windows:**
- Run terminal as Administrator
- Check antivirus isn't blocking Python

### **macOS/Linux:**
```bash
chmod +w .
```

---

## 🔴 Issue: Plots not displaying in Jupyter

### **Solution:**
Add at the top of notebook:
```python
%matplotlib inline
```

Or use:
```python
import matplotlib.pyplot as plt
plt.show()
```

---

## 🔴 Issue: OpenRouter API key not working

### **Check .env file:**
```bash
# Make sure .env exists in project root
cat .env  # Linux/Mac
type .env  # Windows
```

### **Verify format:**
```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

### **Test API key:**
```python
import os
from dotenv import load_dotenv
load_dotenv()
print(os.environ.get('OPENROUTER_API_KEY'))
```

---

## 🔴 Issue: Slow training on CPU

### **Expected Times (CPU):**
- Phase 1 (EDA): 2-3 minutes
- Phase 2 (ML): 5-10 minutes
- Phase 3 (DL/NLP): 15-30 minutes ⚠️ Slow
- Phase 4 (RAG): 5-10 minutes

### **Speed up options:**
1. **Use GPU** (10x faster for Phase 3)
2. **Reduce sample size** (see "Out of memory" section)
3. **Use pre-trained models** (skip training)

---

## 🔴 Issue: Dashboard window too small/large

### **Solution:**
Edit the geometry in the script:

**For app.py:**
```python
# Line ~880
root.geometry("1350x820")  # Change to your preferred size
```

**For individual phase scripts:**
```python
# Find the line with root.geometry()
root.geometry("1280x780")  # Adjust as needed
```

---

## 🔴 Issue: "ValueError: could not convert string to float"

### **Cause:**
Missing or invalid data in CSV.

### **Solution:**
Re-run Phase 1 to clean data:
```bash
python analysis.py
```

This will regenerate `webmd_cleaned.csv` with proper data types.

---

## 🔴 Issue: Import errors in Jupyter Notebook

### **Solution:**
Make sure kernel matches your environment:
```bash
# Install ipykernel in your environment
pip install ipykernel
python -m ipykernel install --user --name=webmd

# In Jupyter, select Kernel > Change Kernel > webmd
```

---

## 🔴 Issue: Gradio share link not working

### **Solution 1: Check firewall**
Allow Python through firewall.

### **Solution 2: Use different port**
```python
demo.launch(share=True, server_port=7861)
```

### **Solution 3: Local only**
```python
demo.launch(share=False)  # Access via localhost only
```

---

## 🟡 Python Version Compatibility Matrix

| Python Version | Status | Notes |
|----------------|--------|-------|
| 3.8 | ✅ Supported | Minimum version |
| 3.9 | ✅ Supported | Recommended |
| 3.10 | ✅ Supported | Recommended |
| 3.11 | ✅ Supported | **Best choice** |
| 3.12 | ⚠️ Partial | Some packages may have issues |
| 3.13 | ❌ Not supported | transformers incompatible |

---

## 🟢 Recommended Setup

### **Best Configuration:**
```bash
# Python 3.11 with conda
conda create -n webmd python=3.11
conda activate webmd
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow, sklearn, chromadb; print('All packages installed!')"
```

### **Minimal Configuration (No GPU):**
```bash
pip install numpy pandas matplotlib seaborn scipy
pip install scikit-learn xgboost joblib
pip install tensorflow-cpu nltk
pip install Pillow python-dotenv
```

---

## 📞 Still Having Issues?

1. **Check Python version:**
   ```bash
   python --version
   ```
   Should be 3.8-3.11

2. **Update pip:**
   ```bash
   pip install --upgrade pip
   ```

3. **Clean install:**
   ```bash
   pip uninstall -y -r requirements.txt
   pip install -r requirements.txt
   ```

4. **Check logs:**
   Look for error messages in terminal output

5. **Create GitHub issue:**
   Include:
   - Python version
   - OS (Windows/Mac/Linux)
   - Full error message
   - Steps to reproduce

---

## ✅ Verification Checklist

After fixing issues, verify everything works:

```bash
# Test imports
python -c "import numpy, pandas, sklearn, tensorflow, chromadb"

# Test Phase 1
python analysis.py

# Test Phase 2
python ml_model.py

# Test Phase 3
python dl_nlp.py

# Test Phase 4
python rag_system.py

# Test unified dashboard
python app.py
```

---

## 🎯 Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| Python 3.13 errors | Use Python 3.11 |
| Missing modules | `pip install -r requirements.txt` |
| Out of memory | Reduce sample sizes |
| GPU not detected | Install tensorflow[and-cuda] |
| Slow training | Use GPU or reduce data |
| File not found | Run phases in order |
| API key issues | Check .env file format |

---

**Most issues are solved by using Python 3.10 or 3.11 instead of 3.13!**
