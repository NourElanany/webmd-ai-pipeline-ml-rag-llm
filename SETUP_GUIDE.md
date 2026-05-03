# WebMD AI Pipeline — Setup Guide

## Quick Start (5 Minutes)

### Step 1: Install Python
Make sure you have Python 3.8 or higher installed:
```bash
python --version
```

### Step 2: Clone Repository
```bash
git clone https://github.com/nourelanany/webmd-ai-pipeline.git
cd webmd-ai-pipeline
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install all required packages (~2-3 minutes).

### Step 4: Download Dataset
1. Go to [Kaggle WebMD Drug Reviews](https://www.kaggle.com/datasets)
2. Download `webmd.csv`
3. Place it in the project root directory

### Step 5: Run the Application
```bash
python app.py
```

That's it! The unified dashboard will launch with all features.

---

## Detailed Setup

### Option A: Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option B: Conda Environment
```bash
conda create -n webmd python=3.9
conda activate webmd
pip install -r requirements.txt
```

---

## Optional: LLM Integration

To enable LLM-powered answers in the RAG system:

1. **Get API Key:**
   - Visit [openrouter.ai](https://openrouter.ai)
   - Sign up for a free account
   - Copy your API key

2. **Create .env file:**
   ```bash
   cp .env.example .env
   ```

3. **Edit .env:**
   ```
   OPENROUTER_API_KEY=your_actual_key_here
   OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
   ```

4. **Run RAG system:**
   ```bash
   python rag_system.py
   ```

The system works without an API key (template mode), but LLM features require a valid key.

---

## Running Individual Phases

### Phase 1: EDA
```bash
python analysis.py
```
**Output:**
- `webmd_cleaned.csv` (cleaned dataset)
- `plots/` folder with 8 visualizations
- Interactive EDA dashboard

**Time:** ~2-3 minutes

---

### Phase 2: Machine Learning
```bash
python ml_model.py
```
**Requirements:** `webmd_cleaned.csv` must exist (run Phase 1 first)

**Output:**
- `rf_effectiveness_model.pkl` (trained model)
- `ml_plots/` folder with 5 visualizations
- Interactive ML dashboard with live predictor

**Time:** ~5-10 minutes (depending on CPU)

---

### Phase 3: Deep Learning NLP
```bash
python dl_nlp.py
```
**Requirements:** `webmd_cleaned.csv` must exist

**Output:**
- `lstm_sentiment_model.keras` (trained model)
- `nlp_plots/` folder with 6 visualizations
- Interactive NLP dashboard with live analyzer

**Time:** ~10-20 minutes (CPU) or ~3-5 minutes (GPU)

**GPU Acceleration:**
If you have an NVIDIA GPU with CUDA:
```bash
pip install tensorflow-gpu
```

---

### Phase 4+5: RAG + LLM
```bash
python rag_system.py
```
**Requirements:** `webmd_cleaned.csv` must exist

**Output:**
- `chroma_db/` folder (vector index)
- Interactive RAG Q&A interface

**Time:** ~5-10 minutes (first run to build index)

**Note:** Subsequent runs are instant (index is cached)

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Install missing package
```bash
pip install <package_name>
```

### Issue: "webmd.csv not found"
**Solution:** Download dataset from Kaggle and place in project root

### Issue: "Out of memory" during training
**Solution:** Reduce sample size in the code:
- `analysis.py`: No changes needed
- `ml_model.py`: Already optimized
- `dl_nlp.py`: Change `sample_size=80000` to `sample_size=40000`
- `rag_system.py`: Change `SAMPLE_SIZE = 50000` to `SAMPLE_SIZE = 20000`

### Issue: TensorFlow GPU not working
**Solution:**
1. Check CUDA installation: `nvidia-smi`
2. Install correct TensorFlow version:
   ```bash
   pip install tensorflow-gpu==2.10.0
   ```
3. Verify GPU detection:
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

### Issue: ChromaDB build fails
**Solution:** Install build tools
- **Windows:** Install Visual C++ Build Tools
- **macOS:** `xcode-select --install`
- **Linux:** `sudo apt-get install build-essential`

### Issue: Tkinter not found
**Solution:**
- **Windows:** Included with Python
- **macOS:** `brew install python-tk`
- **Linux:** `sudo apt-get install python3-tk`

---

## Running in Google Colab

### Gradio Demo (Recommended for Colab)

1. **Upload files to Colab:**
   - `gradio_demo.py`
   - `webmd_cleaned.csv`

2. **Install dependencies:**
   ```python
   !pip install gradio tensorflow scikit-learn chromadb sentence-transformers
   ```

3. **Run demo:**
   ```python
   !python gradio_demo.py
   ```

4. **Access via public link** (automatically generated)

### Jupyter Notebook

1. **Upload to Colab:**
   - `WebMD_AI_Pipeline.ipynb`
   - `webmd_cleaned.csv`

2. **Run cells sequentially**

---

## System Requirements

### Minimum:
- **CPU:** Dual-core 2.0 GHz
- **RAM:** 8 GB
- **Storage:** 5 GB free space
- **OS:** Windows 10, macOS 10.14, Ubuntu 18.04 or newer

### Recommended:
- **CPU:** Quad-core 3.0 GHz or better
- **RAM:** 16 GB
- **GPU:** NVIDIA GPU with 4GB+ VRAM (for faster training)
- **Storage:** 10 GB free space
- **OS:** Latest version

---

## Performance Tips

1. **Use GPU for deep learning:**
   - 10x faster training
   - Install `tensorflow-gpu`

2. **Reduce dataset size for testing:**
   - Modify sample sizes in code
   - Faster iteration during development

3. **Cache models:**
   - Models are saved after training
   - Subsequent runs load from disk (instant)

4. **Use SSD:**
   - Faster data loading
   - Better overall performance

---

## Next Steps

After setup:

1. **Explore the unified dashboard:**
   ```bash
   python app.py
   ```

2. **Read the concept explanation:**
   - Open `PROJECT_EXPLANATION.md`
   - Understand how each phase works

3. **Try the Jupyter notebook:**
   - Step-by-step code walkthrough
   - Experiment with parameters

4. **Deploy Gradio demo:**
   - Share with others
   - Get feedback

5. **Customize for your needs:**
   - Modify models
   - Add new features
   - Integrate with other systems

---

## Getting Help

- **Documentation:** `README.md` and `PROJECT_EXPLANATION.md`
- **Code comments:** Extensive inline documentation
- **GitHub Issues:** Report bugs or ask questions
- **Email:** your.email@example.com

---

## Common Workflows

### Workflow 1: Quick Demo
```bash
pip install -r requirements.txt
python gradio_demo.py
```

### Workflow 2: Full Pipeline
```bash
python analysis.py
python ml_model.py
python dl_nlp.py
python rag_system.py
```

### Workflow 3: Development
```bash
jupyter notebook WebMD_AI_Pipeline.ipynb
# Experiment with code
# Modify and test
python app.py  # Test in dashboard
```

### Workflow 4: Production Deployment
```bash
# Train all models
python analysis.py
python ml_model.py
python dl_nlp.py
python rag_system.py

# Deploy Gradio app
python gradio_demo.py
# Or integrate into existing system
```

---

## Updates & Maintenance

### Updating Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Rebuilding Models
Delete model files and re-run training scripts:
```bash
rm *.pkl *.keras
python ml_model.py
python dl_nlp.py
```

### Rebuilding Vector Index
Delete ChromaDB folder and re-run:
```bash
rm -rf chroma_db/
python rag_system.py
```

---

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded (`webmd.csv`)
- [ ] Phase 1 completed (`webmd_cleaned.csv` exists)
- [ ] Models trained (`.pkl` and `.keras` files exist)
- [ ] Vector index built (`chroma_db/` folder exists)
- [ ] Dashboard launches successfully (`python app.py`)
- [ ] (Optional) LLM API key configured (`.env` file)

---

**Ready to start? Run `python app.py` and explore!**
