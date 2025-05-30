# Tubes 2 IF3270 Machine Learning - Neural Network Implementation from Scratch

Repository ini berisi implementasi dari scratch berbagai arsitektur neural network untuk tugas klasifikasi, termasuk CNN, RNN, dan LSTM. Proyek ini merupakan bagian dari Tugas Besar 2 mata kuliah IF3270 Machine Learning.

## Deskripsi

Tugas besar 2 ini mengimplementasikan tiga jenis neural network dari scratch:
- **CNN (Convolutional Neural Network)** - untuk klasifikasi gambar
- **RNN (Recurrent Neural Network)** - untuk pemrosesan data sekuensial
- **LSTM (Long Short-Term Memory)** - untuk klasifikasi teks dengan sentiment analysis

Setiap implementasi mencakup:
- Training menggunakan Keras sebagai baseline
- Implementasi forward propagation dari scratch
- Perbandingan hasil antara Keras dan implementasi from scratch
- Analisis pengaruh hyperparameter terhadap performa model

## Struktur Repository

```
tubes2_if3270_ml/
├── src/
│   ├── cnn/                           # CNN Implementation
│   │   ├── cnn_keras_training.ipynb   # CNN training dengan Keras
│   │   ├── cnn_from_scratch_testing.ipynb # Testing implementasi from scratch
│   │   ├── from_scratch/              # CNN layers dari scratch
│   │   │   ├── layers.py              # CNN layers implementation
│   │   │   └── model.py               # CNN model implementation
│   │   ├── cnn_keras.weights.h5       # Model weights Keras
│   │   ├── cnn_keras_final.weights.h5 # Model weights final
│   │   └── best_model_config.json     # Konfigurasi model terbaik
│   ├── rnn/                           # RNN Implementation
│   │   ├── notebook/
│   │   │   ├── rnn_keras_training.ipynb # RNN training dengan Keras
│   │   │   ├── rnn_from_scratch.ipynb # Testing implementasi from scratch
│   │   │   └── rnn_experiment_results.json # Hasil eksperimen RNN
│   │   ├── from_scratch/              # RNN layers dari scratch
│   │   │   ├── layers.py              # RNN layers implementation
│   │   │   └── model.py               # RNN model implementation
│   │   ├── rnn_keras_best.weights.h5  # Model weights terbaik
│   │   ├── rnn_keras_final.weights.h5 # Model weights final
│   │   └── rnn_saved_data.pkl         # Data test untuk validasi
│   ├── lstm/                          # LSTM Implementation
│   │   ├── lstm.ipynb                 # LSTM lengkap (training + testing)
│   │   ├── lstm_keras_training.ipynb  # LSTM training dengan Keras
│   │   ├── from_scratch/              # LSTM layers dari scratch
│   │   │   ├── model.py               # Model utama LSTM
│   │   │   └── layers.py              # Layer-layer LSTM
│   │   ├── figures/                   # Grafik dan visualisasi
│   │   ├── lstm_keras_best.weights.h5 # Model weights terbaik
│   │   ├── lstm_keras_final.weights.h5# Model weights final
│   │   ├── lstm_saved_data.pkl        # Data test untuk validasi
│   │   └── lstm_experiment_results.json # Hasil eksperimen
│   └── data/                          # Dataset
│       └── nusax_sentiment/           # NusaX dataset
│           ├── train.csv              # Data training
│           ├── valid.csv              # Data validasi
│           └── test.csv               # Data testing
├── pyproject.toml                     # Konfigurasi project
├── uv.lock                           # Lock file dependencies
└── README.md
```

## Setup dan Instalasi

### Prerequisites
- uv (Python package manager)
- Jupyter Notebook atau JupyterLab

### 1. Clone Repository
```bash
git clone https://github.com/satriadhikara/tubes2_if3270_ml
cd tubes2_if3270_ml
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Aktivasi Virtual Environment
```bash
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

## Cara Menjalankan Program

### CNN (Convolutional Neural Network)
```bash
# Masuk ke direktori CNN
cd src/cnn/

# Training CNN dengan Keras
jupyter notebook cnn_keras_training.ipynb

# Untuk testing implementasi from scratch
jupyter notebook cnn_from_scratch_testing.ipynb
```

### LSTM (Long Short-Term Memory)
```bash
# Masuk ke direktori LSTM
cd src/lstm/

# Opsi 1: LSTM lengkap (training + from scratch testing)
jupyter notebook lstm.ipynb

# Opsi 2: Menjalankan secara terpisah
# Training LSTM dengan Keras
jupyter notebook lstm_keras_training.ipynb

# Testing implementasi from scratch
jupyter notebook LSTM.ipynb
```

## Dataset

- **CNN**: CIFAR-10 dataset (klasifikasi gambar 10 kelas)
- **RNN**: Time series atau sequential dataset (untuk prediksi/klasifikasi sekuensial)
- **LSTM**: NusaX-Sentiment dataset (sentiment analysis bahasa Indonesia)

Dataset LSTM akan didownload secara otomatis ke `src/data/nusax_sentiment/` saat menjalankan notebook pertama kali.

## Eksperimen dan Analisis

### LSTM Experiments 
- **Experiment 1**: Pengaruh jumlah layer LSTM (1, 2, 3 layers)
- **Experiment 2**: Pengaruh jumlah unit per layer (32, 64, 128 units)
- **Experiment 3**: Pengaruh arah LSTM (unidirectional vs bidirectional)

### CNN Experiments
- **Experiment 1**: Pengaruh jumlah filter pada layer konvolusi (32, 64, 128 filters)
- **Experiment 2**: Pengaruh ukuran kernel pada layer konvolusi (3x3, 5x5, 7x7)
- **Experiment 3**: Pengaruh jumlah layer konvolusi (2, 3, 4 layers)
- **Experiment 4**: Pengaruh dropout rate (0.2, 0.5, 0.7)

### RNN Experiments
- **Experiment 1**: Pengaruh jumlah layer RNN (1, 2, 3 layers)
- **Experiment 2**: Pengaruh jumlah unit per layer (32, 64, 128 units)
- **Experiment 3**: Pengaruh sequence length input (10, 20, 50 timesteps)
- **Experiment 4**: Pengaruh learning rate (0.001, 0.01, 0.1)

## Hasil dan Metrik

- **Metrik Evaluasi**: Macro F1-Score
- **Perbandingan**: Keras vs From Scratch implementation
- **Visualisasi**: Training curves, confusion matrix, performance comparison

## Pembagian Tugas Kelompok

| Anggota | NIM | Tugas |
|---------|-----|-------|
| Satriadhikara Panji Yudhistira | 13522125 | **CNN Implementation** <br> - `cnn_keras_training.ipynb` <br> - `cnn_from_scratch_testing.ipynb` <br> - CNN layers implementation <br> - CNN experiments & analysis |
| Mohammad Andhika Fadhillah | 13522128 | **RNN Implementation** <br> - `notebook/rnn_keras_training.ipynb` <br> - `notebook/rnn_from_scratch.ipynb` <br> - RNN layers implementation <br> - RNN experiments & analysis |
| Jonathan Emmanuel Saragih | 13522121 | **LSTM Implementation** <br> - `lstm.ipynb` (complete) <br> - `lstm_keras_training.ipynb` <br> - `from_scratch/model.py` & `layers.py` <br> - LSTM experiments & analysis |

### Detail Pembagian:

#### CNN 
- **File yang dikerjakan:**
  - `src/cnn/cnn_keras_training.ipynb`
  - `src/cnn/cnn_from_scratch_testing.ipynb`
  - `src/cnn/from_scratch/` (implementasi layers)
- **Tugas:**
  - Implementasi layer CNN dari scratch (Conv2D, MaxPooling2D, etc.)
  - Training model CNN dengan Keras untuk baseline
  - Eksperimen dengan berbagai konfigurasi CNN
  - Analisis performa dan perbandingan hasil

#### RNN 
- **File yang dikerjakan:**
  - `src/rnn/notebook/rnn_keras_training.ipynb`
  - `src/rnn/notebook/rnn_from_scratch.ipynb`
  - `src/rnn/from_scratch/` (implementasi layers)
- **Tugas:**
  - Implementasi layer RNN dari scratch (SimpleRNN, etc.)
  - Training model RNN dengan Keras untuk baseline
  - Eksperimen dengan berbagai konfigurasi RNN
  - Analisis performa dan perbandingan hasil

#### LSTM 
- **File yang dikerjakan:**
  - `src/lstm/lstm.ipynb` (implementasi lengkap)
  - `src/lstm/lstm_keras_training.ipynb`
  - `src/lstm/from_scratch/model.py`
  - `src/lstm/from_scratch/layers.py`
- **Tugas:**
  - Implementasi layer LSTM dari scratch (LSTM, Bidirectional, Embedding)
  - Training model LSTM dengan Keras untuk baseline
  - Eksperimen dengan jumlah layer, unit per layer, dan arah LSTM
  - Analisis performa menggunakan NusaX-Sentiment dataset

## Output Files

### LSTM Files (sudah ada):
- `lstm_keras_best.weights.h5` - Weights model terbaik dari Keras
- `lstm_keras_final.weights.h5` - Weights model final setelah extended training
- `lstm_saved_data.pkl` - Data test untuk validasi from scratch
- `lstm_experiment_results.json` - Hasil eksperimen dalam format JSON

### CNN Files (sudah ada):
- `cnn_keras.weights.h5` - Weights model Keras
- `cnn_keras_final.weights.h5` - Weights model final
- `best_model_config.json` - Konfigurasi model terbaik

### RNN Files (sudah ada):
- `rnn_keras_best.weights.h5` - Weights model terbaik
- `rnn_keras_final.weights.h5` - Weights model final
- `rnn_saved_data.pkl` - Data test untuk validasi
- `rnn_experiment_results.json` - Hasil eksperimen RNN

## Dependencies

```
tensorflow>=2.19.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
requests>=2.25.0
datasets>=2.0.0
```

---

**IF3270 Machine Learning - Institut Teknologi Bandung**
