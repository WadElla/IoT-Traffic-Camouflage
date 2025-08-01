# IoT Traffic Obfuscation Framework

This repository contains a comprehensive implementation of an IoT traffic obfuscation framework. It enables traffic manipulation, performance evaluation, and learning-based analysis for assessing the privacy and efficiency trade-offs introduced by different obfuscation techniques.

The framework includes scripts for:

- Preprocessing raw PCAP traffic into structured datasets
- Applying packet-level obfuscation and deobfuscation transformations
- Measuring communication and system overheads (execution time, CPU, memory, bytes)
- Training and evaluating machine learning models on original and obfuscated datasets

---

## 📂 Directory Structure

```
IoT-Traffic-Camouflage/
│
├── preprocessing/            # Convert PCAP to CSV with traffic features
├── obfuscation/              # Obfuscation and corresponding deobfuscation scripts
├── performance/              # Evaluate system and communication overhead
├── model_training/           # Train ML and DNN models, cross-validation, adaptation
├── requirements.txt          # List of Python dependencies
├── replay_packet.py          # Replay PCAPs to regenerate traffic
└── README.md
```

---

## ⚙️ Installation

1. **Create a virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

2. **Install required Python packages**:

```bash
pip install -r requirements.txt
```

3. **Ensure system dependencies are available**:

- `tshark` (for parsing packets with PyShark)
- `libpcap` (required for live packet replay)

---

## 🚀 Usage Guide

### 1. Preprocessing

Convert `.pcap` files into structured CSV datasets for analysis:

```bash
python preprocessing/extract_features_from_pcap.py
```

This step extracts statistical features and organizes flow-level packet information per stream.

---

### 2. Obfuscation and Deobfuscation

Apply transformations to obscure traffic characteristics:

```bash
python obfuscation/obfuscate_padding_xor.py         # Apply Padding + XOR
python obfuscation/obfuscate_fragmentation.py   # Fragmentation to split packets
python obfuscation/obfuscate_delay_randomization.py    # Random delays
```

All techniques include deobfuscation logic except the delay_randomization:

```bash
python performance/deobfuscation/depadxor.py      # Reverses pad + XOR
```

Each script reads PCAP input, performs transformation, and saves a modified PCAP.

#### Replay Tool

After downloading PCAPs from public datasets, you can regenerate traffic using:

```bash
sudo python replay_packet.py
```

- Update `iface` in the script to match your network interface (e.g., `en0`, `eth0`)
- Set `input_pcap` to the file you wish to replay

This is useful for generating reproducible traffic before obfuscation.

---

### 3. Performance Evaluation

Measure the overhead introduced by each obfuscation method:

```bash
python performance/obfuscation/fragmentation.py
```

Metrics captured:

- ⏱ Execution Time: per-packet transformation delay
- 🧠 Memory Usage: runtime memory consumption
- ⚙️ CPU Load: processor usage during obfuscation
- 📦 Extra Bytes: communication overhead introduced

---

### 4. Model Training and Evaluation

#### Training

We train five models to classify IoT device types based on traffic features:

- Decision Tree (DT)
- Random Forest (RF)
- Gradient Boosting Machine (GBM)
- K-Nearest Neighbor (kNN)
- Deep Neural Network (DNN)

Use the following scripts:

```bash
python model_training/train_models.py              # Classical ML models
python model_training/neural_model.py              # Initial DNN training
```

#### Fine-Tuning & Incremental Learning

After training on normal traffic, the DNN is further refined on obfuscated traffic:

```bash
python model_training/neural_fine_tune.py          # Fine-tune DNN
python model_training/neural_incremental.py        # Incremental training
```

These steps assess the robustness of the framework against adaptive adversaries by retraining models on obfuscated traffic. Fine-tuning adapts the DNN to a specific obfuscation technique, while incremental training (also known as online learning or continual learning) simulates real-world scenarios where models progressively incorporate new obfuscated data over time without retraining from scratch. These methods were evaluated across all three datasets using a retrained deep neural network to examine how well obfuscation methods continue to prevent accurate inference.

#### Cross-Validation

To assess robustness across splits and datasets:

```bash
python model_training/cross_val_final.py
python model_training/cross_val_neural_net_final.py
```

Metrics used for evaluation:

- Accuracy
- Precision
- Recall
- F1 Score

These steps provide insight into how each obfuscation technique impacts model performance.

---

## 📟 Dataset Access

This framework has been evaluated using three publicly available IoT datasets. These datasets can be accessed through the official sources cited in the corresponding research literature:

- **IoT-AD Dataset**: H. Zahan, M. W. Al Azad, I. Ali, and S. Mastorakis, “IoT-AD: A framework to detect anomalies among interconnected IoT devices,” *IEEE Internet of Things Journal*, vol. 11, no. 1, pp. 478–489, 2023.

- **IoT Sentinel Dataset**: M. Miettinen, S. Marchal, I. Hafeez, N. Asokan, A.-R. Sadeghi, and S. Tarkoma, “IoT Sentinel: Automated device-type identification for security enforcement in IoT,” in *Proc. of IEEE ICDCS*, pp. 2177–2184, 2017.

- **UNSW Dataset**: A. Sivanathan et al., “Classifying IoT devices in smart environments using network traffic characteristics,” *IEEE Transactions on Mobile Computing*, vol. 18, no. 8, pp. 1745–1759, 2018.

Please consult the original papers to download the datasets from their official repositories.

You may also use the `replay_packet.py` script to regenerate traffic from PCAP files before applying the obfuscation steps.

---

## ✅ Quick Start

```bash
# Environment setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Preprocess PCAP to CSV
python preprocessing/extract_features_from_pcap.py

# Obfuscate traffic
python obfuscation/pad_xor.py

# Evaluate resource and network performance, e.g
python performance/obfuscation/fragmentation.py

# Train ML or DNN models
python model_training/neural_model.py

```

---

## 📦 Requirements

All required Python packages are listed in `requirements.txt`.
To install them:

```bash
pip install -r requirements.txt
```

---

