## Neural Network with CUDA  

This project implements a simple feedforward neural network in CUDA (`nn.cu`) and provides a Python script (`data.py`) for handling data processing for the MNiST Dataset.  

### Files  
- **`nn.cu`** – Implements a fully connected neural network with:  
  - Forward and backward propagation using CUDA kernels  
  - GeLU activation function  
  - Softmax for classification  
  - Cross-entropy loss  
  - Training on batch data  
- **`data.py`** – Prepares and saves training/testing MNiST dataset in a binary format for use in `nn.cu`.  

### Compilation & Execution  
Compile and run the CUDA program:  
```sh
nvcc -o nn nn.cu  
./nn
```
Generate data with python:
```sh
python3 data.py
```

### Requirements
- CUDA-enabled GPU
- NVIDIA CUDA Toolkit
- Python (for data.py)

### Notes
Ensure dataset files are correctly generated before running nn.cu.
