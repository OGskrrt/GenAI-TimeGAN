# TimeGAN 

## **Note on Training Constraints**  
Due to hardware limitations, training this GAN (Generator + Discriminator) for more epochs or with larger batch sizes is not feasible here. As a result, the model does not fully achieve maximum accuracy or fidelity. This code, however, demonstrates how to generate synthetic HR-like signals and train a basic GAN under constrained conditions.

## Project Overview

This project demonstrates:
1. **Reading HR signals** from WFDB files in the [Stress Recognition in Automobile Drivers dataset](https://www.kaggle.com/datasets/bjoernjostein/stress-recognition-in-automobile-drivers).  
2. **Loading data** into a `BatchLoader` that normalizes, detrends, and windows the HR signals for GAN training.  
3. **Building a GAN** with an LSTM-based **Generator** and **Discriminator**, then combining them into a single GAN model.  
4. **Training** the GAN over multiple epochs, comparing generated (fake) HR windows to real HR windows.  
5. **Generating new HR signals** after training and plotting these generated signals.  
6. **Optional** display of the first 600 seconds of actual HR data for each driver.

> **Dataset Note**  
> The same dataset (Stress Recognition in Automobile Drivers) was also used in the accompanying LSTM Autoencoder project, where anomaly detection was performed on HR signals. Here, we adapt a GAN approach to synthesize new HR-like data from this very dataset.

## Main Steps

1. **BatchLoader**  
   - Reads each `.dat` file’s HR channel and trims it to a maximum duration (`max_duration_hours`).  
   - Interpolates missing points, normalizes the range [0, 1], applies `detrend`, and creates windows of size `window_size`.  
   - Returns training batches of shape `(batch_size, window_size, 1)`.

2. **GAN Components**  
   - **Generator** (LSTM-based): Takes random noise `(window_size, 1)` as input and outputs a synthetic signal of the same shape.  
   - **Discriminator** (LSTM-based): Predicts whether a given `(window_size, 1)` signal is real or fake.

3. **Training**  
   - **Discriminator** trains on real vs. fake windows each iteration.  
   - **Generator** is trained via the **GAN** model (frozen discriminator weights) to fool the Discriminator.

4. **Generation & Plotting**  
   - After training, we generate synthetic signals by sampling random noise and feeding it into the Generator.  
   - We also plot the actual HR data (first 600 seconds) for each driver in the dataset to visualize real signals.

## Usage

- **Requirements**: `wfdb`, `numpy`, `pandas`, `matplotlib`, `tensorflow`, `scipy`, `sklearn`.  
- **Paths**: Point `data_dir` to your directory of `.dat` files.  
- **Parameters**: Adjust `batch_size`, `window_size`, `max_duration_hours`, and `epochs` as needed.  
- **Run**:  
  1. Install WFDB: `!pip install wfdb` (if in Jupyter/Kaggle).  
  2. Execute each cell in sequence.

## Dataset Link

- [Kaggle: Stress Recognition in Automobile Drivers](https://www.kaggle.com/datasets/bjoernjostein/stress-recognition-in-automobile-drivers)  
- Same data is used in the LSTM Autoencoder project for anomaly detection on HR signals.
