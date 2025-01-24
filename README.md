# TimeGAN
> **Note on Training Constraints**  
> Due to hardware limitations, training the GAN (Generator + Discriminator) for more epochs or with larger batch sizes is not feasible here. As a result, the model does not fully realize its potential accuracy or fidelity. This code nevertheless serves as a proof of concept for generating synthetic HR-like signals and demonstrating how to train a basic GAN in a constrained environment.

## Project Overview

This project demonstrates:
1. **Reading HR signals** from WFDB files in the [Stress Recognition in Automobile Drivers dataset](#dataset-link).
2. **Loading data** into a `BatchLoader` that normalizes, detrends, and windows the HR signals for GAN training.
3. **Building a GAN** with an LSTM-based **Generator** and **Discriminator**, compiling them into a combined model.
4. **Training** the GAN for a specified number of epochs (`epochs`), using batched real data vs. generated data.
5. **Generating new HR signals** after training and plotting these generated signals.
6. **Optional** plotting of the first 600 seconds of HR data for each driver in the dataset.

## Key Scripts and Steps

1. **BatchLoader**  
   - Splits each `.dat` file’s HR channel into windows of length `window_size`.
   - Normalizes (0–1 scale), detrends, and returns training batches.

2. **GAN Components**  
   - **Generator**: Takes random noise of shape `(window_size, 1)` and produces a synthetic signal of the same shape.  
   - **Discriminator**: Learns to distinguish real HR windows from synthetic ones.

3. **Training**  
   - **Discriminator** is trained on real and fake data in each iteration.  
   - **Generator** is trained to fool the Discriminator (WGAN-like approach with linear outputs and negative labels for the Generator).

4. **Generation & Plotting**  
   - The code finally generates synthetic HR signals and plots them in a single column.  
   - Plots each driver’s first 600 seconds of actual HR data in a separate figure.

## Usage

1. **Environment**: Ensure you have the required libraries (`wfdb`, `numpy`, `pandas`, `matplotlib`, `tensorflow`, `scipy`, `sklearn`).  
2. **Data Directory**: Update `data_dir` to point to your `.dat` files if needed.  
3. **Parameters**: Adjust `batch_size`, `window_size`, `max_duration_hours`, and `epochs` to suit your hardware constraints.  
4. **Execution**:  
   - Install WFDB: `!pip install wfdb` (in a Jupyter or Kaggle environment).  
   - Run the notebook cells sequentially.  

## Dataset Link

- The dataset is located [here on Kaggle](https://www.kaggle.com/datasets/bjoernjostein/stress-recognition-in-automobile-drivers).  
- It contains multiparameter driving data from healthy volunteers and is intended for research on driver stress detection.

