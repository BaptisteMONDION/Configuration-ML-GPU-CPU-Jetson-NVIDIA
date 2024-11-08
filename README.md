# Test GPU Jetson Nano avec SVM

## Description

In this project, I tested the GPU capabilities of the **Jetson Nano** by training a **Support Vector Machine (SVM)** model on a randomly generated dataset. The main objective was to leverage **cuML**, a library from **RAPIDS AI**, to train the model on the GPU, enabling performance measurement and exploring the hardware acceleration provided by the Jetson Nano.

## Technologies Used

- **Jetson Nano** (with integrated GPU)
- **Python 3** (for developing the program)
- **cuML** (for SVM training on GPU)
- **scikit-learn** (for data generation and result comparison)
- **CUDA** (GPU acceleration)

## Installation

### 1) Updating Package List
The first step was to update the system packages on my **Jetson Nano**.

### 2) Installing pip
I faced an issue when trying to install **pip3** directly via **apt-get**, as the package was not available in the standard repositories. I solved this by manually downloading the `get-pip.py` script from another device and transferring it to the Jetson Nano. After transferring the file, I executed the script to install pip.

### 3) Installing cuML
Once pip was installed, I used it to install the **cuML** library (optimized for GPU acceleration) via **pip3**. This enabled me to leverage GPU acceleration for training the SVM model.

### 4) Generating Data and Training the SVM Model
After installing the necessary dependencies, I generated a **classification dataset** using **scikit-learn**. Then, I used **cuML** to train the **SVM model** on the GPU, which allowed me to test the use of GPU for standard machine learning tasks.

### 5) Running the Program
Once everything was set up, I ran the program to train the model and measure performance. I monitored **GPU usage** using the `tegrastats` command to ensure that the graphics processor was properly utilized.

## Issues Encountered

### 1) pip3 Installation Issues
I encountered issues installing pip3 directly via **apt-get** as the package could not be found in the standard repositories. I solved this by downloading and transferring the installation script manually from another device.

### 2) SSL Connection Issues to Download get-pip.py
When downloading `get-pip.py` on the Jetson Nano, I faced **SSL errors**. I bypassed this issue by using the `--no-check-certificate` option with `wget` to avoid SSL certificate verification.

### 3) cuML Compatibility
Installing **cuML** requires a specific version of **CUDA**. I ensured that the **CUDA version** on the Jetson Nano was compatible with the cuML library to ensure proper GPU training.

## Conclusion

This project allowed me to test the **GPU acceleration** capabilities of the **Jetson Nano** by running an **SVM model** on a classification dataset. The use of **cuML** resulted in a significant performance improvement over CPU training, and I was able to directly observe the impact of GPU acceleration on this machine learning project. Additionally, I resolved various technical issues related to package installations and **CUDA** configuration, ensuring that the system worked as intended.
