# NGC Models: Command Reference Guide

This document provides a comprehensive list of commands used to simulate, compare, and run the Neural Generative Coding (NGC) models in our experiments.

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [Model Training Commands](#3-model-training-commands)
4. [Model Evaluation Commands](#4-model-evaluation-commands)
5. [Visualization Commands](#5-visualization-commands)
6. [Non-Gaussian Distribution Extensions](#6-non-gaussian-distribution-extensions)
7. [Comparison Experiments](#7-comparison-experiments)

## 1. Environment Setup

### Check GPU Availability
```bash
nvidia-smi
```

### Navigate to Project Directory
```bash
cd D:/NAC_Project/First_simulation/NAC-Experiments/Model_Comparison/Stable_Version/
```

### List Available Models
```bash
dir walkthroughs/walkthrough1/
```

## 2. Data Preparation

The MNIST dataset is automatically downloaded and prepared by the simulation scripts. The data is stored in:
```
../../../../data/mnist/
```

## 3. Model Training Commands

### Train GNCN-PDH Model
```bash
python sim_train.py --config=gncn_pdh/fit.cfg --gpu_id=0
```

### Train GNCN-t1 Model
```bash
python sim_train.py --config=gncn_t1/fit.cfg --gpu_id=0
```

### Train GNCN-t1-Sigma Model
```bash
python sim_train.py --config=gncn_t1_sigma/fit.cfg --gpu_id=0
```

### Train Non-Gaussian GNCN-t1 Model (Student's t-distribution)
```bash
python sim_train.py --config=gncn_t1_student/fit.cfg --gpu_id=0
```

### Train Non-Gaussian GNCN-t1 Model (Laplace distribution)
```bash
python sim_train.py --config=gncn_t1_laplace/fit.cfg --gpu_id=0
```

### Train Non-Gaussian GNCN-t1 Model (Mixture of Gaussians)
```bash
python sim_train.py --config=gncn_t1_mog/fit.cfg --gpu_id=0
```

## 4. Model Evaluation Commands

### Analyze GNCN-PDH Model
```bash
python sim_analyze.py --config=gncn_pdh/analyze.cfg --gpu_id=0
```

### Analyze GNCN-t1 Model
```bash
python sim_analyze.py --config=gncn_t1/analyze.cfg --gpu_id=0
```

### Analyze GNCN-t1-Sigma Model
```bash
python sim_analyze.py --config=gncn_t1_sigma/analyze.cfg --gpu_id=0
```

### Analyze Non-Gaussian GNCN-t1 Model (Student's t-distribution)
```bash
python sim_analyze.py --config=gncn_t1_student/analyze.cfg --gpu_id=0
```

### Analyze Non-Gaussian GNCN-t1 Model (Laplace distribution)
```bash
python sim_analyze.py --config=gncn_t1_laplace/analyze.cfg --gpu_id=0
```

### Analyze Non-Gaussian GNCN-t1 Model (Mixture of Gaussians)
```bash
python sim_analyze.py --config=gncn_t1_mog/analyze.cfg --gpu_id=0
```

## 5. Visualization Commands

### Generate Reconstructions
```bash
python sim_visualize.py --config=gncn_pdh/visualize.cfg --mode=reconstruction --gpu_id=0
python sim_visualize.py --config=gncn_t1/visualize.cfg --mode=reconstruction --gpu_id=0
python sim_visualize.py --config=gncn_t1_sigma/visualize.cfg --mode=reconstruction --gpu_id=0
python sim_visualize.py --config=gncn_t1_student/visualize.cfg --mode=reconstruction --gpu_id=0
python sim_visualize.py --config=gncn_t1_laplace/visualize.cfg --mode=reconstruction --gpu_id=0
python sim_visualize.py --config=gncn_t1_mog/visualize.cfg --mode=reconstruction --gpu_id=0
```

### Generate Latent Space Visualizations
```bash
python sim_visualize.py --config=gncn_pdh/visualize.cfg --mode=latent --gpu_id=0
python sim_visualize.py --config=gncn_t1/visualize.cfg --mode=latent --gpu_id=0
python sim_visualize.py --config=gncn_t1_sigma/visualize.cfg --mode=latent --gpu_id=0
python sim_visualize.py --config=gncn_t1_student/visualize.cfg --mode=latent --gpu_id=0
python sim_visualize.py --config=gncn_t1_laplace/visualize.cfg --mode=latent --gpu_id=0
python sim_visualize.py --config=gncn_t1_mog/visualize.cfg --mode=latent --gpu_id=0
```

### Generate Samples
```bash
python sim_visualize.py --config=gncn_pdh/visualize.cfg --mode=sample --gpu_id=0
python sim_visualize.py --config=gncn_t1/visualize.cfg --mode=sample --gpu_id=0
python sim_visualize.py --config=gncn_t1_sigma/visualize.cfg --mode=sample --gpu_id=0
python sim_visualize.py --config=gncn_t1_student/visualize.cfg --mode=sample --gpu_id=0
python sim_visualize.py --config=gncn_t1_laplace/visualize.cfg --mode=sample --gpu_id=0
python sim_visualize.py --config=gncn_t1_mog/visualize.cfg --mode=sample --gpu_id=0
```

### Plot Training Curves
```bash
python plot_training_curves.py --models=gncn_pdh,gncn_t1,gncn_t1_sigma,gncn_t1_student,gncn_t1_laplace,gncn_t1_mog
```

## 6. Non-Gaussian Distribution Extensions

### Create Configuration Files for Non-Gaussian Models

To create a new configuration file for a Student's t-distribution model:

```bash
cp gncn_t1/fit.cfg gncn_t1_student/fit.cfg
```

Then edit the file to change the distribution:

```bash
# Edit the distribution parameter in the configuration file
sed -i 's/distribution = gaussian/distribution = student_t/g' gncn_t1_student/fit.cfg
sed -i 's/# df = 3/df = 3/g' gncn_t1_student/fit.cfg
```

Similarly for Laplace distribution:

```bash
cp gncn_t1/fit.cfg gncn_t1_laplace/fit.cfg
sed -i 's/distribution = gaussian/distribution = laplace/g' gncn_t1_laplace/fit.cfg
```

And for Mixture of Gaussians:

```bash
cp gncn_t1/fit.cfg gncn_t1_mog/fit.cfg
sed -i 's/distribution = gaussian/distribution = mog/g' gncn_t1_mog/fit.cfg
sed -i 's/# n_components = 2/n_components = 3/g' gncn_t1_mog/fit.cfg
```

## 7. Comparison Experiments

### Compare All Models on MNIST Classification
```bash
python compare_models.py --task=classification --models=gncn_pdh,gncn_t1,gncn_t1_sigma,gncn_t1_student,gncn_t1_laplace,gncn_t1_mog
```

### Compare All Models on Reconstruction Quality
```bash
python compare_models.py --task=reconstruction --models=gncn_pdh,gncn_t1,gncn_t1_sigma,gncn_t1_student,gncn_t1_laplace,gncn_t1_mog
```

### Compare All Models on Generative Quality
```bash
python compare_models.py --task=generation --models=gncn_pdh,gncn_t1,gncn_t1_sigma,gncn_t1_student,gncn_t1_laplace,gncn_t1_mog
```

### Compare All Models on Robustness to Noise
```bash
python compare_models.py --task=robustness --noise_levels=0.1,0.2,0.3,0.4,0.5 --models=gncn_pdh,gncn_t1,gncn_t1_sigma,gncn_t1_student,gncn_t1_laplace,gncn_t1_mog
```

### Compare All Models on Out-of-Distribution Detection
```bash
python compare_models.py --task=ood_detection --ood_dataset=fashion_mnist --models=gncn_pdh,gncn_t1,gncn_t1_sigma,gncn_t1_student,gncn_t1_laplace,gncn_t1_mog
```

### Compare Training Time
```bash
python compare_models.py --task=training_time --models=gncn_pdh,gncn_t1,gncn_t1_sigma,gncn_t1_student,gncn_t1_laplace,gncn_t1_mog
```

### Compare Memory Usage
```bash
python compare_models.py --task=memory_usage --models=gncn_pdh,gncn_t1,gncn_t1_sigma,gncn_t1_student,gncn_t1_laplace,gncn_t1_mog
```

## Additional Useful Commands

### Check Training Progress
```bash
tail -f logs/gncn_t1_student_train.log
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Create Backup of Trained Models
```bash
mkdir -p backups/$(date +%Y%m%d)
cp -r checkpoints/* backups/$(date +%Y%m%d)/
```

### Run Multiple Models in Sequence
```bash
for model in gncn_pdh gncn_t1 gncn_t1_sigma gncn_t1_student gncn_t1_laplace gncn_t1_mog; do
    python sim_train.py --config=${model}/fit.cfg --gpu_id=0
    python sim_analyze.py --config=${model}/analyze.cfg --gpu_id=0
done
```

### Run Multiple Models in Parallel (if multiple GPUs are available)
```bash
python sim_train.py --config=gncn_pdh/fit.cfg --gpu_id=0 &
python sim_train.py --config=gncn_t1/fit.cfg --gpu_id=1 &
python sim_train.py --config=gncn_t1_sigma/fit.cfg --gpu_id=2 &
python sim_train.py --config=gncn_t1_student/fit.cfg --gpu_id=3 &
```
