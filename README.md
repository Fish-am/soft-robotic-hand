# Soft Robotic Hand RL Training

This repository contains code for training a soft robotic hand using reinforcement learning with PyBullet simulation.

## Quick Start in Google Colab

1. Create a new Colab notebook
2. Clone this repository:
```python
!git clone https://github.com/YOUR_USERNAME/soft-robotic-hand.git
%cd soft-robotic-hand
!pip install -r requirements.txt
```

3. Run the training:
```python
!python train.py
```

4. Monitor training with TensorBoard:
```python
%load_ext tensorboard
%tensorboard --logdir logs
```

## Project Structure

- `train.py`: Main training script using SAC algorithm
- `test.py`: Script to evaluate trained models
- `env/hand_env.py`: Custom Gymnasium environment for the soft robotic hand
- `urdf/soft_hand.urdf`: URDF description of the soft robotic hand

## Features

- Pressure-based control for soft fingers
- FEM-based soft body simulation
- Domain randomization for sim-to-real transfer
- Support for different grasp types (whole hand, tripod, pinch)
- Real-world calibration parameters

## Training Parameters

Key parameters that can be modified in `train.py`:
- Total timesteps: 2,000,000
- Learning rate: 0.0003
- Batch size: 256
- Buffer size: 1,000,000

Environment parameters in `env/hand_env.py`:
- Success threshold: 2cm
- Maximum pressure: 100 PSI
- Pressure rate limit: 20 PSI/s 