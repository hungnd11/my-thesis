# README

- This repository contains work about my master's thesis: "An evaluation of passive-crowdsourcing methods for large-scale construction of WiFi radio map".

- Although it was initially intended for passive-crowdsourcing for WiFi radio map construction, this repository contains many useful code for indoor location based on WiFi fingerprinting and inertial sensing.

## Setup

- Creating environment and install necessary libraries
  - In this work, we use `python3.10` and the built-in `venv` for creating a virtual environment. Any environment management tools (i.e. `conda`, `mamba`, `poetry`) are applicable.
  
  ```bash
  # Create a virtual env
  python3.10 -m venv venv
  source venv/bin/activate
  # Install necessary libraries
  python -m pip install -r requirements.txt
  ```

- Downloading the Microsoft Indoor Location 2.0 dataset and extracting into the `data` directory.
  - The dataset can be downloaded from [Kaggle: Indoor Location and Navigation](https://www.kaggle.com/competitions/indoor-location-navigation/data)

## Repository Structure

- `code` is the root directory for all code in the project, which is organized into subdirectories.
  - `experiments`: Contain code for experiments with Zee passive crowdsourcing and motion models.
  - `indoor-location-competition-20`: Sample code for the `Indoor Location and Navigation` contest, which was cloned from this repository: [location-competition/indoor-location-competition-20](https://github.com/location-competition/indoor-location-competition-20)
  - `notebooks`: Various notebooks created while I was exploring the topic. Most of them are trials (and errors :(). This directory contains notebooks implementing WiFi fingerprinting indoor location based on kNN, motion models, passive crowdsourcing (Zee, [PiLoc](https://ieeexplore.ieee.org/document/6846748), [LiFS](https://dl.acm.org/doi/10.1145/2348543.2348578)).
  - `py_indoor_loc`: A library containing many useful code for indoor location based on WiFi fingerprinting and inertial sensing. Notable packages are `knn` for implementation of WiFi fingerprinting indoor location using kNN, `pdr` for implementation of motion models, `zee.py` for Zee implementation.
- `figures`: This directory contains some figures I used in my thesis.
- `README`: This file contains instruction and description.

## Experiments

- For running experiment with [Zee: Zero-effort crowdsourcing for indoor localization](https://www.cs.princeton.edu/courses/archive/spring17/cos598A/papers/zee.pdf), follow the notebook: [Zee](./code/experiments/zee.ipynb)

- For running experiments with motion models, follow the notebook [Motion Models](./code/experiments/motion_models.ipynb) and [Motion Model Evaluation](./code/experiments/motion_models_result_analysis.ipynb).
