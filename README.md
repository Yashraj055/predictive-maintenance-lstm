* Problem Statement

In real-world industrial systems (motors, pumps, engines), failures do not occur suddenly.
Instead, system efficiency degrades gradually over time, and this degradation must be detected early to prevent breakdowns and reduce maintenance costs.

This project focuses on unsupervised anomaly detection using sensor data from the NASA C-MAPSS Turbofan Engine Dataset.


* Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

Multiple engines

Multivariate sensor readings

Run-to-failure simulations

Realistic noise and operating conditions

Dataset is not included in the repo due to size.
Place raw files inside data/raw/.

* Project Structure

predictive-maintenance-lstm/
│
├── src/
│   ├── train.py            # Model training
│   ├── evaluate.py         # Anomaly detection & evaluation
│   ├── model.py            # LSTM Autoencoder architecture
│   ├── preprocessing.py   # Scaling & feature processing
│   ├── data_loader.py     # Dataset loading
│   ├── plots.py            # Visualization utilities
│   ├── visualize.py        # Plot helpers
│   ├── health_index.py     # Health index computation
│   └── rul_estimation.py   # RUL extension logic
│
├── data/
│   ├── raw/                # Raw C-MAPSS files (ignored in git)
│   └── processed/          # Preprocessed data (ignored)
│
├── models/                 # Saved trained models (ignored)
├── results/                # Evaluation outputs (ignored)
│
├── requirements.txt
├── README.md
└── .gitignore

* Author

Yash Raj Singh
AI / ML Enthusiast
📌 Focus: Predictive Maintenance, Deep Learning, Time-Series Analysis
