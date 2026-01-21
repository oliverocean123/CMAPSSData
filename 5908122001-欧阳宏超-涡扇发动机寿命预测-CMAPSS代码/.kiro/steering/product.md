# Product Overview

## Remaining Useful Life (RUL) Prediction System

This project implements a comprehensive machine learning system for predicting the Remaining Useful Life (RUL) of aircraft engines using the NASA Turbofan Engine Degradation Simulation Dataset.

### Purpose
- Predict how many operational cycles an aircraft engine can continue to operate before failure
- Support predictive maintenance strategies to prevent unexpected failures
- Analyze engine degradation patterns across different operational conditions and fault modes

### Dataset Information
The system works with four datasets (FD001-FD004) representing different scenarios:
- **FD001**: Single operating condition, single fault mode (HPC Degradation)
- **FD002**: Six operating conditions, single fault mode (HPC Degradation) 
- **FD003**: Single operating condition, two fault modes (HPC + Fan Degradation)
- **FD004**: Six operating conditions, two fault modes (HPC + Fan Degradation)

### Key Features
- Multi-algorithm approach combining traditional ML and deep learning models
- Comprehensive feature engineering with statistical and time-series features
- Performance evaluation using industry-standard PHM (Prognostics and Health Management) scoring
- Visualization and analysis tools for model comparison and results interpretation

### Target Users
- Aerospace engineers working on predictive maintenance
- Data scientists developing prognostics models
- Researchers in the field of condition monitoring and fault prediction