# Project: Rockfall-Forecasting-using-Ensemble-Deep-Learningand-Temporal-Gradient-Based-Explanations


This repository contains the code for Rockfall-Forecasting-using-Ensemble-Deep-Learningand-Temporal-Gradient-Based-Explanations.

## Data

- **Météo France data**: Available on Zenodo [https://doi.org/10.5281/zenodo.16978847](https://doi.org/10.5281/zenodo.16985288) :

- **Rockfall catalogue**: Real data is restricted. To request access, contact Amitrano David <david.amitrano@univ-grenoble-alpes.fr> and Agnès Helmstetter <agnes.helmstetter@univ-grenoble-alpes.fr>.

- **Fake catalogue**: A mock dataset will be included in `fake_catalogue/` for testing purposes.


## Script Description                                                                      

- `Train_models.py`: Trains individual deep learning models on the processed dataset  using cross-validation.

- `Hard_voting.py`: Combines predictions from multiple models using hard voting to produce an ensemble forecast. 

- `GradCAM_vote.py`: Applies Grad-CAM explanations on the ensemble predictions to identify temporal features influencing rockfall forecasts. 


## Usage

1. Modify the meteorological and sismic data paths in each script to point to your data paths. 

2. Run the scripts in the following order:  
    ```bash
    python3 Train_models.py
    python3 Hard_voting.py
    python3 GradCAM_vote.py
