from gradcam_Inception.utils import  process_rockfall_data_multi, charge,  preprocess_segments_test
import pandas as pd 
import numpy as np 
import time
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score,accuracy_score , balanced_accuracy_score
from glob import glob
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
from sklearn.utils.class_weight import compute_sample_weight
from gradcam_Inception.InceptionTime_attention import InceptionTimeModel 
import pickle

# Apply HARD voting 
vote = 'HARD' 

# Load rain and rockfall data of the blind test year 2024 
df_sismo= pd.read_csv('.data/ev_sismo2_2024.csv', delimiter=',')
df_RR1= pd.read_csv('.data/1.0_RR1_2024.csv', delimiter=',')

# Compute water charge for the blind test year 2024
df_precip = pd.read_csv('.data/data_meteo_2024.csv', delimiter=',')
df_precip['AAAAMMJJHH'] = df_RR1['AAAAMMJJHH'].values
df_precip['AAAAMMJJHH'] = pd.to_datetime(df_precip['AAAAMMJJHH'])
lamb=0.2
df_precip['H'] = charge(df_precip,lamb) 


with open("scaler_rr1.pkl", "rb") as f:
    scaler_rr1 = pickle.load(f)

with open("scaler_h.pkl", "rb") as f:
    scaler_h = pickle.load(f)


# Select computation device: use GPU ("cuda") if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for hp in [1,3] : 

    df_1 = process_rockfall_data_multi(df_RR1, df_sismo, x_days=hp, col_name='RR1',  concat_with_R=True, jour_pluie=14)
    df_1['sample_id'] = df_1.index

    df_2 = process_rockfall_data_multi(df_precip, df_sismo, x_days=hp, col_name='H', concat_with_R=False, jour_pluie=14)
    df_2['sample_id'] = df_2.index

    df = pd.merge(df_2, df_1, on='sample_id')
    df = df.drop(columns=['sample_id'])

    df.set_index('AAAAMMJJHH', inplace=True)

    X_rain = df[[f"RR1_hour_{i}" for i in range(336)]].values  # Colmuns of Rain (windows of 14*24)
    X_charge = df[[f"H_hour_{i}" for i in range(336)]].values  # Colmuns of water charge (windows of 14*24)

    x_rain_scaled, _ = preprocess_segments_test(X_rain, scaler_rain)
    x_charge_scaled, _ = preprocess_segments_test(X_charge, scaler_charge)

    y = df['rockfall_target'].values
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Conversion to PyTorch tensors
    X_tensor_charge = torch.FloatTensor(x_charge_scaled).view(-1, 1, x_charge_scaled.shape[1]).to(device)
    X_tensor_rain = torch.FloatTensor(x_rain_scaled).view(-1, 1, x_rain_scaled.shape[1]).to(device)

    y_tensor = torch.LongTensor(y_encoded)

    # Initialize lists
    all_preds = []
    all_probs=[]
    all_weighted_probs=[]
    w1_all=[]
    for test_year in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
        model_paths = glob(f"testing_models/HP_{hp}/test_year_{test_year}/val_*.pt")
        if not model_paths:
            print(f"No model found for {test_year}")

        for model_path in model_paths:
            metric_path= model_path[:-11]+"test_metrics_val_year"+model_path[-7:-3]+".json"
            # Model configuration 
            model = InceptionTimeModel(
                in_channels_1=1,  # 1st time series (1 channel)
                in_channels_2=1,  # 2nd time series (1 channel)
                num_classes=2, merge_mode='concat' )

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            # Loadweights into model
            model.load_state_dict(checkpoint)
            model = model.to(device)
            model.eval()

            with torch.no_grad():
                outputs = model( X_tensor_rain, X_tensor_charge )
                probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Get probabilities
                all_probs.append(probs) 
            
            preds = outputs.argmax(dim=1).cpu().numpy() # Get predictions 
            all_preds.append(preds)

    results = []

    #----------------------------Majority HARD vote --------------------------------
    if vote =="HARD" : 
        all_preds_array = np.stack(all_preds)   # de shape (90, 351) 
        #sum of votes for class 1 across models 
        votes_for_class_1 = np.sum(all_preds_array == 1, axis=0) 
        #more than half the models predict 1
        final_preds_majority = (votes_for_class_1 > 90 / 2).astype(int)
        print(final_preds_majority)

    # Metrics
    accuracy = accuracy_score(y_tensor, final_preds_majority)
    weights = compute_sample_weight("balanced",  y_tensor)
    acc_weighted = accuracy_score(y_tensor, final_preds_majority, sample_weight=weights)

    precision = precision_score(y_tensor, final_preds_majority, average="weighted",zero_division=0)
    recall = recall_score(y_tensor, final_preds_majority, average="weighted",zero_division=0)
    f1 = f1_score(y_tensor, final_preds_majority, average="weighted")
    conf_matrix = confusion_matrix(y_tensor, final_preds_majority)

    # Per-class metrics
    precision_per_class = precision_score(y_tensor, final_preds_majority, average=None,zero_division=0)
    recall_per_class = recall_score(y_tensor, final_preds_majority, average=None,zero_division=0)
    f1_per_class=f1_score(y_tensor, final_preds_majority, average=None)

    results.append({
    "f1_score": f1,
    "precision": precision,
    "recall": recall,
    "accuracy": accuracy,
    "weighted-accuracy" : acc_weighted,
    'precision_class_0': precision_per_class[0],
    'recall_class_0': recall_per_class[0],
    'precision_class_1': precision_per_class[1],
    'recall_class_1': recall_per_class[1],
    'f1_class_0': f1_per_class[0],
    'f1_class_1': f1_per_class[1],
    'confusion_matrix': conf_matrix.tolist()
    })

    print(f"Acuuracy={accuracy:.3f} | Precision={precision:.3f} | Recall={recall:.3f} | F1={f1:.3f} | ")
    print(results)

    
    # Create the folder 'vote' if it doesn't exist to save vote results 
    os.makedirs("testing_models/HP_{hp}/vote", exist_ok=True)
    with open(os.path.join(f"testing_models/HP_{hp}/vote", f"results_vote_{vote}.json" ), "w") as f:
        json.dump(results, f, indent=4)

