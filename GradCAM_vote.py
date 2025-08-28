from gradcam_Inception.InceptionTime_attention import InceptionTimeModel 
from gradcam_Inception.gradcam import MultiBranchGradCAM_2 , plot_gradcam_heatmaps_averaged, plot_mean_sample_per_modality
from gradcam_Inception.utils import  process_rockfall_data_multi, charge, preprocess_segments
import pandas as pd 
import numpy as np 
import time
import torch
from sklearn.preprocessing import LabelEncoder
from glob import glob
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import os 
import cv2



plt.rcParams.update({'axes.labelsize': 13,    # Taille du texte des labels des axes
                     'axes.titlesize': 12,   # Taille du texte du titre des axes
                     'xtick.labelsize': 14,  # Taille du texte des labels des ticks X
                     'ytick.labelsize': 14,  # Taille du texte des labels des ticks Y
                     'font.size': 13,        # Taille générale du texte
                     'legend.fontsize': 12})


modalities = ["Rain", "Charge"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for hp in [1,3,7]:
    # Path of folder where to save gradcam results
    folder= f'gradcam_all_models/HP_{hp}'
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    df_sismo= pd.read_csv('.data/ev_sismo2_2024.csv', delimiter=',')
    df_RR1= pd.read_csv('.data/1.0_RR1_2024.csv', delimiter=',')
    df_precip = pd.read_csv('.data/data_meteo_2024.csv', delimiter=',')
    df_precip['AAAAMMJJHH'] = df_RR1['AAAAMMJJHH'].values
    df_precip['AAAAMMJJHH'] = pd.to_datetime(df_precip['AAAAMMJJHH'])
    lamb=0.2
    df_precip['H'] = charge(df_precip,lamb) 

    df_1 = process_rockfall_data_multi(df_RR1, df_sismo, x_days=hp, col_name='RR1',  concat_with_R=True, jour_pluie=14)
    df_1['sample_id'] = df_1.index
    df_2 = process_rockfall_data_multi(df_precip, df_sismo, x_days=hp, col_name='H', concat_with_R=False, jour_pluie=14)
    df_2['sample_id'] = df_2.index
    df = pd.merge(df_2, df_1, on='sample_id')
    df = df.drop(columns=['sample_id'])
    df.set_index('AAAAMMJJHH', inplace=True)
    X_rain = df[[f"RR1_hour_{i}" for i in range(336)]].values  
    X_charge = df[[f"H_hour_{i}" for i in range(336)]].values 
    x_rain_scaled, _ = preprocess_segments(X_rain)
    x_charge_scaled, _ = preprocess_segments(X_charge)

    y = df['rockfall_target'].values
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Conversion to PyTorch tensors
    X_tensor_charge = torch.FloatTensor(x_charge_scaled).view(-1, 1, x_charge_scaled.shape[1]).to(device)
    X_tensor_rain = torch.FloatTensor(x_rain_scaled).view(-1, 1, x_rain_scaled.shape[1]).to(device)
    y_tensor = torch.LongTensor(y_encoded)

    # Original signal's tensor
    X_test_tensor_orig_charge = torch.FloatTensor(X_charge).view(-1, 1, X_charge.shape[1]).to(device) #tesnor du signal original
    X_test_tensor_orig_rain = torch.FloatTensor(X_rain).view(-1, 1, X_rain.shape[1]).to(device)


    for classe in [1]:
        for pred in [1]:
            if classe == pred :
                check = True 
            else :
                check = False

            gradcam_dict = defaultdict(lambda: defaultdict(list))
            samples_dict = defaultdict(lambda: defaultdict(list))
            # Initialize lists
            all_preds = []
            for test_year in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
                print(f"\nGradCam for models of {test_year}, pred = {pred} / class = {classe}")
                model_paths = glob(f"testing_models/HP_{hp}/test_year_{test_year}/val_*.pt")
                if not model_paths:
                    print(f"No model found for {test_year}")

                for model_path in model_paths:
                    metric_path= model_path[:-11]+"test_metrics_val_year"+model_path[-7:-3]+".json"
                    # Model configuration 
                    model = InceptionTimeModel(
                        in_channels_1=1,  
                        in_channels_2=1,  
                        num_classes=2, merge_mode='concat' , use_attention=True)

                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    # Loadweights into model
                    model.load_state_dict(checkpoint)
                    model = model.to(device)
                    model.eval()

                    with torch.no_grad():
                        outputs = model( X_tensor_rain, X_tensor_charge )
                    y_pred_classes = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.append(y_pred_classes)

                    #---------------------------------------GradCam computing ------------------------------
                    
                    # Obtain correctly classified indices:
                    if check == True : 
                        classified_indices = np.where(y_pred_classes == y)[0]
                        # Filter only the correctly classified samples of the current class 'classe'
                        pred_class_indices = [idx for idx in classified_indices if y_pred_classes[idx] == classe]
                    
                    # Obtain misclassified:
                    else : 
                        classified_indices = np.where(y_pred_classes != y)[0]
                        pred_class_indices = [idx for idx in classified_indices if y_pred_classes[idx] != classe]


                    for idx in pred_class_indices:
                        # Normalized samples
                        sample_tensor_charge = X_tensor_charge[idx].unsqueeze(0).to(device)
                        sample_tensor_rain = X_tensor_rain[idx].unsqueeze(0).to(device)

                        # Originals samples
                        original_samples = {
                            "Rain": X_test_tensor_orig_rain[idx].unsqueeze(0).to(device),
                            "Charge": X_test_tensor_orig_charge[idx].unsqueeze(0).to(device) }

                        for branch_index, name in enumerate(modalities):
                            gradcam = MultiBranchGradCAM_2(model, target_branch=f'branch_{branch_index+1}')
                            heatmap = gradcam( sample_tensor_rain, sample_tensor_charge)
                            gradcam_dict[idx][name].append(heatmap)
                            if len(samples_dict[idx][name]) == 0:
                                samples_dict[idx][name].append(original_samples[name])


            avg_gradcam_dict = {}
            sample_dict_maj={}
            for idx in gradcam_dict:
                avg_gradcam_dict[idx] = {}
                sample_dict_maj[idx]= {}
                for modality in gradcam_dict[idx]:
                    heatmaps = gradcam_dict[idx][modality]  
                    if len(heatmaps) > 90 / 2:   # Majority votes
                        heatmaps_tensor = [torch.tensor(h) for h in heatmaps]
                        stacked = torch.stack(heatmaps_tensor)
                        avg = stacked.mean(dim=0)     
                        avg_gradcam_dict[idx][modality] = avg
                        # Save only samples that verif the majority vote
                        sample_dict_maj[idx][modality] = samples_dict[idx][modality]


            plot_gradcam_heatmaps_averaged("Rain", avg_gradcam_dict, classe, folder, pred )
            plot_gradcam_heatmaps_averaged("Charge", avg_gradcam_dict, classe, folder, pred )
            plot_mean_sample_per_modality(sample_dict_maj, "Rain", folder, classe, pred)
            plot_mean_sample_per_modality(sample_dict_maj, "Charge", folder, classe, pred)






