from gradcam_Inception.InceptionTime_attention import InceptionTimeModel 
from gradcam_Inception.gradcam import MultiBranchGradCAM_2
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

def plot_gradcam_heatmaps_averaged(modality, avg_gradcam_dict, classee, path_save, pred):
    all_heatmaps = []
    for idx in avg_gradcam_dict:
        if modality in avg_gradcam_dict[idx]:
            heatmap = avg_gradcam_dict[idx][modality]
            all_heatmaps.append(heatmap.cpu().numpy())  # if it is a tensor

    if not all_heatmaps:
        print(f"Aucune heatmap trouvée pour la modalité {modality}.")
        return

    stacked = np.stack(all_heatmaps)  # (N_samples, T)
    mean_vals = np.mean(stacked, axis=0)
    median_vals = np.median(stacked, axis=0)
    perc_25 = np.percentile(stacked, 25, axis=0)
    perc_75 = np.percentile(stacked, 75, axis=0)

    time = np.arange(-len(mean_vals), 0)
    plt.figure(figsize=(6, 3))
    plt.plot(time, np.flip(mean_vals), label="Mean", color="blue")
    plt.plot(time, np.flip(median_vals), label="Median", color="red", linestyle="--")
    plt.fill_between(time, np.flip(perc_25), np.flip(perc_75), color="gray", alpha=0.3, label="[25\%, 75\%]")
    plt.xlabel("Time (hours)")
    plt.ylabel("Grad-CAM Importance ")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)  # Adjust 'ncol' for multiple columns
    plt.tight_layout()
    filename = f"{modality}_gradcams_stats_{pred}sur{classee}.png"
    filepath = os.path.join(path_save, filename)
    plt.savefig(filepath)
    plt.close()


def plot_mean_sample_per_modality(samples_dict, modality, save_path, classe, pred):
    all_samples = []
    for idx in samples_dict:
        if modality in samples_dict[idx] and len(samples_dict[idx][modality]) > 0:
            tensor = samples_dict[idx][modality][0]  
            all_samples.append(tensor.squeeze().cpu().numpy()) 

    if not all_samples:
        print(f"No sample found for modality {modality}")
        return

    stacked = np.stack(all_samples)  # (N_samples, T)
    mean_signal = np.mean(stacked, axis=0)
    min_signal= np.min(stacked, axis=0)
    max_signal=np.max(stacked, axis=0)
    median_signal = np.median(stacked, axis=0)
    perc25 = np.percentile(stacked, 25, axis=0)
    perc75 = np.percentile(stacked, 75, axis=0)
    time = np.arange(-len(mean_signal), 0)

    plt.figure(figsize=(6, 3))
    plt.plot(time, np.flip(mean_signal), label="Mean", color="blue")
    plt.plot(time, np.flip(median_signal), label="Median", color="red", linestyle="--")
    plt.fill_between(time, np.flip(perc25), np.flip(perc75), color="gray", alpha=0.3, label="[25\%, 75\%]")
    plt.xlabel("Time (hours)")
    plt.ylabel(f"{modality} signal")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)  # Adjust 'ncol' for multiple columns
    plt.tight_layout()
    filename = f"{modality}_Signal_stats_{pred}/{classe}.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath)
    plt.close()

def plot_with_heatmap(sample_signal, heatmap, model, output_path, layer_name, jour_R, class_label, facteur , plot_type='both'):
    # Heatmap normalisation
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalisation between 0 et 1
    heatmap_resized = cv2.resize(heatmap, (1, sample_signal.shape[2]))  # Résize pour correspondre à la taille du signal
    heatmap_resized = heatmap_resized.T  # Devenir de taille (336, 1)
    time_axis = np.arange(-sample_signal.shape[2], 0)
    # `sample_signal` should be a 2D array: [time_steps, amplitude]
    if plot_type in ['both', 'subplot']:
        filename = f'{output_path}hp={jour_R}_{class_label}_subplot_{model.__class__.__name__}_{layer_name}.png'
        plt.figure(figsize=(14, 3))

        # Plot the signal
        plt.subplot(1, 2, 1)
        plt.plot(time_axis, sample_signal[0][0, :], label='Signal')  # Utilisation du signal normalisé
        plt.xlabel('Time')
        plt.ylabel(f'{facteur}')
        plt.grid()

        # Plot the heatmap
        ax_heatmap = plt.subplot(1, 2, 2)
        im = ax_heatmap.imshow(heatmap_resized, aspect='auto', cmap='jet', vmin=0, vmax=heatmap.max())
        plt.xlabel('Time')
        plt.grid()

        cbar = plt.colorbar(im, ax=ax_heatmap, label='Importance')
        cbar.ax.set_ylabel('Importance')
        plt.show()
        plt.savefig(filename, bbox_inches='tight')

    if plot_type in ['both', 'overlay']:
        filename = f'{output_path}overlay_{model.__class__.__name__}_{layer_name}.png'
        plt.figure(figsize=(7, 2.5))

        # Plot the signal with the heatmap as color
        for i in range(len(sample_signal[0][0, :]) - 1):
            color = plt.cm.jet(heatmap_norm[i])   # Use normalized heatmap values for the color
            plt.plot([time_axis[i], time_axis[i + 1]], [sample_signal[0][0, i], sample_signal[0][0, i + 1]],
                     color=color, linewidth=2.5)
        plt.xlabel('Time (hours)')
        plt.ylabel(f'{facteur}')
        plt.grid()

        # Normalize the colormap for the colorbar
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=heatmap.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Importance')
        plt.show()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()



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






