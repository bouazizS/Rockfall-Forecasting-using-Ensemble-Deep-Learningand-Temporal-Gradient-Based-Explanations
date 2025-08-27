#from gradcam_Inception.inception_time_multi import InceptionTimeModel
from gradcam_Inception.inception_attention import InceptionTimeModel
from gradcam_Inception.gradcam import MultiBranchGradCAM_2
from gradcam_Inception.utils import  process_rockfall_data_multi, charge, preprocess_segments
import pandas as pd 
import numpy as np 
import time
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score,accuracy_score
from glob import glob
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import pickle
import os 
import cv2

#for model with attention
from gradcam_Inception.inception_attention import InceptionTimeModel


plt.rcParams.update({'axes.labelsize': 13,    # Taille du texte des labels des axes
                     'axes.titlesize': 12,   # Taille du texte du titre des axes
                     'xtick.labelsize': 14,  # Taille du texte des labels des ticks X
                     'ytick.labelsize': 14,  # Taille du texte des labels des ticks Y
                     'font.size': 13,        # Taille générale du texte
                     'legend.fontsize': 12})

def plot_gradcam_heatmaps_averaged_per_idx(modality, avg_gradcam_dict, classee, path_save, pred):
    all_heatmaps = []

    for idx in avg_gradcam_dict:
        if modality in avg_gradcam_dict[idx]:
            heatmap = avg_gradcam_dict[idx][modality]
            all_heatmaps.append(heatmap.cpu().numpy())  # au cas où c'est un tensor

    if not all_heatmaps:
        print(f"Aucune heatmap trouvée pour la modalité {modality}.")
        return

    stacked = np.stack(all_heatmaps)  # (N_samples, T)
    mean_vals = np.mean(stacked, axis=0)
    median_vals = np.median(stacked, axis=0)
    perc_10 = np.percentile(stacked, 25, axis=0)
    perc_90 = np.percentile(stacked, 75, axis=0)

    time = np.arange(-len(mean_vals), 0)

    plt.figure(figsize=(6, 3))
    plt.plot(time, np.flip(mean_vals), label="Mean", color="blue")
    plt.plot(time, np.flip(median_vals), label="Median", color="red", linestyle="--")
    plt.fill_between(time, np.flip(perc_10), np.flip(perc_90), color="gray", alpha=0.3, label="[25\%, 75\%]")
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
            tensor = samples_dict[idx][modality][0]  # prendre le seul élément
            all_samples.append(tensor.squeeze().cpu().numpy()) 

    if not all_samples:
        print(f"Aucun échantillon trouvé pour la modalité {modality}")
        return

    stacked = np.stack(all_samples)  # (N_samples, T)
    mean_signal = np.mean(stacked, axis=0)
    min_signal= np.min(stacked, axis=0)
    max_signal=np.max(stacked, axis=0)
    median_signal = np.median(stacked, axis=0)
    perc10 = np.percentile(stacked, 25, axis=0)
    perc90 = np.percentile(stacked, 75, axis=0)
    time = np.arange(-len(mean_signal), 0)

    plt.figure(figsize=(6, 3))

    # for i, signal in enumerate(all_samples):
    #     plt.plot( time, np.flip(signal), alpha=0.7)

    plt.plot(time, np.flip(mean_signal), label="Mean", color="blue")
    plt.plot(time, np.flip(median_signal), label="Median", color="red", linestyle="--")
    #plt.plot(time, np.flip(min_signal), label="Min", color="yellow", linestyle="--")
    #plt.plot(time, np.flip(max_signal), label="Max", color="green", linestyle="--")
    plt.fill_between(time, np.flip(perc10), np.flip(perc90), color="gray", alpha=0.3, label="[25\%, 75\%]")
    plt.xlabel("Time (hours)")
    plt.ylabel(f"{modality} signal")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)  # Adjust 'ncol' for multiple columns
    plt.tight_layout()
    filename = f"{modality}_Signal_stats_{pred}sur{classe}.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath)
    plt.close()

def plot_with_heatmap_new(sample_signal, heatmap, model, output_path, layer_name, jour_R, class_label, facteur , plot_type='both'):
    # Normalisation de la heatmap
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normaliser entre 0 et 1
    heatmap_resized = cv2.resize(heatmap, (1, sample_signal.shape[2]))  # Résize pour correspondre à la taille du signal
    heatmap_resized = heatmap_resized.T  # Devenir de taille (336, 1)
    time_axis = np.arange(-sample_signal.shape[2], 0)
    #`sample_signal` doit etre un tableau 2D : [time_steps, amplitude]
    if plot_type in ['both', 'subplot']:
        filename = f'{output_path}hp={jour_R}_{class_label}_subplot_{model.__class__.__name__}_{layer_name}.png'
        plt.figure(figsize=(14, 3))

        # Signal
        plt.subplot(1, 2, 1)
        plt.plot(time_axis, sample_signal[0][0, :], label='Signal')  # Utilisation du signal normalisé
        plt.xlabel('Time')
        plt.ylabel(f'{facteur}')
        plt.grid()

        # Heatmap
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

        #Trace avec le heatmap comme couleur
        for i in range(len(sample_signal[0][0, :]) - 1):
            color = plt.cm.jet(heatmap_norm[i])  # Utilisation du heatmap normalisé pour la couleur
            plt.plot([time_axis[i], time_axis[i + 1]], [sample_signal[0][0, i], sample_signal[0][0, i + 1]],
                     color=color, linewidth=2.5)

        plt.xlabel('Time (hours)')
        plt.ylabel(f'{facteur}')
        plt.grid()

        #Normalisation de la colormap pour le cbar
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=heatmap.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Importance')
        plt.show()
        plt.savefig(filename, bbox_inches='tight')

        plt.close()


def plot_gradcam_heatmaps_averaged_per_sample(modality, avg_gradcam_dict, classee, path_save, pred):
    all_heatmaps = []
    avg_gradcam_dict=avg_gradcam_dict.cpu().numpy()
    time = np.arange(-len(avg_gradcam_dict), 0)
    plt.figure(figsize=(6, 2.5))
    plt.plot(time, np.flip(avg_gradcam_dict), label="Mean", color="blue")
    plt.xlabel("Time (hours)")
    plt.ylabel("Grad-CAM Importance ")
    plt.tight_layout()
    filename = f"{modality}_gradcams_stats_{pred}sur{classee}_averaged_per_sample.png"
    filepath = os.path.join(path_save, filename)
    plt.savefig(filepath)
    plt.close()

folder ='gradcam_all_models/Rain_H_shuffle_attention'

modalities = ["Rain", "Charge"]

hp=1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()
df_sismo= pd.read_csv('.data/events/ev_sismo2_2024.csv', delimiter=',')
df_RR1= pd.read_csv('.data/events/1.0_RR1_2024.csv', delimiter=',')
#charge/discharge H 
df_precip = pd.read_csv('.data/events/data_meteo_Sabrine2_2024.csv', delimiter=',')
df_precip['AAAAMMJJHH'] = df_RR1['AAAAMMJJHH'].values
df_precip['AAAAMMJJHH'] = pd.to_datetime(df_precip['AAAAMMJJHH'])
lamb=0.2
df_precip['H'] = charge(df_precip,lamb) 

df_1 = process_rockfall_data_multi(df_RR1, df_sismo, x_days=hp, col_name='RR1',  concat_with_R=True, jour_pluie=14)
df_1['sample_id'] = df_1.index
#CHARGE H
df_2 = process_rockfall_data_multi(df_precip, df_sismo, x_days=hp, col_name='H', concat_with_R=False, jour_pluie=14)
df_2['sample_id'] = df_2.index

df = pd.merge(df_2, df_1, on='sample_id')
df = df.drop(columns=['sample_id'])
df.set_index('AAAAMMJJHH', inplace=True)

X_rain = df[[f"RR1_hour_{i}" for i in range(336)]].values  #----> colonnes de précipitations
X_charge = df[[f"H_hour_{i}" for i in range(336)]].values 

x_rain_scaled, _ = preprocess_segments(X_rain)
x_charge_scaled, _ = preprocess_segments(X_charge)

y = df['rockfall_target'].values
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

#conversion en tenseurs PyTorch
X_tensor_charge = torch.FloatTensor(x_charge_scaled).view(-1, 1, x_charge_scaled.shape[1]).to(device)
X_tensor_rain = torch.FloatTensor(x_rain_scaled).view(-1, 1, x_rain_scaled.shape[1]).to(device)
y_tensor = torch.LongTensor(y_encoded)

#original signal's tensor
X_test_tensor_orig_charge = torch.FloatTensor(X_charge).view(-1, 1, X_charge.shape[1]).to(device) #tesnor du signal original
X_test_tensor_orig_rain = torch.FloatTensor(X_rain).view(-1, 1, X_rain.shape[1]).to(device)


for classe in [0]:
    for pred in [1]:
        if classe == pred :
            check = True 
        else :
            check = False

        gradcam_dict = defaultdict(lambda: defaultdict(list))
        samples_dict = defaultdict(lambda: defaultdict(list))
        #initialize lists
        all_preds = []
        for test_year in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
            print(f"\nGradCam for models of {test_year}, pred = {pred} / class = {classe}")
            model_paths = glob(f"testing_models_gkf_multi_branch/Rain_H_all_models_attention_shuffle/HP_{hp}/test_year_{test_year}/val_*.pt")
            if not model_paths:
                print(f"Aucun modèle trouvé pour {test_year}")

            for model_path in model_paths:
                metric_path= model_path[:-11]+"test_metrics_val_year"+model_path[-7:-3]+".json"
                # print(model_path)
                #configuration du modèle
                model = InceptionTimeModel(
                    in_channels_1=1,  #time series (1 channel)
                    in_channels_2=1,  #second time series (1 channel)
                    num_classes=2, merge_mode='concat') # , use_attention=True)

                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                #loadweights into model
                model.load_state_dict(checkpoint)
                model = model.to(device)
                model.eval()

                with torch.no_grad():
                    outputs = model( X_tensor_rain, X_tensor_charge )
                y_pred_classes = outputs.argmax(dim=1).cpu().numpy()
                all_preds.append(y_pred_classes)

                #---------------------------------------GradCam computing ------------------------------
                #obtain correctly classified indices:

                if check == True : 
                    correctly_classified_indices = np.where(y_pred_classes == y)[0]
                    #filter only the correctly classified samples of the current class 'classe'
                    correct_class_indices = [idx for idx in correctly_classified_indices if y_pred_classes[idx] == classe]
                
                
                else : 
                    #obtain misclassified:
                    correctly_classified_indices = np.where(y_pred_classes != y)[0]
                    correct_class_indices = [idx for idx in correctly_classified_indices if y_pred_classes[idx] != classe]  #for miss classified !

                #print(correct_class_indices)

                for idx in correct_class_indices:
                    #normalized samples
                    sample_tensor_charge = X_tensor_charge[idx].unsqueeze(0).to(device)
                    sample_tensor_rain = X_tensor_rain[idx].unsqueeze(0).to(device)

                    #originals samples
                    original_samples = {
                        "Rain": X_test_tensor_orig_rain[idx].unsqueeze(0).to(device),
                        "Charge": X_test_tensor_orig_charge[idx].unsqueeze(0).to(device) }

                    for branch_index, name in enumerate(modalities):
                        #print(f'branch_{branch_index+1}')
                        gradcam = MultiBranchGradCAM_2(model, target_branch=f'branch_{branch_index+1}')
                        heatmap = gradcam( sample_tensor_rain, sample_tensor_charge)
                        gradcam_dict[idx][name].append(heatmap)
                        if len(samples_dict[idx][name]) == 0:
                            samples_dict[idx][name].append(original_samples[name])


        avg_gradcam_dict = {}
        # for idx in gradcam_dict:
        #     avg_gradcam_dict[idx] = {}
        #     for modality in gradcam_dict[idx]:
        #         heatmaps = gradcam_dict[idx][modality]  
        #         heatmaps_tensor = [torch.tensor(h) for h in heatmaps]
        #         #stack et moyenne
        #         stacked = torch.stack(heatmaps_tensor)
        #         avg = stacked.mean(dim=0)     
        #         avg_gradcam_dict[idx][modality] = avg

        sample_dict_maj={}

        for idx in gradcam_dict:
            avg_gradcam_dict[idx] = {}
            sample_dict_maj[idx]= {}
            for modality in gradcam_dict[idx]:
                heatmaps = gradcam_dict[idx][modality]  
                if len(heatmaps) >= 90 / 2:   # > moitié des modèles
                    heatmaps_tensor = [torch.tensor(h) for h in heatmaps]
                    stacked = torch.stack(heatmaps_tensor)
                    avg = stacked.mean(dim=0)     
                    avg_gradcam_dict[idx][modality] = avg
                    #save only samples that verif the majority vote
                    sample_dict_maj[idx][modality] = samples_dict[idx][modality]


        #print(samples_dict)
        #print(avg_gradcam_dict)

        plot_gradcam_heatmaps_averaged_per_idx("Rain", avg_gradcam_dict, classe, folder, pred )
        plot_gradcam_heatmaps_averaged_per_idx("Charge", avg_gradcam_dict, classe, folder, pred )
        plot_mean_sample_per_modality(sample_dict_maj, "Rain", folder, classe, pred)
        plot_mean_sample_per_modality(sample_dict_maj, "Charge", folder, classe, pred)






#For plotting one example of gradcam on one model overlayed on the signal
#model_ind=4
# sample_idx=328
# name="Rain"
# # sample_precip_flipped = np.flip(samples_dict[sample_idx][name][0].cpu().numpy(), axis=-1)
# # heatmap_flipped = np.flip(gradcam_dict[sample_idx][name][model_ind], axis=-1)
# # plot_with_heatmap_new(sample_precip_flipped, heatmap_flipped, model, folder, "branch_1", hp ,"Class 1",facteur="Rainfall [mm/h]", plot_type='overlay')

# plot_gradcam_heatmaps_averaged_per_sample(name, avg_gradcam_dict[sample_idx][name], classe, folder, pred )



