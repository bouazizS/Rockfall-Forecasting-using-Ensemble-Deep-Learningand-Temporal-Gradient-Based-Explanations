
import wfdb
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                #r"\usepackage{cmbright}",
            ]
        ),
    }
)


def preprocess_segments(segments):
    """ 
    Normalize segments and reshape for the model.
    """
    scaler = StandardScaler()
    segments_scaled = scaler.fit_transform(segments.reshape(-1, segments.shape[-1])).reshape(segments.shape)
    return segments_scaled[..., np.newaxis], scaler  # Add a dimension for Conv1D

def preprocess_segments_multi(segments_temp, segments_rain):
    scaler_temp = StandardScaler()
    scaler_rain = StandardScaler()

    #normalize temperature and precipitation segments separately
    segments_temp_scaled = scaler_temp.fit_transform(segments_temp.reshape(-1, segments_temp.shape[-1])).reshape(segments_temp.shape)
    segments_rain_scaled = scaler_rain.fit_transform(segments_rain.reshape(-1, segments_rain.shape[-1])).reshape(segments_rain.shape)

    #combine both temperature and precipitation segments along the last axis (336, 2)
    segments_combined = np.stack([segments_temp_scaled, segments_rain_scaled], axis=-1)

    return segments_combined, scaler_temp, scaler_rain

def denormalize_signal(normalized_signal, scaler):
    """
    Dénormalise un signal à partir du scaler.
    """
    # Assurez-vous que le signal est 2D pour scaler.inverse_transform
    original_shape = normalized_signal.shape
    if len(original_shape) > 2:
        normalized_signal = normalized_signal.reshape(-1, original_shape[-1])

    # Inverse transform
    denormalized = scaler.inverse_transform(normalized_signal)

    # Restaurez la forme originale
    return denormalized.reshape(original_shape)

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, output_path, hp):
    epochs_range = range(1, len(train_losses) + 1)

    # Courbe de perte
    loss_filename = f'{output_path}plot_model_loss_acc_hp={hp}.pdf'
    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Sauvegarder l'image contenant les deux courbes (perte et précision)
    plt.tight_layout()
    plt.savefig(loss_filename)
    plt.close()


def plot_PR(precision, recall, pr_auc, thresholds, path, hp ):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'PR Curve (AUC = {pr_auc:.4f})')
    #thresholds = np.append(thresholds, 1) 
    for i in range(0, len(thresholds), max(1, len(thresholds) // 10)):
        plt.annotate(f"{thresholds[i]:.2f}", (recall[i], precision[i]), textcoords="offset points", xytext=(-10,5), ha='center')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (hp={hp})')
    plt.legend()
    plt.grid()
    plt.savefig(f"{path}PR_Curve_hp={hp}.png")
    plt.close()

def plot_roc(fpr, tpr, roc_auc, path , hp ):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonale (aléatoire)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC Curve (hp={hp})')
    plt.legend()
    plt.grid()
    plt.savefig(f"{path}ROC_Curve_hp={hp}.png")
    plt.show()

def process_rockfall_data_year(df_RR, df_sismo, x_days, jour_pluie=14):
    df_RR['AAAAMMJJHH'] = pd.to_datetime(df_RR['AAAAMMJJHH'])
    df_RR.set_index('AAAAMMJJHH', inplace=True)
    df_RR_per_day = df_RR.resample('1D').sum()
    df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
    df_RR.reset_index('AAAAMMJJHH', inplace=True)

    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH'] = df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])

    #dates communes entre df_sismo et df_RR_per_day
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_RR_per_day['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_RR_per_day['AAAAMMJJHH'].max())

    df_sismo_ = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    df_RR_per_day = df_RR_per_day[(df_RR_per_day['AAAAMMJJHH'] >= start_date) & (df_RR_per_day['AAAAMMJJHH'] <= end_date)]
    df_RR = df_RR[(df_RR['AAAAMMJJHH'] >= start_date) & (df_RR['AAAAMMJJHH'] <= end_date)]

    df_sismo_.loc[:, 'AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_RR_per_day.set_index('AAAAMMJJHH', inplace=True)
    df_RR.set_index('AAAAMMJJHH', inplace=True)

    rockfall_x_days = []
    volume_x_days = []

    #boucle pour vérifier les rockfalls dans les jours suivants
    for date in df_RR_per_day.index:
        end_date_range = date + pd.Timedelta(days=x_days)
        rockfalls_in_x_days = df_sismo_[
            (df_sismo_['AAAAMMJJHH'].dt.date >= date.date()) &
            (df_sismo_['AAAAMMJJHH'].dt.date < end_date_range.date()) &
            (df_sismo_['type'] == 'R')
        ]
        if not rockfalls_in_x_days.empty:
            rockfall_x_days.append(1)
            volume_x_days.append(rockfalls_in_x_days['A(nm/s)'].sum())
        else:
            rockfall_x_days.append(0)
            volume_x_days.append(0)

    #ajout des résultats au dataframe
    df_RR_per_day['rockfall'] = rockfall_x_days
    df_RR_per_day['volume'] = volume_x_days

    #creer les colonnes pour les précipitations des 14 jours précédents
    for i in range(1, jour_pluie + 1):
        df_RR_per_day[f'RR1_day-{i}'] = df_RR_per_day['RR1'].shift(i)

    df_RR_per_day['rockfall_target'] = df_RR_per_day['rockfall'].shift(0).astype(int)
    df_RR_per_day['Amp_target'] = df_RR_per_day['volume'].shift(0)
    df_RR_per_day = df_RR_per_day.dropna()

    #selection des colonnes Rockfall et volume
    df_final_columns = [col for col in df_RR_per_day.columns if 'RR1_day-' in col]
    df_final_columns.extend(['rockfall_target', 'Amp_target'])

    df_hourP_R = df_RR_per_day[df_final_columns].copy()
    df_hourP_R.loc[:, 'AAAAMMJJHH'] = df_RR_per_day.index 

    #creer une plage horaire complète et remplir les heures manquantes
    all_hours = pd.date_range(start=df_RR.index.min(), end=df_RR.index.max(), freq='h')
    df_RR = df_RR.reindex(all_hours, fill_value=0)

    #creer un dataframe avec les précipitations décalées sur 14 jours
    window_size = jour_pluie * 24
    df_shifted = pd.DataFrame(index=df_RR.index)

    #creer un dictionnaire pour stocker les colonnes
    shifted_columns = {f'precip_hour_{i}': df_RR['RR1'].shift(i) for i in range(window_size)}

    #ajouter toutes les colonnes en une seule fois
    df_shifted = pd.concat(shifted_columns, axis=1).fillna(0)

    df_shifted = df_shifted[df_shifted.index.hour == 23]
    df_shifted = df_shifted.iloc[jour_pluie - 1:]  # pour éviter le décalage

    #concaténer avec les cibles rockfall et amplitude
    colonnes_a_concatener = df_hourP_R[['AAAAMMJJHH', 'rockfall_target', 'Amp_target']].reset_index(drop=True)
    df_concatene = pd.concat([df_shifted.reset_index(drop=True), colonnes_a_concatener], axis=1)

    #utiliser la colonne 'AAAAMMJJHH' comme index final
    df_concatene.set_index('AAAAMMJJHH', inplace=True)

    return df_concatene

def charge(dfpluie, lamb):
    pluv = dfpluie['pluvio'].values  
    H = np.zeros_like(pluv) 
    Dt = 1 / 24  
    for p in range(1, len(pluv)):
        H[p] = H[p-1] * np.exp(-Dt / lamb) + pluv[p]
    return H



def process_rockfall_data(df_RR, df_sismo, x_days, jour_pluie=14):
    df_RR['AAAAMMJJHH'] = pd.to_datetime(df_RR['AAAAMMJJHH'])
    df_RR.set_index('AAAAMMJJHH', inplace=True)
    df_RR_per_day = df_RR.resample('1D').sum()
    df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
    df_RR.reset_index('AAAAMMJJHH', inplace=True)

    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])


    #dates communes entre df_sismo et df_RR_per_day
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_RR_per_day['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_RR_per_day['AAAAMMJJHH'].max())

    df_sismo_ = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    df_RR_per_day = df_RR_per_day[(df_RR_per_day['AAAAMMJJHH'] >= start_date) & (df_RR_per_day['AAAAMMJJHH'] <= end_date)]
    df_RR = df_RR[(df_RR['AAAAMMJJHH']>= start_date) & (df_RR['AAAAMMJJHH'] <= end_date)]

    #df_sismo_['AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_sismo_.loc[:, 'AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_RR_per_day.set_index('AAAAMMJJHH', inplace=True)
    df_RR.set_index('AAAAMMJJHH', inplace=True)

    rockfall_x_days = []
    volume_x_days = []

    #boucle pour vérifier les rockfalls dans les jours suivants
    for date in df_RR_per_day.index:
        end_date_range = date + pd.Timedelta(days=x_days)
        rockfalls_in_x_days = df_sismo_[
            (df_sismo_['AAAAMMJJHH'].dt.date >= date.date()) &
            (df_sismo_['AAAAMMJJHH'].dt.date < end_date_range.date()) &
            (df_sismo_['type'] == 'R')
        ]
        if not rockfalls_in_x_days.empty:
            rockfall_x_days.append(1)
            volume_x_days.append(rockfalls_in_x_days['A(nm/s)'].sum())
        else:
            rockfall_x_days.append(0)
            volume_x_days.append(0)

    df_RR_per_day['rockfall'] = rockfall_x_days
    df_RR_per_day['volume'] = volume_x_days

    #creation colonnes pour les précipitations des 14 jours précédents
    for i in range(1, jour_pluie+1):
        df_RR_per_day[f'RR1_day-{i}'] = df_RR_per_day['RR1'].shift(i)

    df_RR_per_day['rockfall_target'] = df_RR_per_day['rockfall'].shift(0).astype(int)
    df_RR_per_day['Amp_target'] = df_RR_per_day['volume'].shift(0)
    df_RR_per_day = df_RR_per_day.dropna()

    #colonnes Rockfall et volume
    df_final_columns = [col for col in df_RR_per_day.columns if 'RR1_day-' in col]
    df_final_columns.extend(['rockfall_target', 'Amp_target'])

    df_hourP_R = df_RR_per_day[df_final_columns]


    #creation plage horaire complète et remplir les heures manquantes
    all_hours = pd.date_range(start=df_RR.index.min(), end=df_RR.index.max(), freq='h')
    df_RR = df_RR.reindex(all_hours, fill_value=0)

    #creer dataframe avec les précipitations décalées sur 14 jours
    window_size = jour_pluie * 24
    df_shifted = pd.DataFrame(index=df_RR.index)

    #dictionnaire pour stocker les colonnes
    shifted_columns = {f'precip_hour_{i}': df_RR['RR1'].shift(i) for i in range(window_size)}

    #ajouter les colonnes en une seule fois
    df_shifted = pd.concat(shifted_columns, axis=1).fillna(0)


    df_shifted = df_shifted[df_shifted.index.hour == 23]
    df_shifted = df_shifted.iloc[jour_pluie - 1:]  # pour éviter le décalage

    #concat avec les targets rockfall et amplitude
    colonnes_a_concatener = df_hourP_R[['rockfall_target', 'Amp_target']]
    df_concatene = pd.concat([df_shifted.reset_index(drop=True), colonnes_a_concatener.reset_index(drop=True)], axis=1)

    return df_concatene



def process_rockfall_data_multi(df_RR, df_sismo, x_days, col_name='RR1',concat_with_R=True, jour_pluie=14):
    df_RR['AAAAMMJJHH'] = pd.to_datetime(df_RR['AAAAMMJJHH'])
    df_RR.set_index('AAAAMMJJHH', inplace=True)
    if col_name=='RR1':
      df_RR_per_day = df_RR.resample('1D').sum()
      df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
      df_RR.reset_index('AAAAMMJJHH', inplace=True)
    else: 
      df_RR_per_day = df_RR.resample('1D').mean()
      df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
      df_RR.reset_index('AAAAMMJJHH', inplace=True)   

    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])


    #dates communes entre df_sismo et df_RR_per_day
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_RR_per_day['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_RR_per_day['AAAAMMJJHH'].max())

    df_sismo_ = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    df_RR_per_day = df_RR_per_day[(df_RR_per_day['AAAAMMJJHH'] >= start_date) & (df_RR_per_day['AAAAMMJJHH'] <= end_date)]
    df_RR = df_RR[(df_RR['AAAAMMJJHH']>= start_date) & (df_RR['AAAAMMJJHH'] <= end_date)]

    #df_sismo_['AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_sismo_.loc[:, 'AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_RR_per_day.set_index('AAAAMMJJHH', inplace=True)
    df_RR.set_index('AAAAMMJJHH', inplace=True)

    rockfall_x_days = []
    volume_x_days = []

    #boucle pour vérifier les rockfalls dans les jours suivants
    for date in df_RR_per_day.index:
        end_date_range = date + pd.Timedelta(days=x_days)
        rockfalls_in_x_days = df_sismo_[
            (df_sismo_['AAAAMMJJHH'].dt.date >= date.date()) &
            (df_sismo_['AAAAMMJJHH'].dt.date < end_date_range.date()) &
            (df_sismo_['type'] == 'R')
        ]
        if not rockfalls_in_x_days.empty:
            rockfall_x_days.append(1)
            volume_x_days.append(rockfalls_in_x_days['A(nm/s)'].sum())
        else:
            rockfall_x_days.append(0)
            volume_x_days.append(0)

    df_RR_per_day['rockfall'] = rockfall_x_days
    df_RR_per_day['volume'] = volume_x_days

    #creation colonnes pour les précipitations des 14 jours précédents
    for i in range(1, jour_pluie+1):
        df_RR_per_day[f'{col_name}_day-{i}'] = df_RR_per_day[col_name].shift(i)

    df_RR_per_day['rockfall_target'] = df_RR_per_day['rockfall'].shift(0).astype(int)
    df_RR_per_day['Amp_target'] = df_RR_per_day['volume'].shift(0)
    df_RR_per_day = df_RR_per_day.dropna()

    #colonnes Rockfall et volume
    df_final_columns = [col for col in df_RR_per_day.columns if f'{col_name}_day-' in col]
    df_final_columns.extend(['rockfall_target', 'Amp_target'])

    df_hourP_R = df_RR_per_day[df_final_columns]
    df_hourP_R.loc[:, 'AAAAMMJJHH'] = df_RR_per_day.index 


    #creation plage horaire complète et remplir les heures manquantes
    all_hours = pd.date_range(start=df_RR.index.min(), end=df_RR.index.max(), freq='h')
    df_RR = df_RR.reindex(all_hours, fill_value=0)

    #creer dataframe avec les précipitations décalées sur 14 jours
    window_size = jour_pluie * 24
    df_shifted = pd.DataFrame(index=df_RR.index)

    #dictionnaire pour stocker les colonnes
    shifted_columns = {f'{col_name}_hour_{i}': df_RR[col_name].shift(i) for i in range(window_size)}

    #ajouter les colonnes en une seule fois
    df_shifted = pd.concat(shifted_columns, axis=1).fillna(0)


    df_shifted = df_shifted[df_shifted.index.hour == 23]
    df_shifted = df_shifted.iloc[jour_pluie - 1:]  # pour éviter le décalage

    #concat avec les targets rockfall et amplitude
    colonnes_a_concatener = df_hourP_R[['AAAAMMJJHH','rockfall_target', 'Amp_target']]
    df_concatene = pd.concat([df_shifted.reset_index(drop=True), colonnes_a_concatener.reset_index(drop=True)], axis=1)
    #utiliser la colonne 'AAAAMMJJHH' comme index final
    #df_concatene.set_index('AAAAMMJJHH', inplace=True)

    if concat_with_R:
     return df_concatene
    else :
      return df_shifted.reset_index(drop=True)



def plot_save_conf_matrix(conf_matrix, class_labels,  save_path):
    # Configuration du plot
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    #plt.title('Confusion Matrix')
    # Sauvegarde de l'image
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_rockfall_data_target_vect(df_RR, df_sismo, x_days, jour_pluie=14):
    df_RR['AAAAMMJJHH'] = pd.to_datetime(df_RR['AAAAMMJJHH'])
    df_RR.set_index('AAAAMMJJHH', inplace=True)
    df_RR_per_day = df_RR.resample('1D').sum()
    df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
    df_RR.reset_index('AAAAMMJJHH', inplace=True)

    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
    df_sismo['AAAAMMJJHH']=df_sismo['AAAAMMJJHH'].dt.strftime('%Y-%m-%d')
    df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])


    #dates communes entre df_sismo et df_RR_per_day
    start_date = max(df_sismo['AAAAMMJJHH'].min(), df_RR_per_day['AAAAMMJJHH'].min())
    end_date = min(df_sismo['AAAAMMJJHH'].max(), df_RR_per_day['AAAAMMJJHH'].max())

    df_sismo_ = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]
    df_RR_per_day = df_RR_per_day[(df_RR_per_day['AAAAMMJJHH'] >= start_date) & (df_RR_per_day['AAAAMMJJHH'] <= end_date)]
    df_RR = df_RR[(df_RR['AAAAMMJJHH']>= start_date) & (df_RR['AAAAMMJJHH'] <= end_date)]

    #df_sismo_['AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_sismo_.loc[:, 'AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
    df_RR_per_day.set_index('AAAAMMJJHH', inplace=True)
    df_RR.set_index('AAAAMMJJHH', inplace=True)
    rockfall_days = {f'rockfall_{i+1}d': [] for i in range(x_days)}

    for date in df_RR_per_day.index:
        for i in range(x_days):
            day_to_check = date + pd.Timedelta(days=i)
            rockfalls_on_day = df_sismo_[
                (df_sismo_['AAAAMMJJHH'].dt.date == day_to_check.date()) &
                (df_sismo_['type'] == 'R')
            ]
            rockfall_days[f'rockfall_{i+1}d'].append(1 if not rockfalls_on_day.empty else 0)

    for key, values in rockfall_days.items():
        df_RR_per_day[key] = values

    #--> a ce niveau colonne rockfall de df_RR_per_day indique s'il y a de rockfall de la date dans colonne AAAAMMJJHH jusqu'à x_days-1  
    
    #creation colonnes pour les précipitations des 14 jours précédents
    for i in range(1, jour_pluie+1):
        df_RR_per_day[f'RR1_day-{i}'] = df_RR_per_day['RR1'].shift(i)

    #--> a ce niveau colonne  AAAAMMJJHH de df_RR_per_day indique la date du lendemain de la periode de jour_pluie (elle est non incluse !! )
    # date à partir de laquelle on voit il ya deu rockfall ou non !! 

    for i in range(x_days):
        df_RR_per_day[f'rockfall_target_{i+1}d'] = df_RR_per_day[f'rockfall_{i+1}d'].shift(0).astype(int)

    #df_RR_per_day['Amp_target'] = df_RR_per_day['volume'].shift(0)
    df_RR_per_day = df_RR_per_day.dropna()

    #colonnes Rockfall et volume
    df_final_columns = [col for col in df_RR_per_day.columns if 'RR1_day-' in col]
    df_final_columns.extend([f'rockfall_target_{i+1}d' for i in range(x_days)])


    df_hourP_R = df_RR_per_day[df_final_columns]


    #creation plage horaire complète et remplir les heures manquantes
    all_hours = pd.date_range(start=df_RR.index.min(), end=df_RR.index.max(), freq='h')
    df_RR = df_RR.reindex(all_hours, fill_value=0)


    #creer dataframe avec les précipitations décalées sur 14 jours
    window_size = jour_pluie * 24
    df_shifted = pd.DataFrame(index=df_RR.index)
    shifted_columns = {f'precip_hour_{i}': df_RR['RR1'].shift(i) for i in range(window_size)}
    df_shifted = pd.concat(shifted_columns, axis=1).fillna(0)
    df_shifted = df_shifted[df_shifted.index.hour == 23]
    df_shifted = df_shifted.iloc[jour_pluie - 1:]  # pour éviter le décalage prendre à partir de la date complete avec les données 

    #concat pluie sur jour avant avec les targets rockfall et amplitude
    colonnes_a_concatener = df_hourP_R[[f'rockfall_target_{i+1}d' for i in range(x_days)]]
    df_concatene = pd.concat([df_shifted.reset_index(drop=True), colonnes_a_concatener.reset_index(drop=True)], axis=1)

    return df_concatene


def process_rockfall_data_target_vect_hour(df_RR, df_sismo, x_hours, jour_pluie=14):
  df_RR['AAAAMMJJHH'] = pd.to_datetime(df_RR['AAAAMMJJHH'])
  df_RR.set_index('AAAAMMJJHH', inplace=True)
  df_RR_per_day = df_RR.resample('1D').sum()
  df_RR_per_day.reset_index('AAAAMMJJHH', inplace=True)
  df_RR.reset_index('AAAAMMJJHH', inplace=True)

  df_sismo['AAAAMMJJHH'] = pd.to_datetime(df_sismo['AAAAMMJJHH'])
  df_sismo['AAAAMMJJHH']=pd.to_datetime(df_sismo['AAAAMMJJHH'], format='%Y%m%d%H')
  df_sismo['AAAAMMJJHH'] =df_sismo['AAAAMMJJHH'] .dt.floor('h')

  #dates communes entre df_sismo et df_RR_per_day
  start_date = max(df_sismo['AAAAMMJJHH'].min(), df_RR_per_day['AAAAMMJJHH'].min())
  end_date = min(df_sismo['AAAAMMJJHH'].max(), df_RR_per_day['AAAAMMJJHH'].max())

  df_sismo_ = df_sismo[(df_sismo['AAAAMMJJHH'] >= start_date) & (df_sismo['AAAAMMJJHH'] <= end_date)]

  df_RR_per_day = df_RR_per_day[(df_RR_per_day['AAAAMMJJHH'] >= start_date) & (df_RR_per_day['AAAAMMJJHH'] <= end_date)]

  start_date_next_day = start_date.normalize() + pd.Timedelta(days=1)
  df_RR = df_RR[(df_RR['AAAAMMJJHH']>= start_date_next_day ) & (df_RR['AAAAMMJJHH'] <= end_date)]

  #df_sismo_['AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
  df_sismo_.loc[:, 'AAAAMMJJHH'] = pd.to_datetime(df_sismo_['AAAAMMJJHH'])
  df_RR_per_day.set_index('AAAAMMJJHH', inplace=True)
  df_RR.set_index('AAAAMMJJHH', inplace=True)
  rockfall_hours = {f'rockfall_{i+1}h': [] for i in range(x_hours)}

  for date in df_RR_per_day.index:
      for i in range(x_hours):
          day_to_check = date + pd.Timedelta(hours=i)
          rockfalls_on_hour = df_sismo_[
              (df_sismo_['AAAAMMJJHH'] == day_to_check) &
              (df_sismo_['type'] == 'R')
          ]
          rockfall_hours[f'rockfall_{i+1}h'].append(1 if not rockfalls_on_hour.empty else 0)

  for key, values in rockfall_hours.items():
      df_RR_per_day[key] = values

    #--> a ce niveau colonne rockfall de df_RR_per_day indique s'il y a de rockfall de la date dans colonne AAAAMMJJHH jusqu'à x_days-1

  #creation colonnes pour les précipitations des 14 jours précédents
  for i in range(1, jour_pluie+1):
      df_RR_per_day[f'RR1_day-{i}'] = df_RR_per_day['RR1'].shift(i)

  #--> a ce niveau colonne  AAAAMMJJHH de df_RR_per_day indique la date du lendemain de la periode de jour_pluie (elle est non incluse !! )
  # date à partir de laquelle on voit il ya deu rockfall ou non !!

  for i in range(x_hours):
      df_RR_per_day[f'rockfall_target_{i+1}h'] = df_RR_per_day[f'rockfall_{i+1}h'].shift(0).astype(int)

  #df_RR_per_day['Amp_target'] = df_RR_per_day['volume'].shift(0)
  df_RR_per_day = df_RR_per_day.dropna()

  #colonnes Rockfall et volume
  df_final_columns = [col for col in df_RR_per_day.columns if 'RR1_day-' in col]
  df_final_columns.extend([f'rockfall_target_{i+1}h' for i in range(x_hours)])

  df_hourP_R = df_RR_per_day[df_final_columns]

  #creation plage horaire complète et remplir les heures manquantes
  all_hours = pd.date_range(start=df_RR.index.min(), end=df_RR.index.max(), freq='h')
  df_RR = df_RR.reindex(all_hours, fill_value=0)

  #creer dataframe avec les précipitations décalées sur 14 jours
  window_size = jour_pluie * 24
  df_shifted = pd.DataFrame(index=df_RR.index)
  shifted_columns = {f'precip_hour_{i}': df_RR['RR1'].shift(i) for i in range(window_size)}
  df_shifted = pd.concat(shifted_columns, axis=1).fillna(0)
  df_shifted = df_shifted[df_shifted.index.hour == 23]
  df_shifted = df_shifted.iloc[jour_pluie - 1:]  # pour éviter le décalage prendre à partir de la date complete avec les données

  #concat pluie sur jour avant avec les targets rockfall et amplitude
  colonnes_a_concatener = df_hourP_R[[f'rockfall_target_{i+1}h' for i in range(x_hours)]]
  df_concatene = pd.concat([df_shifted.reset_index(drop=True), colonnes_a_concatener.reset_index(drop=True)], axis=1)

  return df_concatene



#for cross val
def plot_training_curves_gkf(train_losses, val_losses, train_accuracies, val_accuracies, output_path, hp, year, val_year):
    epochs_range = range(1, len(train_losses) + 1)

    # Courbe de perte
    loss_filename = f'{output_path}plot_model_loss_acc_hp={hp}_test{year}_val{val_year}_.pdf'
    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Training Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Sauvegarder l'image contenant les deux courbes (perte et précision)
    plt.tight_layout()
    plt.savefig(loss_filename)
    plt.close()


def plot_PR_gkf_per_year_val(precision, recall, pr_auc, thresholds, path , year, classe):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'PR Curve (AUC = {pr_auc:.4f})')
    #thresholds = np.append(thresholds, 1) 
    for i in range(0, len(thresholds), max(1, len(thresholds) // 10)):
        plt.annotate(f"{thresholds[i]:.2f}", (recall[i], precision[i]), textcoords="offset points", xytext=(-10,5), ha='center')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {classe}')
    plt.legend()
    plt.grid()
    PR_name = os.path.join(path, f'PR_val_{year}.png')
    plt.savefig(PR_name)
    plt.close()


def plot_PR_gkf(precision, recall, pr_auc, thresholds, path, hp , year, classe):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'PR Curve (AUC = {pr_auc:.4f})')
    #thresholds = np.append(thresholds, 1) 
    for i in range(0, len(thresholds), max(1, len(thresholds) // 10)):
        plt.annotate(f"{thresholds[i]:.2f}", (recall[i], precision[i]), textcoords="offset points", xytext=(-10,5), ha='center')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (hp={hp}) {classe}')
    plt.legend()
    plt.grid()
    plt.savefig(f"{path}PR_Curve_hp={hp}_{year}_{classe}.png")
    plt.close()