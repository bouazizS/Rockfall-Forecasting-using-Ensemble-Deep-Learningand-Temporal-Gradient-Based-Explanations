
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
                r"\usepackage[T1]{fontenc}"
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

def preprocess_segments_test(segments, scaler):
    """
    Apply a scaler (already fitted on training data) to a segment matrix.
    """
    segments_scaled = scaler.transform(segments.reshape(-1, segments.shape[-1])).reshape(segments.shape)
    return segments_scaled[..., np.newaxis]  


class DualInputDataset:
    def __init__(self, x1, x2, y):
        self.x1 = x1  # 1st time series tensor [n_samples, 1, 336]
        self.x2 = x2  # 2nd time series tensor [n_samples, 1, 336]
        self.y = y    # Labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x1[idx], self.x2[idx]), self.y[idx] # Tuples


# Fonction for computing water charge from precipiation intensity
def charge(dfpluie, lamb):
    pluv = dfpluie['pluvio'].values  
    H = np.zeros_like(pluv) 
    Dt = 1 / 24  
    for p in range(1, len(pluv)):
        H[p] = H[p-1] * np.exp(-Dt / lamb) + pluv[p]
    return H


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


def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(v) for v in obj]
    else:
        return obj


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


