from gradcam_Inception.inception_time_multi import InceptionTimeModel
from gradcam_Inception.utils import  process_rockfall_data_multi, charge, preprocess_segments
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

#for attention model!!!
from gradcam_Inception.inception_attention import InceptionTimeModel


vote = 'HARD' #'SOFT'  # 
weighted=False   #if weighted by precision 

df_sismo= pd.read_csv('.data/events/ev_sismo2_2024.csv', delimiter=',')
df_RR1= pd.read_csv('.data/events/1.0_RR1_2024.csv', delimiter=',')
#charge/discharge H 
df_precip = pd.read_csv('.data/events/data_meteo_Sabrine2_2024.csv', delimiter=',')
df_precip['AAAAMMJJHH'] = df_RR1['AAAAMMJJHH'].values
df_precip['AAAAMMJJHH'] = pd.to_datetime(df_precip['AAAAMMJJHH'])
lamb=0.2
df_precip['H'] = charge(df_precip,lamb) 
hp=1

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

#initilize lists
all_preds = []
all_probs=[]
all_weighted_probs=[]
w1_all=[]
for test_year in [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]:
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
            num_classes=2, merge_mode='concat' )

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        #loadweights into model
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        #load precision metric
        with open(metric_path, 'r') as f:
            metrics = json.load(f)
        w0 = metrics["test_precision_class_0"]
        w1 = metrics["test_precision_class_1"]
        w1_all.append(w1)

        with torch.no_grad():
            outputs = model( X_tensor_rain, X_tensor_charge )
            probs = torch.softmax(outputs, dim=1).cpu().numpy()  # get probabilities
            all_probs.append(probs) #.cpu().numpy()) 
        
        #for weighted by precision 
        weighted_probs = np.zeros_like(probs)
        weighted_probs[:, 0] = probs[:, 0] * w0
        weighted_probs[:, 1] = probs[:, 1] * w1
        all_weighted_probs.append(weighted_probs)  
        
        #precitions  
        # !!!!!!!!!!!!!!!!  if weighted vote for hard Voting !!!!!!!!!!!!!!
        #preds = weighted_probs.argmax(axis=1)

        # if not weighted 
        preds = outputs.argmax(dim=1).cpu().numpy() #get predictions 
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

    if weighted : 
        print("*******************weighted**********")
        print(all_preds_array)
        weights = np.array(w1_all)
        print(w1_all)
        #weighted vote only for class 1
        weighted_votes = (all_preds_array == 1) * weights[:, np.newaxis]
        votes_for_class_1 = np.sum(weighted_votes > 0 , axis=0) 
        final_preds_majority = (votes_for_class_1 > 90 / 2).astype(int)

#----------------------------Majority SOFT vote --------------------------------
if vote == "SOFT" :
    all_probs_array = np.stack(all_probs)   # if not weighted voting
    if weighted : 
        all_probs_array = np.stack(all_weighted_probs)  #  (90, 351, 2) #weighted

    mean_probs = np.mean(all_probs_array, axis=0)  # (351, 2)
    #soft-voting prediction
    final_preds_majority = np.argmax(mean_probs, axis=1) 
    print(y_tensor)
    print(final_preds_majority)

#----------------------------Metrics
accuracy = accuracy_score(y_tensor, final_preds_majority)
balanced_acc = balanced_accuracy_score(y_tensor, final_preds_majority)
#weighted_acc = accuracy_score(y_tensor, final_preds_majority, sample_weight=sample_weight)

weights = compute_sample_weight(None,  y_tensor)
acc_weighted = accuracy_score(y_tensor, final_preds_majority, sample_weight=weights)



precision = precision_score(y_tensor, final_preds_majority, average="weighted",zero_division=0)
recall = recall_score(y_tensor, final_preds_majority, average="weighted",zero_division=0)
f1 = f1_score(y_tensor, final_preds_majority, average="weighted")
conf_matrix = confusion_matrix(y_tensor, final_preds_majority)

#per class metrics
precision_per_class = precision_score(y_tensor, final_preds_majority, average=None,zero_division=0)
recall_per_class = recall_score(y_tensor, final_preds_majority, average=None,zero_division=0)
f1_per_class=f1_score(y_tensor, final_preds_majority, average=None)

results.append({
"f1_score": f1,
"precision": precision,
"recall": recall,
"accuracy": accuracy,
"weighted-accuracy" : acc_weighted,
"balanced accuracy" : balanced_acc,
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


with open(os.path.join(f"testing_models_gkf_multi_branch/Rain_H_all_models_attention_shuffle/HP_{hp}/vote", f"results_vote_{vote}_{weighted}_1.json" ), "w") as f:
    json.dump(results, f, indent=4)





'''
best_f1 = 0
best_threshold = None
for threshold in range(1, 91):
    vote_result = (np.sum(all_preds_array, axis=0) >= threshold).astype(int)  
    f1 = f1_score(y_tensor, vote_result)
    recall = recall_score(y_encoded, vote_result)
    precision = precision_score(y_encoded, vote_result,zero_division=0)
    accuracy = accuracy_score(y_encoded, vote_result)

    print(f"Seuil {threshold} : F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}")
    
    results.append({
    "threshold": threshold,
    "f1_score": f1,
    "precision": precision,
    "recall": recall,
    "accuracy": accuracy
    })



    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_metrics = (f1, recall, precision)


print(f"Meilleur seuil de vote : {best_threshold}, F1 : {best_metrics[0]:.3f}, Recall: {best_metrics[1]:.3f}, Precision: {best_metrics[2]:.3f}")

df_results = pd.DataFrame(results)
df_results.to_csv("metrics_vote.csv", index=False)

'''


# plt.hist(mean_probs[:, 1], bins=50)
# plt.title("Distribution of averaged class 1 probabilities")
# plt.xlabel("P(class 1)")
# plt.ylabel("Count")
# plt.grid()
# plt.show()