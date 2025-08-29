import numpy as np
import pandas as pd
import time 
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from gradcam_Inception.InceptionTime_attention import InceptionTimeModel 
from gradcam_Inception.utils import  DualInputDataset, process_rockfall_data_multi, charge, preprocess_segments, plot_PR_gkf_per_year_val, convert_ndarray
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch.utils.data as data
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import f1_score, recall_score, precision_score , balanced_accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import numpy as np
import seaborn as sns



# To precise InceptionTime model with attention gate or not
use_attention=False

# Load df of RR1 and  sismo events
df_sismo= pd.read_csv('.data/ev_sismo2.csv')
df_RR1= pd.read_csv('.data/1.0_RR1.csv', delimiter='\t')

# Obtain water charge Dataframe 
df_precip = pd.read_csv('.data/data_meteo.csv', delimiter=';')
df_precip['AAAAMMJJHH'] = df_RR1['AAAAMMJJHH'].values
df_precip['AAAAMMJJHH'] = pd.to_datetime(df_precip['AAAAMMJJHH'])
lamb=0.2
df_precip['H'] = charge(df_precip,lamb) 


start_time = time.time()

results_cross_validation = []
for hp in [1, 3, 7]:
    path= f'testing_models/HP_{hp}/'
    path_curves = f'testing_models/HP_{hp}/curves/'
    # Create both folders if they don't exist
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_curves, exist_ok=True)

    print(f"\nrunning model with hp = {hp}")

    df_1 = process_rockfall_data_multi(df_RR1, df_sismo, x_days=hp, col_name='RR1',  concat_with_R=True, jour_pluie=14)
    df_1['sample_id'] = df_1.index
    # Water charge H
    df_2 = process_rockfall_data_multi(df_precip, df_sismo, x_days=hp, col_name='H', concat_with_R=False, jour_pluie=14)
    df_2['sample_id'] = df_2.index

    df = pd.merge(df_2, df_1, on='sample_id')
    df = df.drop(columns=['sample_id'])

    # Set index of df AAAAMMJJHH (to use it for year colmun)
    df.set_index('AAAAMMJJHH', inplace=True)
    # Generate a column of years as groups
    df['year'] = pd.to_datetime(df.index).year  #year of rockfall
    df = df[(df['year'] != 2013) & (df['year'] != 2024)] # Remove both 2013, as it is an incomplete year, and 2024, which is reserved for blind testing

    X_1 = df[[f"RR1_hour_{i}" for i in range(336)]].values  # Colmuns of Rain (windows of 14*24)
    X_2 = df[[f"H_hour_{i}" for i in range(336)]].values    # Colmuns of water charge (windows of 14*24)

    x1_scaled, scaler1 = preprocess_segments(X_1) 
    x2_scaled, scaler2 = preprocess_segments(X_2)    # shape (n_samp, 336, 1) 

    y = df['rockfall_target'].values
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Conversion to pytorch tensors
    x1_tensor = torch.FloatTensor(x1_scaled).permute(0, 2, 1)   # shape (n_samp, 1,336)
    x2_tensor = torch.FloatTensor(x2_scaled).permute(0, 2, 1)   # shape (n_samp, 1,336)
    y_tensor = torch.LongTensor(y_encoded)

    groups = df['year'].values  # Groups by years
    unique_years = np.unique(groups)
    gkf = GroupKFold(n_splits=len(unique_years))  # Number of groups = unique years
    
    # Select computation device: use GPU ("cuda") if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Class weights 
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weight_tensor = torch.FloatTensor(class_weights).to(device)


    # For saving  performances
    fold_results = []
    all_test_metrics = {}
    all_val_metrics={}
    all_train_metrics={}
    all_test_results = []  #

    # Groupkfold : 
    for fold, (train_val_idx, test_idx) in enumerate(gkf.split(x1_tensor, y_encoded, groups=groups)):
        print(f"\n--- fold {fold + 1} ---")

        # Split data into train+val and test sets for dual input 
        x1_train_val, x1_test = x1_tensor[train_val_idx], x1_tensor[test_idx]
        x2_train_val, x2_test = x2_tensor[train_val_idx], x2_tensor[test_idx]
        y_train_val, y_test = y_tensor[train_val_idx], y_tensor[test_idx]
        groups_train_val = groups[train_val_idx]
        
        test_year = np.unique(groups[test_idx])
        remaining_years = np.setdiff1d(unique_years, [test_year])

        # Select the validation year from the remaining years
        print('test_year: ', test_year)

        # Create a folder test_year to save the 9 models 
        test_year_folder = os.path.join(path, f'test_year_{test_year[0]}')
        os.makedirs(test_year_folder, exist_ok=True)

        test_metrics=[] 
     
        for val_year in remaining_years :
            print('val_year: ', val_year)
            remaining_train_year= np.setdiff1d(remaining_years, [val_year])

            train_idx = np.where(groups_train_val != val_year)[0]
            val_idx = np.where(groups_train_val == val_year)[0]
            
            x1_train, x1_val = x1_train_val[train_idx], x1_train_val[val_idx]
            x2_train, x2_val = x2_train_val[train_idx], x2_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            # Load data with dataloader
            bs=32
            train_dataset = DualInputDataset(x1_train, x2_train, y_train)
            val_dataset = DualInputDataset(x1_val, x2_val, y_val)
            test_dataset = DualInputDataset(x1_test, x2_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
        
            # Configuration model with dual input for this fold
            if use_attention :  # model with attention
                model = InceptionTimeModel( in_channels_1=1, in_channels_2=1, num_classes=len(np.unique(y_encoded)), n_blocks=2, merge_mode='concat', use_attention=True)
            else :
                model = InceptionTimeModel( in_channels_1=1, in_channels_2=1, num_classes=len(np.unique(y_encoded)), n_blocks=2, merge_mode='concat', use_attention=False)
                
            model = model.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            train_losses = []
            train_accuracies = []
            train_precisions = []
            train_recalls = []
            train_precision_class_0 = []
            train_recall_class_0 = []
            train_precision_class_1 = []
            train_recall_class_1 = []

            val_losses = []
            val_accuracies = []
            train_accuracies = []
            val_f1_scores = []
            val_recalls = []
            val_precision = []
            val_precision_class_0 = []
            val_recall_class_0 = []
            val_f1_class_0 = []
            val_precision_class_1 = []
            val_recall_class_1 = []
            val_f1_class_1 = []

            # Initialize each test metrics dict to save metrics for a test year and a specific val_year
            test_metrics_per_model={}

            # Early stopping parameters
            patience = 35  
            best_val_loss = float('inf')  
            counter = 0 
            epochs_no_improve=0

            #------------------------------------Training for each fold
            nb_epochs=200
            for epoch in range(nb_epochs):
                model.train()
                train_loss = 0
                correct = 0
                total = 0
                all_train_targets = []
                all_train_predictions = []

                for (data1,data2), target in train_loader: 
                    optimizer.zero_grad()
                    # Pass both inputs to the model
                    output = model(data1.to(device), data2.to(device))
                    loss = criterion(output, target.to(device))
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(output , 1)
                    total += target.size(0)
                    correct += (predicted == target.to(device)).sum().item()
                    # Save the truth and the predections 
                    all_train_targets.extend(target.cpu().numpy())
                    all_train_predictions.extend(predicted.cpu().numpy())


                # Metrics for the training set
                precision_train = precision_score(all_train_targets, all_train_predictions, average='weighted',zero_division=0)
                recall_train = recall_score(all_train_targets, all_train_predictions, average='weighted',zero_division=0)

                class_0_correct = np.sum((all_train_targets == 0) & (all_train_predictions == 0))
                class_0_total = np.sum(all_train_targets == 0)
                class_1_correct = np.sum((all_train_targets == 1) & (all_train_predictions == 1))
                class_1_total = np.sum(all_train_targets == 1)

                precision_train_per_class = precision_score(all_train_targets, all_train_predictions, average=None,zero_division=0)
                recall_train_per_class = recall_score(all_train_targets, all_train_predictions, average=None,zero_division=0)

                train_accuracies.append(correct / total)
                train_losses.append(train_loss / len(train_loader))
    
                
                #---------------------------------------Validation-----------------------------------------------------
                model.eval()
                val_loss = 0
                correct = 0
                total = 0
                all_val_targets = []
                all_val_predictions = []

                with torch.no_grad():
                    for (batch1,batch2), target in val_loader:
                        output = model(batch1.to(device), batch2.to(device))
                        loss = criterion(output, target.to(device))
                        val_loss += loss.item()
                        _, predicted = torch.max(output, 1)
                        total += target.size(0)
                        correct += (predicted == target.to(device)).sum().item()
                        # Save targets and predictions for class-wise metrics
                        all_val_targets.extend(target.cpu().numpy())
                        all_val_predictions.extend(predicted.cpu().numpy())

     
                # Calculate class metrics
                precision = precision_score(all_val_targets, all_val_predictions, average='weighted',zero_division=0)
                recall = recall_score(all_val_targets, all_val_predictions, average='weighted',zero_division=0)
                f1_score_metric = f1_score(all_val_targets, all_val_predictions, average='weighted',zero_division=0)

                precision_per_class = precision_score(all_val_targets, all_val_predictions, average=None,zero_division=0)
                recall_per_class = recall_score(all_val_targets, all_val_predictions, average=None,zero_division=0)
                f1_per_class = f1_score(all_val_targets, all_val_predictions, average=None,zero_division=0)

                all_val_targets = np.array(all_val_targets)
                all_val_predictions = np.array(all_val_predictions)
                class_0_correct = np.sum((all_val_targets == 0) & (all_val_predictions == 0))
                class_0_total = np.sum(all_val_targets == 0)
                class_1_correct = np.sum((all_val_targets == 1) & (all_val_predictions == 1))
                class_1_total = np.sum(all_val_targets == 1)

                # Save global metrics
                val_losses.append(val_loss / len(val_loader))
                val_accuracies.append(correct / total)
                val_precision.append(precision)
                val_recalls.append(recall)
                val_f1_scores.append(f1_score_metric)
                # Save per class metrics
                val_precision_class_0.append(precision_per_class[0])
                val_recall_class_0.append(recall_per_class[0])

                val_precision_class_1.append(precision_per_class[1])
                val_recall_class_1.append(recall_per_class[1])

                print(f'epoch [{epoch + 1}/{nb_epochs}], train loss: {train_loss / len(train_loader):.4f}, '
                    f'val loss: {val_loss / len(val_loader):.4f}, val accuracy: {correct / total:.4f},  recall: {recall:.4f}, '
                    f'val class 0 recall: {recall_per_class[0]:.4f}, val class 1 recall: {recall_per_class[1]:.4f}'
                    )
            
                num_epochs_trained = epoch + 1 

                # Early stopping logic per val_year
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_state = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1  

                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break


            #plot_training_curves_gkf(train_losses, val_losses, train_accuracies, val_accuracies, path_curves, hp,test_year[0], val_year )

            val_metrics ={
                'val_losses': np.mean(val_losses),
                "F1_scores" : np.mean(val_f1_scores),
                'val_accuracies': np.mean(val_accuracies),
                'val_recalls': np.mean(val_recalls),
                'val_precision': np.mean(val_precision),
                'val_precision_class_0': np.mean(val_precision_class_0),
                'val_recall_class_0': np.mean(val_recall_class_0),
                'val_f1_class_0': np.mean(val_f1_class_0),
                'val_precision_class_1': np.mean(val_precision_class_1),
                'val_recall_class_1': np.mean(val_recall_class_1),
                'val_f1_class_1': np.mean(val_f1_class_1),
            }

            # Save each model at each fold of test year 
            model_path = os.path.join(test_year_folder, f'val_{val_year}.pt')
            torch.save( model_state, model_path)

            metrics_filename = os.path.join(test_year_folder, f'metrics_val_year{val_year}.json')
            with open(metrics_filename, 'w') as f:
                json.dump(val_metrics, f, indent=4)

            #--------------------------------Testing
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            all_test_targets = []
            all_test_predictions = []
            all_test_probs = []  # To store the predicted probabilities

            with torch.no_grad():
                for (data1,data2), target in test_loader:
                    output = model(data1.to(device), data2.to(device))
                    loss = criterion(output, target.to(device))
                    test_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    probs = torch.softmax(output, dim=1) 
                    total += target.size(0)
                    correct += (predicted == target.to(device)).sum().item()

                    # Save targets and predictions
                    all_test_targets.extend(target.cpu().numpy())
                    all_test_predictions.extend(predicted.cpu().numpy())
                    all_test_probs.extend(probs.cpu().numpy())#[:, 1])

            all_test_targets = np.array(all_test_targets)
            all_test_predictions = np.array(all_test_predictions)
            all_test_probs = np.array(all_test_probs) 


            # Calculate metrics for test 
            test_accuracy = correct / total            
            test_precision = precision_score(all_test_targets, all_test_predictions, average='weighted', zero_division=0)
            test_f1 = f1_score(all_test_targets, all_test_predictions, average='weighted', zero_division=0)
            test_recall = recall_score(all_test_targets, all_test_predictions, average='weighted', zero_division=0)

            test_precision_per_class = precision_score(all_test_targets, all_test_predictions, average=None, zero_division=0)
            test_recall_per_class = recall_score(all_test_targets, all_test_predictions, average=None, zero_division=0)
            test_f1_per_class = f1_score(all_test_targets, all_test_predictions, average=None, zero_division=0)

            all_test_targets = np.array(all_test_targets)
            all_test_predictions = np.array(all_test_predictions)
            test_class_0_correct = np.sum((all_test_targets == 0) & (all_test_predictions == 0))
            test_class_0_total = np.sum(all_test_targets == 0)
            test_class_1_correct = np.sum((all_test_targets == 1) & (all_test_predictions == 1))
            test_class_1_total = np.sum(all_test_targets == 1)
            
            conf_matrix = confusion_matrix(all_test_targets, all_test_predictions)
            
            precision_0, recall_0, _0 = precision_recall_curve(all_test_targets == 0, all_test_probs[:, 0])
            precision_1, recall_1, _1 = precision_recall_curve(all_test_targets == 1, all_test_probs[:, 1])
            pr_auc_0 = auc(recall_0, precision_0)
            pr_auc_1 = auc(recall_1, precision_1)
            plot_PR_gkf_per_year_val(precision_0, recall_0, pr_auc_0 , _0, test_year_folder , val_year, 'classe 0')
            plot_PR_gkf_per_year_val(precision_1, recall_1, pr_auc_1 , _1, test_year_folder, val_year, 'classe 1')


            test_metrics_per_model={
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_precision_class_0': test_precision_per_class[0],
                'test_recall_class_0': test_recall_per_class[0],
                'test_f1_class_0': test_f1_per_class[0],
                'test_precision_class_1': test_precision_per_class[1],
                'test_recall_class_1': test_recall_per_class[1],
                'test_f1_class_1': test_f1_per_class[1],
                'confusion_matrix': conf_matrix.tolist()
            }
            
            test_metrics.append(test_metrics_per_model)

            print(f'test loss: {test_loss / len(test_loader):.4f}, test accuracy: {test_accuracy:.4f}, '
                f'test f1-score: {test_f1:.4f}, test recall: {test_recall:.4f}, '
                f'test class 0 recall: {test_recall_per_class[0]:.4f}, test class 1 recall: {test_recall_per_class[1]:.4f}'
                )
            
            metrics_test_filename = os.path.join(test_year_folder, f'test_metrics_val_year{val_year}.json')
            with open(metrics_test_filename, 'w') as f:
                json.dump(test_metrics_per_model, f, indent=4)



        # Areaged metrics across all 9 models for each test year
        avg_metrics = {
            'test_year': test_year,
            'avg_test_accuracy': np.mean([r['test_accuracy'] for r in test_metrics]),
            'avg_test_precision': np.mean([r['test_precision'] for r in test_metrics]),
            'avg_test_recall': np.mean([r['test_recall'] for r in test_metrics]),
            'avg_test_F1_score': np.mean([r['test_f1'] for r in test_metrics]),
            'avg_test_precision_class_0': np.mean([r['test_precision_class_0'] for r in test_metrics]),
            'avg_test_precision_class_1': np.mean([r['test_precision_class_1'] for r in test_metrics]),
            'avg_test_recall_class_0': np.mean([r['test_recall_class_0'] for r in test_metrics]),
            'avg_test_recall_class_1': np.mean([r['test_recall_class_1'] for r in test_metrics]),
        }
        
        all_test_results.append(avg_metrics)


    # Conversion of all_test_results before save 
    all_test_results = convert_ndarray(all_test_results)
    # Save
    metrics_filename = f'{path}metrics_hp_{hp}_all_test_years_averaged.json'
    with open(metrics_filename, 'w') as f:
        json.dump(all_test_results, f, indent=4)

    print(f"All test metrics saved at {metrics_filename}")
    end_time = time.time()
    print("execution time: ", end_time - start_time )

