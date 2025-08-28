import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random 
import cv2 
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os


class MultiBranchGradCAM_2:
    def __init__(self, model, target_branch):
        """ param target_branch: 'branch_1', 'branch_2'"""

        self.model = model
        self.target_branch = target_branch
        self.gradients = None
        self.activations = None

        if target_branch == 'branch_1':
            #Target last InceptionBlock in branch_1
            self.target_layer = model.branch_1[-1].inception_3
        elif target_branch == 'branch_2':
            #Target last InceptionBlock in branch_2
            self.target_layer = model.branch_2[-1].inception_3


        # Register a backward hook to capture gradients
        self.target_layer.register_full_backward_hook(self.save_gradient)
        # Register a forward hook to capture the activations
        self.target_layer.register_forward_hook(self.save_activation)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  

    def save_activation(self, module, input, output):
        self.activations = output 

    def __call__(self, x1, x2, target_class=None):
        # Forward pass
        output = self.model(x1, x2)

        # If target_class is not specified, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1)

        # Backward pass to compute gradients with respect to the target class
        self.model.zero_grad()

        # For each item in the batch, compute the loss with respect to its predicted target_class
        class_loss = output[torch.arange(output.size(0)), target_class].sum()
        class_loss.backward(retain_graph=True)

        # Compute the Grad-CAM heatmap
        weights = self.gradients.mean(dim=2, keepdim=True)
        gradcam_map = F.relu((weights * self.activations).sum(dim=1))

        # Upsample heatmap to match input tensor size
        gradcam_map = F.interpolate(gradcam_map.unsqueeze(0),
                                    size=x1.shape[2:],  # x1 and x2 have same length
                                    mode='linear', align_corners=False)
        heatmap = gradcam_map.squeeze().cpu().detach().numpy()

        return heatmap




def plot_gradcam_heatmaps_averaged(modality, avg_gradcam_dict, classee, path_save, pred):
    all_heatmaps = []
    for idx in avg_gradcam_dict:
        if modality in avg_gradcam_dict[idx]:
            heatmap = avg_gradcam_dict[idx][modality]
            all_heatmaps.append(heatmap.cpu().numpy())  # If it is a tensor

    if not all_heatmaps:
        print(f"No heatmap found for modality {modality}.")
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

