import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random 
import cv2 
#from inception_time import InceptionTimeModel
import matplotlib.colors as colors
import matplotlib.cm as cmx

class GradCAM:
    def __init__(self, model, block_idx):
        self.model = model
        self.target_layer = model.inception_blocks[block_idx].inception_3
        self.gradients = None
        self.activations = None 

        # Register a backward hook to capture gradients
        self.target_layer.register_full_backward_hook(self.save_gradient)
        # Register a forward hook to capture the activations
        self.target_layer.register_forward_hook(self.save_activation)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  # Sauver le gradient
        #print(self.gradients)
        #print("Gradients shape:", self.gradients.shape)

    def save_activation(self, module, input, output):
        self.activations = output  # Sauver l'activation
        #print("Activations shape:", self.activations.shape) 
        #print(self.activations)

    def __call__(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)
        
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
        gradcam_map = F.interpolate(gradcam_map.unsqueeze(0), size=input_tensor.shape[2:], mode='linear', align_corners=False)
        heatmap = gradcam_map.squeeze().cpu().detach().numpy()

        return heatmap






class GradCAM_per_target:
    def __init__(self, model, block_idx):
        self.model = model
        self.target_layer = model.inception_blocks[block_idx].inception_3
        self.gradients = None
        self.activations = None 

        # Register a backward hook to capture gradients
        self.target_layer.register_full_backward_hook(self.save_gradient)
        # Register a forward hook to capture the activations
        self.target_layer.register_forward_hook(self.save_activation)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]  


    def save_activation(self, module, input, output):
        self.activations = output  

    def __call__(self, input_tensor, target_day):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor) 
        
        #extraire la sortie du jour cible
        target_output = output[:, target_day]

        #backward pass pour ce jour uniquement
        self.model.zero_grad()
        target_output.sum().backward(retain_graph=True)

        #pondérer les activations par les gradients
        weights = self.gradients.mean(dim=2, keepdim=True)  
        gradcam_map = F.relu((weights * self.activations).sum(dim=1))  

        # Upsample la carte Grad-CAM pour correspondre à l'entrée
        gradcam_map = F.interpolate(
            gradcam_map.unsqueeze(0), size=input_tensor.shape[2:], mode='linear', align_corners=False
        )
        heatmap = gradcam_map.squeeze().cpu().detach().numpy()

        return heatmap




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
        self.gradients = grad_output[0]  # Sauver le gradient
        #print(self.gradients)
        #print("Gradients shape:", self.gradients.shape)

    def save_activation(self, module, input, output):
        self.activations = output  # Sauver l'activation
        #print("Activations shape:", self.activations.shape)
        #print(self.activations)

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
                                    size=x1.shape[2:],  #here i have x1 and x2 have same length
                                    mode='linear', align_corners=False)
        heatmap = gradcam_map.squeeze().cpu().detach().numpy()

        return heatmap




#Fonction, plot avec signal original !!! 
def plot_with_heatmap_orig(sample_signal, heatmap, model, output_path, layer_name, jour_R, class_label, plot_type='both'):

    heatmap_resized = cv2.resize(heatmap, ( 1, sample_signal.shape[-1])) 
    heatmap_resized = heatmap_resized.T  

    if plot_type in ['both', 'subplot']:
        filename = f'{output_path}/with_orig/hp={jour_R}_{class_label}_subplot_{model.__class__.__name__}_{layer_name}_sign_orig.png'
        plt.figure(figsize=(12, 4)) 
        
        #plot du signal
        plt.subplot(1, 2, 1)
        plt.plot(sample_signal, label='Signal') 
        plt.title(f'Signal Original - Class: {class_label}')
        plt.xlabel('Temps')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        
        #plot de la heatmap
        ax_heatmap = plt.subplot(1, 2, 2)
        im = ax_heatmap.imshow(heatmap_resized, aspect='auto', cmap='jet', vmin=0, vmax=heatmap.max())
        ax_heatmap.set_title(f'Heatmap - Class: {class_label}')
        plt.xlabel('Temps')
        plt.grid()
        
        cbar = plt.colorbar(im, ax=ax_heatmap, label='Importance')
        cbar.ax.set_ylabel('Importance')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


def plot_with_heatmap(sample_signal, heatmap, model, output_path, layer_name, jour_R,  class_label, plot_type='both'):
    #verifier taille du heatmap et redimensionnement
    #print(sample_signal.shape) 
    #heatmap_resized = cv2.resize(heatmap, (sample_signal.shape[2],1))  # Adapte selon la taille du signal
    heatmap_ = cv2.resize(heatmap, (1, sample_signal.shape[2]))
    #print( "heatmap_resized = ", heatmap_resized.shape) 
    heatmap_resized = heatmap_.T  #bech tkoun taille (1,336)
    #print( "heatmap_resized.T = ", heatmap_resized.shape) 
    # Assurez-vous que `sample_signal` est un tableau 2D : [time_steps, amplitude]
    if plot_type in ['both', 'subplot']:
        filename = f'{output_path}hp={jour_R}_{class_label}_subplot_{model.__class__.__name__}_{layer_name}.png'
        plt.figure(figsize=(8, 3))
        
        # Signal
        plt.subplot(1, 2, 1)
        #pour plotter signal, il faute de taille (336,)
        plt.plot(sample_signal[0][0, :], label='Signal')  # Ajuste selon la dimension
        #plt.title(f'Signal - Class: {class_label}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid()
        
        #heatmap
        ax_heatmap = plt.subplot(1, 2, 2)
        im = ax_heatmap.imshow(heatmap_resized, aspect='auto', cmap='jet', vmin=0, vmax=heatmap.max())
        #plt.title(f'Heatmap - Class: {class_label}')
        plt.xlabel('Time')
        plt.grid()
        cbar = plt.colorbar(im, ax=ax_heatmap, label='Importance')
        cbar.ax.set_ylabel('Importance') 
        plt.savefig(filename, bbox_inches='tight')

    if plot_type in ['both', 'overlay']:
        filename = f'{output_path}hp={jour_R}_{class_label}_overlay_{model.__class__.__name__}_{layer_name}.png'
        plt.figure(figsize=(8, 3))
        
        #trace avec le heatmap comme couleur
        jet = cm = plt.get_cmap('jet') 
        cNorm  = colors.Normalize(vmin=0, vmax=heatmap.max())
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

        for i in range(len(sample_signal[0][0, :]) - 1):
            colorVal = scalarMap.to_rgba(heatmap_resized.flatten()[i])
            plt.plot([i, i + 1], [sample_signal[0][0, i], sample_signal[0][0, i + 1]], 
                     color=colorVal , linewidth=2.5)
                     #color=plt.cm.jet(heatmap_resized.flatten()[i]), linewidth=2.5)

        #plt.title(f'Signal Grad-CAM Overlay - Class: {class_label}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid()

        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=heatmap.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), label='Importance')  
        plt.savefig(filename, bbox_inches='tight')
        plt.close()




'''
# Define model and inputs
in_channels = 1      # Number of input features
num_classes = 2      # Number of output classes
n_blocks = 2         # Number of Inception blocks
model = InceptionTimeModel(in_channels, num_classes, n_blocks)
print(model.inception_blocks[-1].inception_3)
# Create random input tensor
input_tensor = torch.randn(1, in_channels, 336)  # Example with batch size 1 and sequence length 336

# Initialize GradCAM with the model and target block index
gradcam = GradCAM(model, block_idx=1)

# Run GradCAM to get the heatmap
heatmap = gradcam(input_tensor)
print("Shape of heatmap:", heatmap.shape)


# Créer une image de votre heatmap Grad-CAM
plt.imshow(heatmap[np.newaxis, :], cmap='jet', aspect='auto', interpolation='nearest')
#plt.imshow(heatmap, cmap='viridis')

plt.colorbar()
plt.title("Grad-CAM Heatmap")
plt.savefig("/mnt/SSD1/bouazizs/GradCam/gradcam_heatmap.png")
plt.close()

# Utiliser la fonction pour visualiser le signal et la heatmap
output_path = "/mnt/SSD1/bouazizs/GradCam/.plot_gradcam/"
plot_with_heatmap(input_tensor.numpy(), heatmap, model, output_path, "inception_block_1", "Class 1", plot_type='both')
'''
