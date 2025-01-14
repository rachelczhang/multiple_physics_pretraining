import os
import torch
import torch.nn as nn
import pickle
from torch.utils.data import TensorDataset, DataLoader
import numpy as np 
import matplotlib.pyplot as plt
import wandb

from SparseAutoencoder import SparseAutoencoder
from data_utils.datasets import get_data_loader
from models.avit import build_avit
from utils.YParams import YParams

wandb.init(project="sparse-autoencoder", entity="rczhang")

def load_model_and_params(device):
    # Load model parameters
    params = YParams(os.path.abspath("./config.yaml"), "basic_config", True)

    print('params', params)

    # Build the model
    model = build_avit(params)

    # Load the pretrained checkpoint
    checkpoint = torch.load('./MPP_AViT_Ti', map_location=device,  weights_only=True)

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # # **Print the Layers**
    # print("Model modules:")
    # for name, module in model.named_modules():
    #     print(name, '->', type(module))
    #     print(module)

    return model, params

def collect_activations_batched(model, data_loader, max_batches=None):
    activations = []
    
    def hook_fn(module, input, output):
        print('raw output shape', output.shape)
        activations.append(output.clone().detach().cpu())
    
    # register hook on the spatial attention layer
    spatial_attention_layer = model.blocks[10].spatial.mlp.fc2
    hook = spatial_attention_layer.register_forward_hook(hook_fn)
    
    print("Starting activation collection...")
    
    with torch.no_grad():
        # time_step_count = 0  # Counter to track total time steps plotted

        # previous_file_index = None  # Track the previous file index

        # for batch_idx, data in enumerate(data_loader):
        #     if time_step_count >= 100:
        #         break

        #     inp, file_index, field_labels, bcs, tar = data

        #     # Check if this is a new trajectory by comparing file_index
        #     is_new_trajectory = file_index != previous_file_index
        #     previous_file_index = file_index

        #     for step in range(16):
        #         if time_step_count >= 100:
        #             break

        #         grid = inp[0, step, 0, :, :].numpy()

        #         plt.imshow(grid, cmap='viridis')
        #         plt.colorbar(label='Water Depth')
        #         title = f"Time Step {time_step_count + 1}, Step within Trajectory {step + 1}"
        #         if is_new_trajectory:
        #             title += " (New Trajectory)"
        #         plt.title(title)
        #         plt.xlabel("Grid X")
        #         plt.ylabel("Grid Y")
        #         plt.show()

        #         time_step_count += 1
        print("Total number of batches:", len(data_loader))
        for batch_idx, data in enumerate(data_loader):
            if max_batches and batch_idx >= max_batches:
                break
                
            # unpack the batch
            inp, file_index, field_labels, bcs, _ = data
            inp = inp.to(device)
            field_labels = field_labels.to(device)
            bcs = bcs.to(device)
            
            # Run the batch through the model
            output = model(inp, field_labels, bcs)
            torch.cuda.empty_cache()

            inp = inp.cpu()
            field_labels = field_labels.cpu()
            bcs = bcs.cpu()

            print(f"Processed batch {batch_idx + 1}, input shape: {inp.shape}, output shape: {output.shape}")
    
    # Remove the hook
    hook.remove()
    
    # Concatenate all activations
    activations_tensor = torch.cat(activations, dim=0)
    print("Final activations shape:", activations_tensor.shape)
    
    # Flatten activations
    activations_tensor_flat = activations_tensor.view(activations_tensor.size(0), -1) # Shape: [80, 192 * 8 * 8]
    print("Flattened activations shape:", activations_tensor_flat.shape)
    
    return activations_tensor_flat

def create_new_activations(filename, device, max_batches=None):
    """
    Create new activations by loading the model and processing the data.
    
    Args:
        filename (str): Path where the activations will be saved
        
    Returns:
        torch.Tensor: Flattened activations tensor
    """
    # load model and params
    model, params = load_model_and_params(device)
    print('loaded model and params')

    # load data
    train_data_loader, train_dataset, train_sampler = get_data_loader(
        params, params.train_data_paths,
        False, split='train', rank=0, train_offset=params.embedding_offset
    )

    print("loaded data")

    # Load activations
    activations_tensor_flat = collect_activations_batched(model, train_data_loader, max_batches=max_batches)
    print('collected activations')

    torch.save(activations_tensor_flat, filename)
    return activations_tensor_flat

def load_or_create_activations(filename, device, max_batches=None):
    """
    Load activations from a .pt file if it exists, otherwise create new ones.
    
    Args:
        filename (str): activations file
        
    Returns:
        torch.Tensor: Flattened activations tensor
    """
    if os.path.exists(filename):
        print(f"loading existing activations from {filename}")
        activations_tensor_flat = torch.load(filename, weights_only=True)
        print(f"loaded activations shape: {activations_tensor_flat.shape}")
        return activations_tensor_flat
    else:
        print(f"{filename} does not exist, recreating activations")
        return create_new_activations(filename, device, max_batches=max_batches)

def train_sae(activations, autoencoder, device):
    # **Create Dataset and Dataloader**
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    print(f"Dataset size: {len(dataset)}")

    # # Variable to store the first batch for comparison
    # first_batch = None

    # # Iterate through a few batches to check for duplicates
    # for batch_idx, batch in enumerate(dataloader):
    #     print(f"Batch {batch_idx} data:")
    #     print(batch)  # Print the actual data in the batch for inspection

    #     if first_batch is None:
    #         # Store the first batch to compare with subsequent batches
    #         first_batch = batch
    #     else:
    #         # Compare current batch with the first batch
    #         is_duplicate = torch.equal(batch[0], first_batch[0])  # Check if identical to the first batch
    #         print(f"Is batch {batch_idx} identical to the first batch? {is_duplicate}")
        
    #     # Limit the number of printed batches to avoid excessive output
    #     if batch_idx >= 3:
    #         break


    # **Loss Function and Optimizer**
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    sparsity_lambda = 1e-4
    patience = 300
    best_loss = float('inf')
    epochs_without_improvement = 0

    # **Training Loop**
    num_epochs = 100000
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)

            # Forward pass
            reconstructed, encoded = autoencoder(batch)

            # Compute losses
            reconstruction_loss = criterion(reconstructed, batch)
            sparsity_loss = torch.mean(torch.abs(encoded))
            loss = reconstruction_loss + sparsity_lambda * sparsity_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder weights after optimization step
            autoencoder.normalize_decoder_weights()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
        wandb.log({"epoch": epoch+1, "loss": avg_loss, "reconstruction_loss": reconstruction_loss.item(), "sparsity_loss": sparsity_loss.item()})

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save(autoencoder.state_dict(), f"best_sae_{wandb.run.name}.pth") 
            print(f"New best model saved with loss {best_loss:.6f}")
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"Stopping early after {epoch+1} epochs due to no improvement.")
            break

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    filename = "activations.pt"

    # Load or create activations
    activations_tensor_flat = load_or_create_activations(filename, device, max_batches=500)
    print("Final activations shape:", activations_tensor_flat.shape)

    # Define the SAE
    input_size = activations_tensor_flat.shape[1]
    hidden_size = int(input_size * 1.2)
    autoencoder = SparseAutoencoder(input_size, hidden_size).to(device)
    print(f"Initialized Sparse Autoencoder with input size: {input_size}, hidden size: {hidden_size}")

    # Train the SAE
    train_sae(activations_tensor_flat, autoencoder, device)

    wandb.finish()
