import torch
import matplotlib.pyplot as plt
import os
import pickle
from collections import Counter
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np 
from SparseAutoencoder import SparseAutoencoder
from data_utils.datasets import get_data_loader
from models.avit import build_avit
from utils.YParams import YParams

if __name__ == '__main__':
	# Pass in first snapshot of 10 trajectories and collect activations
	filename = 'analyze_activations.pkl'
	# Initialize activations list
	activations = []
	# Check if the file exists before loading
	if os.path.exists(filename):
		with open(filename, 'rb') as f:
			activations = pickle.load(f)
	else:
		print(f"{filename} does not exist, recreating activations")

	# Define a hook function to capture activations after the attention layer
	def hook_fn(module, input, output):
		print("Hook activated. Captured output shape:", output.shape)
		# Detach and move to CPU to avoid memory issues
		activations.append(output.detach().cpu())

	if len(activations) == 0:
		# Load model parameters
		params = YParams(os.path.abspath("./config.yaml"), "basic_config", True)

		print('params', params)

		# Build the model
		model = build_avit(params)

		# Load the pretrained checkpoint
		checkpoint = torch.load('./MPP_AViT_Ti', map_location=torch.device('cpu'))

		model.load_state_dict(checkpoint)
		model.eval()

		def print_hook(module, input, output):
			print(f"Layer: {module.__class__.__name__}, Output shape: {output.shape}")


		
		# **Locate the Spatial Attention Layer in blocks.8.spatial**
        # Access block number 8's spatial component
		spatial_attention_layer = model.blocks[8].spatial

		# Register the hook on the attention layer output
		spatial_attention_layer.register_forward_hook(hook_fn)

		train_data_loader, train_dataset, train_sampler = get_data_loader(
            params, params.train_data_paths,
            False, split='train', rank=0, train_offset=params.embedding_offset
        )
	
		print("GOT DATA")

		with torch.no_grad():
			print("Total number of batches:", len(train_data_loader))
			# time_step_count = 0  # Counter to track total time steps plotted

			# previous_file_index = None  # Track the previous file index

			# for batch_idx, data in enumerate(train_data_loader):
			# 	if time_step_count >= 33:
			# 		break

			# 	inp, file_index, field_labels, bcs, tar = data

			# 	for step in range(16):
			# 		if time_step_count >= 33:
			# 			break

			# 		grid = inp[0, step, 0, :, :].numpy()

			# 		plt.imshow(grid, cmap='viridis')
			# 		plt.colorbar(label='Water Depth')
			# 		title = f"Time Step {time_step_count + 1}, Step within Trajectory {step + 1}"
			# 		plt.title(title)
			# 		plt.xlabel("Grid X")
			# 		plt.ylabel("Grid Y")
			# 		plt.show()

			# 		time_step_count += 1
			for batch_idx, data in enumerate(train_data_loader):
				if batch_idx >= 10:
					break 
				inp, file_index, field_labels, bcs, tar = data
				print('input shape', inp.shape)
				first_snapshot = inp[:, 0, :, :, :].unsqueeze(1) # Selecting the first time step (shape should now be [5, x, y, z])
				print("First snapshot new shape:", first_snapshot.shape)
				output = model(inp, field_labels, bcs)  # Run the batch through the model
				print('output shape', output.shape)
				print("Captured activations for single snapshot")

		# Save activations
		print("Captured activations from attention layer.")
		with open(filename, 'wb') as f:
			pickle.dump(activations, f)

	# **Process Activations**
	# Concatenate activations into a single tensor
	activations_tensor = torch.cat(activations, dim=0)
	print("Final Activations shape:", activations_tensor.shape)

	# Flatten activations for training
	activations_tensor_flat = activations_tensor.view(activations_tensor.size(0), -1)  # Shape: [80, 192 * 8 * 8]
	print("Flattened Activations shape:", activations_tensor_flat.shape)

	# Pass in activations into now trained SAE
	input_size = activations_tensor_flat.shape[1]
	hidden_size = 128  # size for the latent space
	autoencoder = SparseAutoencoder(input_size, hidden_size)
	print(f"Initialized Sparse Autoencoder with input size: {input_size}, hidden size: {hidden_size}")
	# Load in saved weights
	autoencoder.load_state_dict(torch.load('best_sae_restful-terrain-11.pth'))

	# Save the encoded features per trajectory
	encoded_features_per_trajectory = []

	with torch.no_grad():
		for i in range(activations_tensor_flat.shape[0]):  # Iterate over flattened activations
			activations_sample = activations_tensor_flat[i].unsqueeze(0)  # Shape: [1, 12288]
			encoded = autoencoder.encoder(activations_sample)
			encoded_features_per_trajectory.append(encoded)
			if i == 3:
				print("Activation tensor sample shape:", activations_tensor_flat.shape)
				print("Activations sample shape:", activations_sample.shape)
				print("Encoded features shape:", encoded.shape)

    # Save encoded features
	encoded_filename = 'encoded_features_for_ten_ics.pt'
	torch.save(encoded_features_per_trajectory, encoded_filename)
	print("Encoded features per trajectory saved.")

	# Analyze the encoded features per trajectory

	# Load encoded features per prompt
	encoded_features = torch.load(encoded_filename)

	for encoded in encoded_features:
		print('encoded', encoded)

	# Convert the encoded features to a matrix where each row is a prompt and each column is a feature
	feature_matrix = torch.stack([encoded.max(dim=0)[0] for encoded in encoded_features])

	plt.figure(figsize=(12, 8))
	sns.heatmap(feature_matrix, cmap='viridis', xticklabels=False)
	plt.xlabel('Feature Index')
	plt.ylabel('Trajectory Index')
	plt.title('Feature Activation Heatmap')
	plt.show()
	# plt.savefig('feature_activation_heatmap.png')

	# # Determine the number of prompts in the feature matrix
	# n_samples = feature_matrix.shape[0]
	# # Set the perplexity to be smaller than the number of prompts
	# perplexity = min(30, n_samples - 1)  

	# # Apply t-SNE with adjusted perplexity
	# tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
	# reduced_features = tsne.fit_transform(feature_matrix)

	# # Plot the reduced features
	# plt.figure(figsize=(10, 6))
	# plt.scatter(reduced_features[:, 0], reduced_features[:, 1])
	# plt.xlabel('t-SNE Dimension 1')
	# plt.ylabel('t-SNE Dimension 2')
	# plt.title('t-SNE Visualization of Prompts in Feature Space')
	# plt.savefig('tsne_visualization.png')

	# # Plot histograms for a subset of features 
	# num_features_to_plot = 10
	# plt.figure(figsize=(15, 10))
	# for i in range(num_features_to_plot):
	# 	plt.subplot(2, 5, i + 1)
	# 	plt.hist(feature_matrix[:, i], bins=20, color='skyblue')
	# 	plt.title(f'Feature {i} Activation')
	# 	plt.xlabel('Activation Level')
	# 	plt.ylabel('Frequency')
	# plt.tight_layout()
	# plt.savefig('feature_activation_histograms.png')

	# # Plot activations for a specific feature over token positions for a specific prompt
	# feature_index = 5
	# prompt_index = 2
	# activations = encoded_features[prompt_index].numpy()[:, feature_index]

	# plt.figure(figsize=(8, 5))
	# plt.plot(activations)
	# plt.xlabel('Token Position')
	# plt.ylabel('Feature Activation')
	# plt.title(f'Feature {feature_index} Activation Over Token Positions (Prompt {prompt_index})')
	# plt.savefig('feature_activation_over_token_positions.png')

	# tokenized_prompts = torch.load('tokenized_prompts.pt') 
	# tokenizer = Tokenizer.get_instance()  

	# feature_index = 1
	# # Define a threshold for high activation
	# high_activation_threshold = 0  

	# # Identify prompts where feature has high activation
	# high_activation_prompts = []
	# for idx, encoded in enumerate(encoded_features):
	# 	# encoded is a tensor of shape (seq_len, hidden_size)
	# 	# Extract feature across all tokens
	# 	feature_activations = encoded[:, feature_index].cpu().numpy()
	# 	max_token_activation = torch.max(torch.abs(encoded[:, feature_index])).item()
	# 	print(f"Prompt {idx}, Max Activation for Feature {feature_index}: {max_token_activation}")
	# 	# Print all activations for the prompt for feature
	# 	print(f"All activations for feature {feature_index} in prompt {idx}: {feature_activations}")
	# 	# Check if any token in the prompt has a high activation for feature
	# 	if (feature_activations > high_activation_threshold).any():
	# 		high_activation_prompts.append((idx, feature_activations))

	# print(f'High activation prompts for feature {feature_index}:', high_activation_prompts)
	# # Examine tokens associated with high activation
	# for idx, activations in high_activation_prompts:
	# 	print(f"Analyzing Prompt {idx}:")
	# 	print(f"Undecoded Prompt: {tokenized_prompts[idx]}")
	# 	print(f"Original Prompt: {tokenizer.decode(tokenized_prompts[idx])}")
	# 	# Iterate through the token IDs and match with activations directly
	# 	token_ids = tokenized_prompts[idx]
	# 	if len(token_ids) != len(activations):
	# 		print(f"Warning: Number of tokens ({len(token_ids)}) does not match the number of activation values ({len(activations)}).")
	# 	high_activation_tokens = []
	# 	for i, token_id in enumerate(token_ids):
	# 		token = tokenizer.decode([token_id])
	# 		activation_value = activations[i] if i < len(activations) else None
	# 		if activation_value > 0:
	# 			high_activation_tokens.append((i, token, activation_value))

	# 	print(f"Tokens with high activation for feature {feature_index}:")
	# 	for position, token, activation_value in high_activation_tokens:
	# 		print(f"Token: '{token}' at position {position} with activation value: {activation_value}")