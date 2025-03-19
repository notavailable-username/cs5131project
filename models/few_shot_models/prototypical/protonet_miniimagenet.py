import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# MiniImageNet class definition (needed since it was imported but not defined)
class MiniImageNet(torch.utils.data.Dataset):
    """
    MiniImageNet dataset for few-shot learning
    """
    def __init__(self, root, split='train', transform=None):
        super(MiniImageNet, self).__init__()
        self.root = root
        self.split = split
        self.transform = transform
        
        # Define the path to the split file
        split_file = os.path.join(root, f"{split}.csv")
        
        # Load the data
        self.data = []
        self.targets = []
        
        # Read the CSV file
        import pandas as pd
        df = pd.read_csv(split_file)
        
        # Map unique labels to indices
        self.classes = sorted(df['label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load images and labels
        for _, row in df.iterrows():
            img_path = os.path.join(root, 'images', row['filename'])
            if os.path.exists(img_path):
                self.data.append(img_path)
                self.targets.append(self.class_to_idx[row['label']])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]
        
        # Load the image
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, target

# Define the neural network model
def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

# Function to show images
def show_images(dataset, num_samples=6):
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    for i in range(num_samples):
        image, label = dataset[i]  # Get image and label
        image = image.permute(1, 2, 0)  # Convert from Tensor to NumPy
        
        # Undo normalization to view original image
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = image * std + mean  # De-normalize
        image = torch.clip(image, 0, 1)  # Ensure valid pixel range
        
        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")
    
    plt.show()

# Define prototypical loss function
def prototypical_loss(prototypes, query_samples, query_labels):
    """
    Calculate the prototypical loss as the negative log probability of the correct class.
    
    Args:
        prototypes: Tensor of shape (n_way, embedding_dim) - the prototype of each class
        query_samples: Tensor of shape (n_queries, embedding_dim) - the embedded query samples
        query_labels: Tensor of shape (n_queries) - the labels of the query samples
        
    Returns:
        Loss value and accuracy
    """
    # Calculate distances between query samples and prototypes
    dists = torch.cdist(query_samples, prototypes)**2  # Squared Euclidean distance
    
    # Calculate negative log probability of the correct class
    log_p_y = torch.nn.functional.log_softmax(-dists, dim=1)
    
    # Get the target indices
    target_inds = query_labels
    
    # Calculate the loss
    loss = -log_p_y.gather(1, target_inds.unsqueeze(1)).mean()
    
    # Calculate accuracy
    _, y_hat = log_p_y.max(1)
    acc = torch.eq(y_hat, target_inds).float().mean()
    
    return loss, acc

# Create a class mapping for episode construction
def create_class_mapping(dataset):
    """Create a mapping from class labels to indices of samples with that label."""
    class_map = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in class_map:
            class_map[label] = []
        class_map[label].append(idx)
    return class_map

# Function to create episodic data loader
def create_episode(dataset, class_map, n_way, n_support, n_query):
    """Create an episode for few-shot learning."""
    # Randomly select n_way classes
    classes = random.sample(list(class_map.keys()), n_way)
    
    support_samples = []
    support_labels = []
    query_samples = []
    query_labels = []
    
    for i, cls in enumerate(classes):
        # Get indices of samples for this class
        cls_indices = class_map[cls]
        
        # Randomly select n_support + n_query samples
        selected_indices = random.sample(cls_indices, n_support + n_query)
        
        # Split into support and query sets
        support_indices = selected_indices[:n_support]
        query_indices = selected_indices[n_support:]
        
        # Add samples to support and query sets
        for idx in support_indices:
            img, _ = dataset[idx]
            support_samples.append(img)
            support_labels.append(i)  # Use index as the label
        
        for idx in query_indices:
            img, _ = dataset[idx]
            query_samples.append(img)
            query_labels.append(i)  # Use index as the label
    
    # Convert to tensors
    support_samples = torch.stack(support_samples)
    support_labels = torch.tensor(support_labels)
    query_samples = torch.stack(query_samples)
    query_labels = torch.tensor(query_labels)
    
    return support_samples, support_labels, query_samples, query_labels

# Training function
def train_epoch(model, train_dataset, train_class_map, n_episodes, n_way, n_support, n_query, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        # Create episode
        support_samples, support_labels, query_samples, query_labels = create_episode(
            train_dataset, train_class_map, n_way, n_support, n_query
        )
        
        # Move to device
        support_samples = support_samples.to(device)
        support_labels = support_labels.to(device)
        query_samples = query_samples.to(device)
        query_labels = query_labels.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Compute embeddings
        support_embeddings = model(support_samples)
        query_embeddings = model(query_samples)
        
        # Compute prototypes
        prototypes = torch.zeros(n_way, support_embeddings.shape[1]).to(device)
        for i in range(n_way):
            mask = support_labels == i
            prototypes[i] = support_embeddings[mask].mean(0)
        
        # Compute loss and accuracy
        loss, acc = prototypical_loss(prototypes, query_embeddings, query_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
    
    return total_loss / n_episodes, total_acc / n_episodes

# Validation function
def validate(model, val_dataset, val_class_map, n_episodes, n_way, n_support, n_query, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for episode in tqdm(range(n_episodes), desc="Validating"):
            # Create episode
            support_samples, support_labels, query_samples, query_labels = create_episode(
                val_dataset, val_class_map, n_way, n_support, n_query
            )
            
            # Move to device
            support_samples = support_samples.to(device)
            support_labels = support_labels.to(device)
            query_samples = query_samples.to(device)
            query_labels = query_labels.to(device)
            
            # Compute embeddings
            support_embeddings = model(support_samples)
            query_embeddings = model(query_samples)
            
            # Compute prototypes
            prototypes = torch.zeros(n_way, support_embeddings.shape[1]).to(device)
            for i in range(n_way):
                mask = support_labels == i
                prototypes[i] = support_embeddings[mask].mean(0)
            
            # Compute loss and accuracy
            loss, acc = prototypical_loss(prototypes, query_embeddings, query_labels)
            
            total_loss += loss.item()
            total_acc += acc.item()
    
    return total_loss / n_episodes, total_acc / n_episodes

# Main training loop
def train_model(model, train_dataset, val_dataset, n_epochs, n_episodes, n_way, n_support, n_query, optimizer, lr_scheduler, device):
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Create class mappings
    train_class_map = create_class_mapping(train_dataset)
    val_class_map = create_class_mapping(val_dataset)
    
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_dataset, train_class_map, n_episodes, n_way, n_support, n_query, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_dataset, val_class_map, n_episodes//2, n_way, n_support, n_query, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        lr_scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_protonet_model.pth')
            print(f"Model saved with validation accuracy: {val_acc:.4f}")
    
    return train_losses, train_accs, val_losses, val_accs

# Test function
def test_model(model, test_dataset, n_episodes, n_way, n_support, n_query, device):
    # Load best model
    checkpoint = torch.load('best_protonet_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_acc']:.4f}")
    
    # Create class mapping for test set
    test_class_map = create_class_mapping(test_dataset)
    
    # Test
    model.eval()
    total_acc = 0
    
    with torch.no_grad():
        for episode in tqdm(range(n_episodes), desc="Testing"):
            # Create episode
            support_samples, support_labels, query_samples, query_labels = create_episode(
                test_dataset, test_class_map, n_way, n_support, n_query
            )
            
            # Move to device
            support_samples = support_samples.to(device)
            support_labels = support_labels.to(device)
            query_samples = query_samples.to(device)
            query_labels = query_labels.to(device)
            
            # Compute embeddings
            support_embeddings = model(support_samples)
            query_embeddings = model(query_samples)
            
            # Compute prototypes
            prototypes = torch.zeros(n_way, support_embeddings.shape[1]).to(device)
            for i in range(n_way):
                mask = support_labels == i
                prototypes[i] = support_embeddings[mask].mean(0)
            
            # Compute loss and accuracy
            _, acc = prototypical_loss(prototypes, query_embeddings, query_labels)
            
            total_acc += acc.item()
    
    test_acc = total_acc / n_episodes
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc

# Define inverse transform to convert tensors back to images
def inverse_transform(tensor):
    """Convert normalized tensor to image for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Un-normalize
    tensor = tensor * std + mean
    
    # Clip values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy array and transpose to (H, W, C)
    return tensor.cpu().numpy().transpose(1, 2, 0)

# Load the saved model
def load_model(device):
    model = ProtoNet(x_dim=3, hid_dim=64, z_dim=64)
    checkpoint = torch.load('best_protonet_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_acc']:.4f}")
    return model

# Function to create an episode and visualize it
def visualize_episode(model, test_dataset, n_way=5, n_support=5, n_query=5, device=None):
    """Create an episode, visualize support and query sets, and show model predictions"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create class mapping for test set
    test_class_map = {}
    for idx, (_, label) in enumerate(test_dataset):
        if label not in test_class_map:
            test_class_map[label] = []
        test_class_map[label].append(idx)
    
    # Randomly select n_way classes
    classes = random.sample(list(test_class_map.keys()), n_way)
    
    # Collect original class labels for reference
    original_classes = classes
    
    support_samples = []
    support_labels = []
    query_samples = []
    query_labels = []
    original_images_support = []
    original_images_query = []
    
    for i, cls in enumerate(classes):
        # Get indices of samples for this class
        cls_indices = test_class_map[cls]
        
        # Randomly select n_support + n_query samples
        selected_indices = random.sample(cls_indices, n_support + n_query)
        
        # Split into support and query sets
        support_indices = selected_indices[:n_support]
        query_indices = selected_indices[n_support:]
        
        # Add samples to support and query sets
        for idx in support_indices:
            img, _ = test_dataset[idx]
            support_samples.append(img)
            support_labels.append(i)  # Use index as the label
            
            # Store original image for visualization
            original_images_support.append((img, i, cls))
        
        for idx in query_indices:
            img, _ = test_dataset[idx]
            query_samples.append(img)
            query_labels.append(i)  # Use index as the label
            
            # Store original image for visualization
            original_images_query.append((img, i, cls))
    
    # Convert to tensors
    support_samples = torch.stack(support_samples)
    support_labels = torch.tensor(support_labels)
    query_samples = torch.stack(query_samples)
    query_labels = torch.tensor(query_labels)
    
    # Move to device
    support_samples = support_samples.to(device)
    support_labels = support_labels.to(device)
    query_samples = query_samples.to(device)
    query_labels = query_labels.to(device)
    
    # Compute embeddings
    with torch.no_grad():
        support_embeddings = model(support_samples)
        query_embeddings = model(query_samples)
    
    # Compute prototypes
    prototypes = torch.zeros(n_way, support_embeddings.shape[1]).to(device)
    for i in range(n_way):
        mask = support_labels == i
        prototypes[i] = support_embeddings[mask].mean(0)
    
    # Compute distances and predictions
    dists = torch.cdist(query_embeddings, prototypes)**2
    log_p_y = torch.nn.functional.log_softmax(-dists, dim=1)
    _, predictions = log_p_y.max(1)
    
    # Calculate accuracy
    accuracy = torch.eq(predictions, query_labels).float().mean().item()
    print(f"Episode accuracy: {accuracy:.4f}")
    
    # Calculate distances to prototypes for each query sample
    distances = []
    for i in range(len(query_samples)):
        dist_to_prototypes = dists[i].cpu().numpy()
        distances.append(dist_to_prototypes)
    
    # Convert predictions and query_labels to numpy arrays
    predictions = predictions.cpu().numpy()
    query_labels = query_labels.cpu().numpy()
    
    # Visualize the support set (organized by class)
    plt.figure(figsize=(15, 10))
    plt.suptitle("Support Set (Class Prototypes)", fontsize=16)
    
    # Organize support images by class for better visualization
    for class_idx in range(n_way):
        class_images = [(img, label, cls) for img, label, cls in original_images_support if label == class_idx]
        
        for i, (img_tensor, _, orig_class) in enumerate(class_images):
            # Calculate position for this class's images
            plt.subplot(n_way, n_support, class_idx * n_support + i + 1)
            img = inverse_transform(img_tensor)
            plt.imshow(img)
            plt.title(f"Class {class_idx}\nID: {orig_class}", fontsize=9)
            plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('protonet_support_set.png')
    
    # Visualize query set with predictions
    plt.figure(figsize=(15, 10))
    plt.suptitle("Query Set with Predictions", fontsize=16)
    
    # Calculate grid dimensions
    total_queries = len(original_images_query)
    grid_cols = min(5, total_queries)
    grid_rows = math.ceil(total_queries / grid_cols)
    
    for i, (img_tensor, true_label, orig_class) in enumerate(original_images_query):
        plt.subplot(grid_rows, grid_cols, i + 1)
        img = inverse_transform(img_tensor)
        plt.imshow(img)
        
        pred_label = predictions[i]
        is_correct = pred_label == true_label
        color = 'green' if is_correct else 'red'
        
        # Get closest prototype distance
        closest_dist = float(distances[i][pred_label])
        
        plt.title(f"True: {true_label} (ID: {orig_class})\nPred: {pred_label} (ID: {original_classes[pred_label]})\nDist: {closest_dist:.2f}", 
                  color=color, fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('protonet_predictions.png')
    
    # Print complete analysis
    print("\nDetailed Analysis:")
    print("=================")
    print(f"Classes in this episode: {original_classes}")
    print(f"Number of support samples per class: {n_support}")
    print(f"Number of query samples per class: {n_query}")
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Print per-class accuracy
    print("\nPer-class accuracy:")
    for cls_idx in range(n_way):
        cls_mask = query_labels == cls_idx
        if sum(cls_mask) > 0:
            cls_correct = sum((predictions == query_labels) & cls_mask)
            cls_acc = cls_correct / sum(cls_mask)
            print(f"Class {cls_idx} (ID: {original_classes[cls_idx]}): {cls_acc:.4f}")
    
    # Print confusion matrix
    confusion = np.zeros((n_way, n_way), dtype=int)
    for i in range(len(query_labels)):
        confusion[query_labels[i]][predictions[i]] += 1
    
    print("\nConfusion Matrix:")
    print("True\\Pred", end="")
    for i in range(n_way):
        print(f"  {i}  ", end="")
    print()
    
    for i in range(n_way):
        print(f"    {i}    ", end="")
        for j in range(n_way):
            print(f" {confusion[i][j]:3d} ", end="")
        print()
        
    # Return details for further analysis if needed
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': query_labels,
        'distances': distances,
        'original_classes': original_classes
    }

# Main function to run the visual test
def run_visual_test(test_dataset, device):
    model = load_model(device)
    
    results = []
    # Create multiple episodes to visualize
    for episode in range(3):
        print(f"\nVisualizing Episode {episode+1}")
        episode_results = visualize_episode(model, test_dataset, device=device)
        results.append(episode_results)
    
    # Calculate overall statistics across episodes
    overall_accuracy = np.mean([r['accuracy'] for r in results])
    print(f"\nOverall accuracy across all episodes: {overall_accuracy:.4f}")
    
    return results

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Set dataset path - adjust this to your local path
    dataset_path = "path/to/miniimagenet"
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    try:
        print("Loading datasets...")
        train_dataset = MiniImageNet(root=dataset_path, split="train", transform=transform)
        val_dataset = MiniImageNet(root=dataset_path, split="val", transform=transform)
        test_dataset = MiniImageNet(root=dataset_path, split="test", transform=transform)
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        print(f"Test dataset: {len(test_dataset)} samples")
        
        # Show some sample images
        print("Displaying sample images...")
        show_images(train_dataset)
        
        # Define the episodic training parameters
        n_way = 5  # Number of classes per episode
        n_support = 5  # Number of support samples per class (K-shot)
        n_query = 15  # Number of query samples per class
        n_episodes = 100  # Number of episodes per epoch
        
        # Initialize model
        model = ProtoNet(x_dim=3, hid_dim=64, z_dim=64)  # 3 channels for RGB images
        model = model.to(device)
        
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Train model
        print("Starting model training...")
        n_epochs = 50
        train_losses, train_accs, val_losses, val_accs = train_model(
            model, train_dataset, val_dataset, n_epochs, n_episodes, 
            n_way, n_support, n_query, optimizer, lr_scheduler, device
        )
        
        # Test the model
        test_acc = test_model(model, test_dataset, n_episodes, n_way, n_support, n_query, device)
        
        # Plot training and validation metrics
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig('protonet_training_metrics.png')
        plt.show()
        
        # Run visual test
        print("\nRunning visual test...")
        run_visual_test(test_dataset, device)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
