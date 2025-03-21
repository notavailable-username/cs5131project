import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

# Helper function to get the number of available GPUs
def get_available_gpus():
    """Returns the number of available GPUs on the system."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()

# Function to setup model for multi-GPU training
def setup_model_for_parallel(model, use_distributed=False):
    """
    Sets up the model for parallel processing.
    
    Args:
        model: The model to parallelize
        use_distributed: If True, use DistributedDataParallel, otherwise use DataParallel
        
    Returns:
        The parallelized model
    """
    num_gpus = get_available_gpus()
    
    if (num_gpus <= 1):
        return model  # No parallelization needed
        
    logger.info(f"Using {num_gpus} GPUs for training")
    
    if use_distributed:
        # For DistributedDataParallel (more efficient but requires more setup)
        if not torch.distributed.is_initialized():
            logger.warning("Distributed mode requested but not initialized")
            logger.warning("Falling back to DataParallel")
            return nn.DataParallel(model)
        return nn.parallel.DistributedDataParallel(model)
    else:
        # Simple DataParallel for single-process multi-GPU
        return nn.DataParallel(model)

class MiniImageNet(Dataset):
    """
    Mini-ImageNet dataset for few-shot learning.
    
    Args:
        root (str): Root directory of the dataset
        split (str): Dataset split ('train', 'val', or 'test')
        transform: Transforms to apply to images
    """
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # Load CSV file
        csv_path = os.path.join(root, f"{split}.csv")
        self.data = pd.read_csv(csv_path)

        # Extract image names and labels
        self.image_paths = self.data['filename'].tolist()
        self.labels = self.data['label'].tolist()
        
        # Create class mapping for quick access
        self.class_map = self._create_class_mapping()
        
        logger.info(f"Loaded {split} dataset with {len(self.image_paths)} images and {len(set(self.labels))} classes")

    def _create_class_mapping(self):
        """Create a mapping from class labels to indices of samples with that label."""
        class_map = {}
        for idx, label in enumerate(self.labels):
            if label not in class_map:
                class_map[label] = []
            class_map[label].append(idx)
        return class_map
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def conv_block(in_channels, out_channels):
    """
    Returns a block of convolutional layer followed by batch normalization,
    ReLU activation, and max pooling.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class DistanceNetwork(nn.Module):
    """
    Computes the cosine similarity between query and support samples.
    """
    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support, queries):
        """
        Calculate the cosine distance between support and query samples
        
        Args:
            support: support samples [n_way * n_support, embedding_dim]
            queries: query samples [n_queries, embedding_dim]
            
        Returns:
            similarities: cosine similarities [n_queries, n_way * n_support]
        """
        # Normalize embeddings
        support_norm = support / torch.norm(support, dim=1, keepdim=True)
        queries_norm = queries / torch.norm(queries, dim=1, keepdim=True)
        
        # Calculate cosine similarity
        similarities = torch.mm(queries_norm, support_norm.transpose(0, 1))
        return similarities

class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(AttentionLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        
    def forward(self, x, h=None):
        # x shape: [batch_size, seq_len, input_size]
        output, (h_n, c_n) = self.lstm(x, h)
        return output, (h_n, c_n)

class BiDirectionalLSTM(nn.Module):
    def __init__(self, size):
        super(BiDirectionalLSTM, self).__init__()
        self.lstm_cell_forward = nn.LSTMCell(size, size)
        self.lstm_cell_backward = nn.LSTMCell(size, size)
        
    def forward(self, embeddings):
        """
        Process the support set embeddings with a bidirectional LSTM
        
        Args:
            embeddings: support set embeddings [n_support, embedding_dim]
            
        Returns:
            processed_embeddings: processed embeddings [n_support, embedding_dim]
        """
        # Get dimensions
        batch_size = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]
        device = embeddings.device
        
        # Forward pass through forward LSTM
        forward_embeddings = []
        # Initialize hidden and cell states with batch size 1
        h_forward = torch.zeros(1, embedding_dim, device=device)
        c_forward = torch.zeros(1, embedding_dim, device=device)
        
        for i in range(batch_size):
            input_tensor = embeddings[i:i+1]  # Get single item with batch dimension
            h_forward, c_forward = self.lstm_cell_forward(input_tensor, (h_forward, c_forward))
            forward_embeddings.append(h_forward.clone())  # Clone to prevent modification
        
        # Backward pass through backward LSTM
        backward_embeddings = []
        # Initialize hidden and cell states with batch size 1
        h_backward = torch.zeros(1, embedding_dim, device=device)
        c_backward = torch.zeros(1, embedding_dim, device=device)
        
        for i in range(batch_size-1, -1, -1):
            input_tensor = embeddings[i:i+1]  # Get single item with batch dimension
            h_backward, c_backward = self.lstm_cell_backward(input_tensor, (h_backward, c_backward))
            backward_embeddings.insert(0, h_backward.clone())  # Clone to prevent modification
        
        # Combine forward and backward embeddings
        combined_embeddings = []
        for f, b in zip(forward_embeddings, backward_embeddings):
            combined_embeddings.append(f + b)
        
        # Stack all processed embeddings
        processed_embeddings = torch.cat(combined_embeddings, dim=0)
        
        return processed_embeddings

class MatchingNetwork(nn.Module):
    def __init__(self, n_way, n_support, n_query, feat_dim=64, lstm_layers=1, 
                 unrolling_steps=2, use_fce=True):
        """
        Implementation of Matching Networks for Few-Shot Learning as described in the paper.
        
        Args:
            n_way: Number of classes in a few-shot classification task
            n_support: Number of support examples per class
            n_query: Number of query examples per class
            feat_dim: Feature dimension after the embedding network
            lstm_layers: Number of LSTM layers in the FCE
            unrolling_steps: Number of processing steps for the query embedding
            use_fce: Whether to use Full Context Embeddings
        """
        super(MatchingNetwork, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.feat_dim = feat_dim
        self.lstm_layers = lstm_layers
        self.unrolling_steps = unrolling_steps
        self.use_fce = use_fce
        
        # Encoder network
        self.encoder = nn.Sequential(
            conv_block(3, feat_dim),
            conv_block(feat_dim, feat_dim),
            conv_block(feat_dim, feat_dim),
            conv_block(feat_dim, feat_dim),
            nn.Flatten()
        )
        
        # Calculate the expected feature dimension after flattening
        self.encoder_output_dim = feat_dim * 5 * 5  # 84x84 -> 42x42 -> 21x21 -> 10x10 -> 5x5
        
        # Full Context Embedding components
        if self.use_fce:
            self.g = BiDirectionalLSTM(self.encoder_output_dim)
            self.f = AttentionLSTM(self.encoder_output_dim * 2, self.encoder_output_dim, lstm_layers)
            
        # Distance network for similarity calculation
        self.distance_network = DistanceNetwork()

    def forward(self, support_images, support_labels_one_hot, query_images):
        """
        Forward pass for Matching Networks
        
        Args:
            support_images: Support images [n_way * n_support, channels, height, width]
            support_labels_one_hot: Support labels (one-hot) [n_way * n_support, n_way]
            query_images: Query images [n_way * n_query, channels, height, width]
            
        Returns:
            query_predictions: Class probabilities for query samples [n_way * n_query, n_way]
        """
        # Encode support and query images
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)
        
        # Apply FCE if enabled
        if self.use_fce:
            # Process support embeddings with g
            support_embeddings = self.g(support_embeddings)
            
            # Process query embeddings with f, attending to processed support embeddings
            num_query = query_embeddings.shape[0]
            
            # Initialize query context with zeros
            query_context = torch.zeros_like(query_embeddings)
            
            # Create hidden and cell state for the LSTM
            h = torch.zeros(self.lstm_layers, num_query, self.encoder_output_dim, 
                          device=query_embeddings.device)
            c = torch.zeros(self.lstm_layers, num_query, self.encoder_output_dim, 
                          device=query_embeddings.device)
            
            # Unroll for K steps
            for k in range(self.unrolling_steps):
                # Calculate attention between query and support
                similarities = self.distance_network(support_embeddings, query_embeddings)
                attention = torch.softmax(similarities, dim=1)
                
                # Calculate the read using the attention
                read = torch.mm(attention, support_embeddings)
                
                # Combine query embedding with the read vector
                query_with_context = torch.cat([query_embeddings, read], dim=1).unsqueeze(1)
                
                # Process through the LSTM
                lstm_out, (h, c) = self.f(query_with_context, (h, c))
                query_embeddings = lstm_out.squeeze(1)
        
        # Calculate distances between query and support embeddings
        similarities = self.distance_network(support_embeddings, query_embeddings)
        
        # Apply softmax to get attention weights
        attention = torch.softmax(similarities, dim=1)
        
        # Use attention weights to make predictions
        query_predictions = torch.mm(attention, support_labels_one_hot)
        
        return query_predictions

def cosine_similarity(a, b):
    """
    Compute cosine similarity between query set and support set.
    
    Args:
        a: Tensor of shape (n_queries, embedding_dim)
        b: Tensor of shape (n_support, embedding_dim)
        
    Returns:
        similarity_matrix: Tensor of shape (n_queries, n_support)
    """
    # Normalize the vectors
    a_norm = a / torch.norm(a, dim=1, keepdim=True)
    b_norm = b / torch.norm(b, dim=1, keepdim=True)
    
    # Calculate cosine similarity
    similarity_matrix = torch.mm(a_norm, b_norm.transpose(0, 1))
    
    return similarity_matrix

def create_episode(dataset, n_way, n_support, n_query):
    """
    Create an episode for few-shot learning.
    
    Args:
        dataset (MiniImageNet): The dataset to sample from
        n_way (int): Number of classes in each episode
        n_support (int): Number of support samples per class
        n_query (int): Number of query samples per class
        
    Returns:
        tuple: (support_samples, support_labels_one_hot, query_samples, query_labels)
    """
    # Randomly select n_way classes
    classes = random.sample(list(dataset.class_map.keys()), n_way)
    
    support_samples = []
    support_labels = []
    query_samples = []
    query_labels = []
    
    for i, cls in enumerate(classes):
        # Get indices of samples for this class
        cls_indices = dataset.class_map[cls]
        
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
    query_samples = torch.stack(query_samples)
    query_labels = torch.tensor(query_labels)
    
    # Convert support labels to one-hot encoding for matching nets
    support_labels_one_hot = torch.zeros(len(support_labels), n_way)
    for i, label in enumerate(support_labels):
        support_labels_one_hot[i, label] = 1
    
    return support_samples, support_labels_one_hot, query_samples, query_labels

def matching_loss(predictions, targets):
    """
    Calculate the loss for matching networks.
    
    Args:
        predictions: Predicted class probabilities [n_queries, n_way]
        targets: Ground truth labels [n_queries]
        
    Returns:
        tuple: (loss, accuracy)
    """
    # Compute cross-entropy loss
    loss = torch.nn.functional.cross_entropy(predictions, targets)
    
    # Compute accuracy
    _, predicted_classes = torch.max(predictions, 1)
    acc = torch.eq(predicted_classes, targets).float().mean()
    
    return loss, acc

def train_epoch(model, dataset, n_episodes, optimizer, n_way, n_support, n_query, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataset: The dataset to sample episodes from
        n_episodes: Number of episodes in this epoch
        optimizer: The optimizer
        n_way: Number of classes per episode
        n_support: Number of support samples per class
        n_query: Number of query samples per class
        device: Device to use for computation
        
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.train()
    total_loss = 0
    total_acc = 0
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        # Create episode
        support_samples, support_labels_one_hot, query_samples, query_labels = create_episode(
            dataset, n_way, n_support, n_query
        )
        
        # Move to device
        support_samples = support_samples.to(device)
        support_labels_one_hot = support_labels_one_hot.to(device)
        query_samples = query_samples.to(device)
        query_labels = query_labels.to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Get predictions
        query_predictions = model(support_samples, support_labels_one_hot, query_samples)
        
        # Compute loss and accuracy
        loss, acc = matching_loss(query_predictions, query_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
    
    return total_loss / n_episodes, total_acc / n_episodes

def validate(model, dataset, n_episodes, n_way, n_support, n_query, device):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        dataset: The dataset to sample episodes from
        n_episodes: Number of episodes for validation
        n_way: Number of classes per episode
        n_support: Number of support samples per class
        n_query: Number of query samples per class
        device: Device to use for computation
        
    Returns:
        tuple: (average_loss, average_accuracy)
    """
    model.eval()
    total_loss = 0
    total_acc = 0
    
    with torch.no_grad():
        for episode in tqdm(range(n_episodes), desc="Validating"):
            # Create episode
            support_samples, support_labels_one_hot, query_samples, query_labels = create_episode(
                dataset, n_way, n_support, n_query
            )
            
            # Move to device
            support_samples = support_samples.to(device)
            support_labels_one_hot = support_labels_one_hot.to(device)
            query_samples = query_samples.to(device)
            query_labels = query_labels.to(device)
            
            # Get predictions
            query_predictions = model(support_samples, support_labels_one_hot, query_samples)
            
            # Compute loss and accuracy
            loss, acc = matching_loss(query_predictions, query_labels)
            
            total_loss += loss.item()
            total_acc += acc.item()
    
    return total_loss / n_episodes, total_acc / n_episodes

def test(model, dataset, n_episodes, n_way, n_support, n_query, device):
    """
    Test the model.
    
    Args:
        model: The model to test
        dataset: The dataset to sample episodes from
        n_episodes: Number of episodes for testing
        n_way: Number of classes per episode
        n_support: Number of support samples per class
        n_query: Number of query samples per class
        device: Device to use for computation
        
    Returns:
        float: Average accuracy
    """
    model.eval()
    total_acc = 0
    
    with torch.no_grad():
        for episode in tqdm(range(n_episodes), desc="Testing"):
            # Create episode
            support_samples, support_labels_one_hot, query_samples, query_labels = create_episode(
                dataset, n_way, n_support, n_query
            )
            
            # Move to device
            support_samples = support_samples.to(device)
            support_labels_one_hot = support_labels_one_hot.to(device)
            query_samples = query_samples.to(device)
            query_labels = query_labels.to(device)
            
            # Get predictions
            query_predictions = model(support_samples, support_labels_one_hot, query_samples)
            
            # Compute accuracy
            _, acc = matching_loss(query_predictions, query_labels)
            
            total_acc += acc.item()
    
    return total_acc / n_episodes

def train_model(model, train_dataset, val_dataset, args, device):
    """
    Train the model for multiple epochs.
    
    Args:
        model: The model to train
        train_dataset: The training dataset
        val_dataset: The validation dataset
        args: Command-line arguments with hyperparameters
        device: Device to use for computation
        
    Returns:
        tuple: (train_losses, train_accuracies, val_losses, val_accuracies)
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, args.model_name)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    else:
        lr_scheduler = None
    
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    logger.info("Starting training...")
    
    for epoch in range(args.n_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_dataset, args.n_episodes, 
            optimizer, args.n_way, args.n_support, args.n_query, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(
            model, val_dataset, args.n_val_episodes, 
            args.n_way, args.n_support, args.n_query, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate if scheduler is defined
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Print metrics
        logger.info(f"Epoch {epoch+1}/{args.n_epochs} - Time: {epoch_time:.2f}s - "
                   f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                   f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Save model if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'hyperparameters': vars(args),
            }, model_save_path)
            logger.info(f"Model saved with validation accuracy: {val_acc:.4f}")
    
    # Plot and save training metrics
    plot_training_metrics(train_losses, train_accs, val_losses, val_accs, 
                         save_path=os.path.join(args.output_dir, 'training_metrics.png'))
    
    return train_losses, train_accs, val_losses, val_accs

def plot_training_metrics(train_losses, train_accs, val_losses, val_accs, save_path):
    """
    Plot and save training metrics.
    
    Args:
        train_losses: List of training losses
        train_accs: List of training accuracies
        val_losses: List of validation losses
        val_accs: List of validation accuracies
        save_path: Path to save the plot
    """
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
    plt.savefig(save_path)
    logger.info(f"Training metrics saved to {save_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Matching Networks for Few-Shot Learning')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, default='archive1', 
                        help='Path to the Mini-ImageNet dataset')
    
    # Model parameters
    parser.add_argument('--feat_dim', type=int, default=64, 
                        help='Feature dimension in convolutional layers')
    parser.add_argument('--lstm_layers', type=int, default=1, 
                        help='Number of LSTM layers in the FCE')
    parser.add_argument('--unrolling_steps', type=int, default=2, 
                        help='Number of unrolling steps for the query embedding in FCE')
    parser.add_argument('--use_fce', action='store_true',
                        help='Use Full Context Embeddings (FCE) with LSTM')
    
    # Training parameters
    parser.add_argument('--n_way', type=int, default=5, 
                        help='Number of classes per episode')
    parser.add_argument('--n_support', type=int, default=5, 
                        help='Number of support examples per class (K-shot)')
    parser.add_argument('--n_query', type=int, default=15, 
                        help='Number of query examples per class')
    parser.add_argument('--n_episodes', type=int, default=100, 
                        help='Number of episodes per epoch during training')
    parser.add_argument('--n_val_episodes', type=int, default=50, 
                        help='Number of episodes per validation')
    parser.add_argument('--n_test_episodes', type=int, default=100, 
                        help='Number of episodes for testing')
    parser.add_argument('--n_epochs', type=int, default=50, 
                        help='Number of training epochs')
    
    # Optimizer parameters
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'cosine', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_step_size', type=int, default=20, 
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.5, 
                        help='Gamma for StepLR scheduler')
    
    # Parallel training parameters
    parser.add_argument('--use_distributed', action='store_true',
                        help='Use DistributedDataParallel instead of DataParallel')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Directory to save outputs')
    parser.add_argument('--model_name', type=str, default='best_matchingnet_model.pth', 
                        help='Name of the model file')
    parser.add_argument('--test_only', action='store_true',
                        help='Only perform testing, no training')
    
    args = parser.parse_args()
    return args

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("No GPU found, using CPU")
    
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((84, 84)),  # Resize to Mini-ImageNet size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MiniImageNet(root=args.dataset_path, split="train", transform=transform)
    val_dataset = MiniImageNet(root=args.dataset_path, split="val", transform=transform)
    test_dataset = MiniImageNet(root=args.dataset_path, split="test", transform=transform)
    
    # Initialize the model
    model = MatchingNetwork(
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        feat_dim=args.feat_dim,
        lstm_layers=args.lstm_layers,
        unrolling_steps=args.unrolling_steps,
        use_fce=args.use_fce
    )
    model = model.to(device)
    
    # Setup model for parallel processing if multiple GPUs are available
    model = setup_model_for_parallel(model, use_distributed=args.use_distributed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test only or train+test
    if args.test_only:
        # Load pre-trained model
        model_path = os.path.join(args.output_dir, args.model_name)
        if not os.path.exists(model_path):
            logger.error(f"Model file {model_path} does not exist. Cannot test.")
            return
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint['epoch']} with validation accuracy {checkpoint['val_acc']:.4f}")
    else:
        # Train the model
        train_losses, train_accs, val_losses, val_accs = train_model(
            model, train_dataset, val_dataset, args, device
        )
    
    # Test the model
    logger.info("Starting testing...")
    test_acc = test(model, test_dataset, args.n_test_episodes, 
                 args.n_way, args.n_support, args.n_query, device)
    
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Save test results
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"N-way: {args.n_way}\n")
        f.write(f"K-shot: {args.n_support}\n")
        f.write(f"N-query: {args.n_query}\n")
        f.write(f"N-episodes: {args.n_test_episodes}\n")
        f.write(f"Use FCE: {args.use_fce}\n")
    
    logger.info(f"Test results saved to {os.path.join(args.output_dir, 'test_results.txt')}")

if __name__ == "__main__":
    main()