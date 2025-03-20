import os
import json
from functools import partial
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import numpy as np
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
import time

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
    
    if num_gpus <= 1:
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


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)


class Protonet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        """
        Initialize Prototypical Network with the specified dimensions.
        
        Args:
            x_dim (int): Number of input channels
            hid_dim (int): Hidden dimension in convolutional layers
            z_dim (int): Output dimension of the embedding
        """
        super(Protonet, self).__init__()
        
        # Create the encoder network
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            Flatten()
        )

    def forward(self, xs, xq):
        n_class = xs.size(0)
        n_support = xs.size(1)
        n_query = xq.size(1)
        
        # Reshape to feed through encoder
        x = torch.cat([
            xs.view(n_class * n_support, *xs.size()[2:]),
            xq.view(n_class * n_query, *xq.size()[2:])
        ], 0)
        
        # Get embeddings
        z = self.encoder(x)
        z_dim = z.size(-1)
        
        # Separate support and query embeddings
        z_support = z[:n_class * n_support].view(n_class, n_support, z_dim)
        z_query = z[n_class * n_support:]
        
        # Calculate prototypes (mean of support examples for each class)
        prototypes = z_support.mean(1)
        
        # Calculate distance from each query to each prototype
        dists = euclidean_dist(z_query, prototypes)
        
        return dists, prototypes
    
    def loss(self, sample):
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()
        elif xq.device.type == 'mps':
            # Move target_inds to MPS and ensure it's properly allocated
            target_inds = target_inds.to('mps')

        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class*n_support:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        # If using MPS, move calculation to CPU temporarily to avoid MPS gather issues
        if log_p_y.device.type == 'mps':
            cpu_log_p_y = log_p_y.cpu()
            cpu_target_inds = target_inds.cpu()
            gathered = cpu_log_p_y.gather(2, cpu_target_inds)
            loss_val = -gathered.squeeze().view(-1).mean().to(log_p_y.device)

            # For accuracy calculation
            _, y_hat = cpu_log_p_y.max(2)
            acc_val = torch.eq(y_hat, cpu_target_inds.squeeze()).float().mean().to(log_p_y.device)
        else:
            loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
            _, y_hat = log_p_y.max(2)
            acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


class MiniImageNet(torch.utils.data.Dataset):
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
        img_path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_episode(dataset, n_way, n_support, n_query):
    """
    Create an episode for few-shot learning.
    
    Args:
        dataset: The dataset to sample from
        n_way (int): Number of classes in each episode
        n_support (int): Number of support samples per class
        n_query (int): Number of query samples per class
        
    Returns:
        dict: 'xs' - support samples, 'xq' - query samples
    """
    # Randomly select n_way classes
    classes = random.sample(list(dataset.class_map.keys()), n_way)
    
    support_samples = []
    query_samples = []
    
    for i, cls in enumerate(classes):
        # Get indices of samples for this class
        cls_indices = dataset.class_map[cls]
        
        # Randomly select n_support + n_query samples
        selected_indices = random.sample(cls_indices, n_support + n_query)
        
        # Split into support and query sets
        support_indices = selected_indices[:n_support]
        query_indices = selected_indices[n_support:]
        
        # Add samples to support and query sets
        cls_support = []
        for idx in support_indices:
            img, _ = dataset[idx]
            cls_support.append(img)
        support_samples.append(torch.stack(cls_support))
        
        cls_query = []
        for idx in query_indices:
            img, _ = dataset[idx]
            cls_query.append(img)
        query_samples.append(torch.stack(cls_query))
    
    # Convert to tensors
    xs = torch.stack(support_samples)  # [n_way, n_support, c, h, w]
    xq = torch.stack(query_samples)    # [n_way, n_query, c, h, w]
    
    return {
        'xs': xs,
        'xq': xq
    }

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
        sample = create_episode(dataset, n_way, n_support, n_query)
        
        # Move to device
        sample['xs'] = sample['xs'].to(device)
        sample['xq'] = sample['xq'].to(device)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass and calculate loss
        loss, metrics = model.loss(sample)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += metrics['loss']
        total_acc += metrics['acc']
    
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
            sample = create_episode(dataset, n_way, n_support, n_query)
            
            # Move to device
            sample['xs'] = sample['xs'].to(device)
            sample['xq'] = sample['xq'].to(device)
            
            # Forward pass and calculate loss
            loss, metrics = model.loss(sample)
            
            total_loss += metrics['loss']
            total_acc += metrics['acc']
    
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
            sample = create_episode(dataset, n_way, n_support, n_query)
            
            # Move to device
            sample['xs'] = sample['xs'].to(device)
            sample['xq'] = sample['xq'].to(device)
            
            # Forward pass and calculate loss
            _, metrics = model.loss(sample)
            
            total_acc += metrics['acc']
    
    return total_acc / n_episodes

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

class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = { }
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

    def train(self, **kwargs):
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0, # epochs done so far
            't': 0, # samples seen so far
            'batch': 0, # samples seen in current epoch
            'stop': False
        }

        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()

            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                state['sample'] = sample
                self.hooks['on_sample'](state)

                state['optimizer'].zero_grad()
                loss, state['output'] = state['model'].loss(state['sample'])
                self.hooks['on_forward'](state)

                loss.backward()
                self.hooks['on_backward'](state)

                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prototypical Networks for Few-Shot Learning')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, default='/Users/jadentjeng/Downloads/archive-4', 
                        help='Path to the Mini-ImageNet dataset')
    
    # Model parameters
    parser.add_argument('--x_dim', type=int, default=3, 
                        help='Number of input channels')
    parser.add_argument('--hid_dim', type=int, default=64, 
                        help='Hidden dimension in convolutional layers')
    parser.add_argument('--z_dim', type=int, default=64, 
                        help='Output dimension of the embedding')
    parser.add_argument('--use_fce', action='store_true',
                        help='Use Full Context Embeddings (FCE) with LSTM')
    parser.add_argument('--num_processing_steps', type=int, default=4,
                      help='Number of processing steps for FCE in Matching Networks')
    
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
    parser.add_argument('--model_name', type=str, default='best_protonet_model.pth', 
                        help='Name of the model file')
    parser.add_argument('--test_only', action='store_true',
                        help='Only perform testing, no training')
    
    args = parser.parse_args()
    return args

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
        tuple: (train_losses, train_accs, val_losses, val_accuracies, best_model_path)
    """
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Initialize learning rate scheduler
    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    else:
        scheduler = None
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    best_val_acc = 0.0
    wait = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    logger.info("Starting training...")
    
    for epoch in range(args.n_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model=model,
            dataset=train_dataset,
            n_episodes=args.n_episodes,
            optimizer=optimizer,
            n_way=args.n_way,
            n_support=args.n_support,
            n_query=args.n_query,
            device=device
        )
        
        # Validate
        val_loss, val_acc = validate(
            model=model,
            dataset=val_dataset,
            n_episodes=args.n_val_episodes,
            n_way=args.n_way,
            n_support=args.n_support,
            n_query=args.n_query,
            device=device
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - start_time
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{args.n_epochs} - Time: {epoch_time:.2f}s - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
            
            # Save the best model
            model_path = os.path.join(args.output_dir, args.model_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'hyperparameters': vars(args),
            }, model_path)
            logger.info(f"Model saved to {model_path}")
            
            wait = 0
        else:
            wait += 1
            logger.info(f"Validation accuracy did not improve. Wait: {wait}")
    
    # Plot training metrics
    metrics_path = os.path.join(args.output_dir, 'training_metrics.png')
    plot_training_metrics(train_losses, train_accs, val_losses, val_accs, metrics_path)
    
    return train_losses, train_accs, val_losses, val_accs, os.path.join(args.output_dir, args.model_name)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
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
    model = Protonet(x_dim=args.x_dim, hid_dim=args.hid_dim, z_dim=args.z_dim)
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
        _, _, _, _, best_model_path = train_model(
            model, train_dataset, val_dataset, args, device
        )
        
        # Load best model for testing
        logger.info("Loading best model for testing...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    logger.info("Starting testing...")
    test_acc = test(
        model=model,
        dataset=test_dataset,
        n_episodes=args.n_test_episodes,
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        device=device
    )
    
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    
    # Save test results
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"N-way: {args.n_way}\n")
        f.write(f"K-shot: {args.n_support}\n")
        f.write(f"N-query: {args.n_query}\n")
        f.write(f"N-episodes: {args.n_test_episodes}\n")
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()