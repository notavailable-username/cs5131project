import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """
    A convolutional block with batch normalization, ReLU activation and max pooling
    as described in the Matching Networks paper.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class EmbeddingModule(nn.Module):
    """
    Embedding module for images as described in the paper.
    It consists of 4 convolutional blocks with 64 filters.
    """
    def __init__(self, in_channels=1, out_channels=64, embedding_size=64):
        super(EmbeddingModule, self).__init__()
        self.layer1 = ConvBlock(in_channels, out_channels)
        self.layer2 = ConvBlock(out_channels, out_channels)
        self.layer3 = ConvBlock(out_channels, out_channels)
        self.layer4 = ConvBlock(out_channels, out_channels)
        
        # Final layer to get embedding of correct size if needed
        self.final_layer = nn.Linear(out_channels, embedding_size)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # At this point, x should be [batch_size, channels, 1, 1]
        x = x.view(x.size(0), -1)
        
        # Apply final layer if needed
        if x.size(1) != self.final_layer.out_features:
            x = self.final_layer(x)
            
        return x


class AttentionLSTMCell(nn.Module):
    """
    LSTM cell used for the Full Context Embeddings (FCE)
    """
    def __init__(self, input_size, hidden_size):
        super(AttentionLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM weights
        self.lstm = nn.LSTMCell(input_size + hidden_size, hidden_size)
        
    def forward(self, x, h_prev, c_prev, read):
        """
        Forward pass through the LSTM cell
        x: input tensor
        h_prev: previous hidden state
        c_prev: previous cell state
        read: read vector from attention
        """
        combined = torch.cat([x, read], dim=1)
        h, c = self.lstm(combined, (h_prev, c_prev))
        return h, c


class FullContextEmbedding(nn.Module):
    """
    Full Context Embedding as described in the paper.
    Uses attention and LSTM to process the test example in the context
    of the support set.
    """
    def __init__(self, embedding_size, attention_size, num_processing_steps=4):
        super(FullContextEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.num_processing_steps = num_processing_steps
        
        self.lstm_cell = AttentionLSTMCell(embedding_size, attention_size)
        
    def forward(self, test_embedding, support_embeddings):
        """
        Forward pass to generate full context embeddings
        
        test_embedding: embedding of the test example [batch_size, embedding_size]
        support_embeddings: embeddings of the support set [num_support, embedding_size]
        
        Returns: refined test embedding considering full context of support set
        """
        batch_size = test_embedding.shape[0]
        
        # Initialize hidden and cell state
        h = torch.zeros(batch_size, self.attention_size, device=test_embedding.device)
        c = torch.zeros(batch_size, self.attention_size, device=test_embedding.device)
        
        # Initialize read vector
        read = torch.zeros(batch_size, self.embedding_size, device=test_embedding.device)
        
        # Process for K steps
        for k in range(self.num_processing_steps):
            # Get hidden state
            h_prev = h + test_embedding
            
            # Calculate attention weights
            attention_logits = torch.matmul(h_prev, support_embeddings.transpose(0, 1))
            attention_weights = F.softmax(attention_logits, dim=1)
            
            # Calculate read vector
            read = torch.matmul(attention_weights, support_embeddings)
            
            # Update hidden state
            h, c = self.lstm_cell(test_embedding, h, c, read)
        
        # Return final hidden state + original embedding (residual connection)
        return h + test_embedding


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for processing the support set examples
    in the context of the full support set.
    """
    def __init__(self, embedding_size, hidden_size):
        super(BidirectionalLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        
        # Output projection to map back to embedding size
        self.projection = nn.Linear(hidden_size * 2, embedding_size)
        
    def forward(self, support_embeddings):
        """
        Forward pass through the bidirectional LSTM
        
        support_embeddings: embeddings of the support set [num_support, embedding_size]
        
        Returns: processed support embeddings
        """
        # Process through LSTM
        outputs, _ = self.lstm(support_embeddings.unsqueeze(0))
        outputs = outputs.squeeze(0)
        
        # Project back to embedding size
        processed_embeddings = self.projection(outputs)
        
        # Add residual connection
        processed_embeddings = processed_embeddings + support_embeddings
        
        return processed_embeddings


class MatchingNetwork(nn.Module):
    """
    Matching Networks for One-Shot Learning as described in the paper.
    """
    def __init__(
        self,
        in_channels=1,
        embedding_size=64,
        lstm_hidden_size=32,
        attention_size=64,
        use_fce=True,
        num_processing_steps=4
    ):
        super(MatchingNetwork, self).__init__()
        
        self.embedding_size = embedding_size
        self.use_fce = use_fce
        
        # Embedding function f and g (initially the same network)
        self.embedding_fn = EmbeddingModule(in_channels, embedding_size, embedding_size)
        
        # Optional Full Context Embedding components
        if self.use_fce:
            self.g_bidirectional_lstm = BidirectionalLSTM(embedding_size, lstm_hidden_size)
            self.f_attention_lstm = FullContextEmbedding(
                embedding_size, 
                attention_size, 
                num_processing_steps
            )
    
    def embed_support(self, support_set):
        """
        Embed the support set using function g
        
        support_set: [n_way * k_shot, channels, height, width]
        
        Returns: support set embeddings
        """
        # Get initial embeddings
        support_embeddings = self.embedding_fn(support_set)
        
        # Process with bidirectional LSTM if using FCE
        if self.use_fce:
            support_embeddings = self.g_bidirectional_lstm(support_embeddings)
            
        return support_embeddings
    
    def embed_query(self, query_set, support_embeddings):
        """
        Embed the query set using function f
        
        query_set: [batch_size, channels, height, width]
        support_embeddings: [n_way * k_shot, embedding_size]
        
        Returns: query set embeddings
        """
        # Get initial embeddings
        query_embeddings = self.embedding_fn(query_set)
        
        # Process with attention LSTM if using FCE
        if self.use_fce:
            query_embeddings = self.f_attention_lstm(query_embeddings, support_embeddings)
            
        return query_embeddings
    
    def calculate_cosine_similarity(self, query_embeddings, support_embeddings):
        """
        Calculate cosine similarity between query and support embeddings
        
        query_embeddings: [batch_size, embedding_size]
        support_embeddings: [n_way * k_shot, embedding_size]
        
        Returns: cosine similarity matrix
        """
        # Normalize embeddings
        query_embeddings_norm = F.normalize(query_embeddings, p=2, dim=1)
        support_embeddings_norm = F.normalize(support_embeddings, p=2, dim=1)
        
        # Calculate cosine similarity
        cosine_similarities = torch.mm(query_embeddings_norm, support_embeddings_norm.t())
        
        return cosine_similarities
    
    def forward(self, support_set, support_labels_one_hot, query_set):
        """
        Forward pass through the network
        
        support_set: [n_way * k_shot, channels, height, width]
        support_labels_one_hot: [n_way * k_shot, n_way]
        query_set: [batch_size, channels, height, width]
        
        Returns: predicted probability distribution over classes for query set
        """
        # Embed support set and query set
        support_embeddings = self.embed_support(support_set)
        query_embeddings = self.embed_query(query_set, support_embeddings)
        
        # Calculate cosine similarity
        cosine_similarities = self.calculate_cosine_similarity(query_embeddings, support_embeddings)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(cosine_similarities, dim=1)
        
        # Calculate weighted sum of one-hot labels
        predictions = torch.mm(attention_weights, support_labels_one_hot)
        
        return predictions


class EpisodicTrainer:
    """
    Episodic trainer for training Matching Networks in a meta-learning setup.
    """
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
    def train_episode(self, n_way, k_shot, query_size, data_source):
        """
        Train for a single episode
        
        n_way: number of classes per episode
        k_shot: number of examples per class
        query_size: number of query examples per class
        data_source: a function that generates episodes
        
        Returns: mean loss and accuracy for the episode
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Sample episode
        support_set, support_labels, query_set, query_labels = data_source(
            n_way, k_shot, query_size
        )
        
        # Move data to device
        support_set = torch.tensor(support_set).float().to(self.device)
        support_labels = torch.tensor(support_labels).long().to(self.device)
        query_set = torch.tensor(query_set).float().to(self.device)
        query_labels = torch.tensor(query_labels).long().to(self.device)
        
        # Convert labels to one-hot
        support_labels_one_hot = F.one_hot(support_labels, n_way).float()
        
        # Forward pass
        predictions = self.model(support_set, support_labels_one_hot, query_set)
        
        # Calculate loss and accuracy
        loss = F.cross_entropy(predictions, query_labels)
        
        # Calculate accuracy
        _, predicted_classes = torch.max(predictions, 1)
        accuracy = (predicted_classes == query_labels).float().mean()
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), accuracy.item()
    
    def evaluate_episode(self, n_way, k_shot, query_size, data_source):
        """
        Evaluate for a single episode
        
        n_way: number of classes per episode
        k_shot: number of examples per class
        query_size: number of query examples per class
        data_source: a function that generates episodes
        
        Returns: mean loss and accuracy for the episode
        """
        self.model.eval()
        
        with torch.no_grad():
            # Sample episode
            support_set, support_labels, query_set, query_labels = data_source(
                n_way, k_shot, query_size
            )
            
            # Move data to device
            support_set = torch.tensor(support_set).float().to(self.device)
            support_labels = torch.tensor(support_labels).long().to(self.device)
            query_set = torch.tensor(query_set).float().to(self.device)
            query_labels = torch.tensor(query_labels).long().to(self.device)
            
            # Convert labels to one-hot
            support_labels_one_hot = F.one_hot(support_labels, n_way).float()
            
            # Forward pass
            predictions = self.model(support_set, support_labels_one_hot, query_set)
            
            # Calculate loss and accuracy
            loss = F.cross_entropy(predictions, query_labels)
            
            # Calculate accuracy
            _, predicted_classes = torch.max(predictions, 1)
            accuracy = (predicted_classes == query_labels).float().mean()
            
        return loss.item(), accuracy.item()


def train_matching_networks(
    model,
    optimizer,
    data_source_train,
    data_source_test,
    n_way=5,
    k_shot=1,
    query_size=15,
    num_episodes_train=100000,
    num_episodes_val=100,
    val_interval=1000,
    device='cuda'
):
    """
    Train Matching Networks using episodic training
    
    model: the Matching Networks model
    optimizer: optimizer for training
    data_source_train: function to generate training episodes
    data_source_test: function to generate testing episodes
    n_way: number of classes per episode
    k_shot: number of examples per class
    query_size: number of query examples per class
    num_episodes_train: number of training episodes
    num_episodes_val: number of validation episodes
    val_interval: validate every val_interval episodes
    device: device to use
    """
    trainer = EpisodicTrainer(model, optimizer, device)
    
    best_accuracy = 0.0
    
    for episode in range(1, num_episodes_train + 1):
        # Train for one episode
        train_loss, train_acc = trainer.train_episode(
            n_way, k_shot, query_size, data_source_train
        )
        
        # Validate periodically
        if episode % val_interval == 0:
            val_losses = []
            val_accs = []
            
            for _ in range(num_episodes_val):
                val_loss, val_acc = trainer.evaluate_episode(
                    n_way, k_shot, query_size, data_source_test
                )
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            
            mean_val_loss = np.mean(val_losses)
            mean_val_acc = np.mean(val_accs)
            
            print(f"Episode {episode}/{num_episodes_train}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {mean_val_loss:.4f}, Val Acc: {mean_val_acc:.4f}")
            
            # Save best model
            if mean_val_acc > best_accuracy:
                best_accuracy = mean_val_acc
                torch.save(model.state_dict(), 'best_matching_networks.pt')
                print(f"New best validation accuracy: {best_accuracy:.4f}")
        
        # Print training progress
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes_train}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")


# Example usage for Omniglot dataset
def omniglot_example():
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    import random
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyper-parameters
    embedding_size = 64
    lstm_hidden_size = 32
    attention_size = 64
    learning_rate = 0.001
    
    # Initialize the model
    model = MatchingNetwork(
        in_channels=1,  # Omniglot is grayscale
        embedding_size=embedding_size,
        lstm_hidden_size=lstm_hidden_size,
        attention_size=attention_size,
        use_fce=True,
        num_processing_steps=4
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Omniglot dataset
    train_dataset = torchvision.datasets.Omniglot(
        root='./data',
        background=True,  # Use the background set for training
        download=True,
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
    )
    
    test_dataset = torchvision.datasets.Omniglot(
        root='./data',
        background=False,  # Use the evaluation set for testing
        download=True,
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
    )
    
    # Group data by class
    train_data_by_class = {}
    for img, label in train_dataset:
        if label not in train_data_by_class:
            train_data_by_class[label] = []
        train_data_by_class[label].append(img)
    
    test_data_by_class = {}
    for img, label in test_dataset:
        if label not in test_data_by_class:
            test_data_by_class[label] = []
        test_data_by_class[label].append(img)
    
    # Function to generate episodes
    def generate_episode(n_way, k_shot, query_size, data_by_class):
        """Generate an episode for training or testing"""
        # Sample n_way classes
        classes = random.sample(list(data_by_class.keys()), n_way)
        
        # Initialize arrays
        support_set = []
        support_labels = []
        query_set = []
        query_labels = []
        
        # For each class, sample k_shot + query_size examples
        for i, cls in enumerate(classes):
            # Get all examples for this class
            examples = data_by_class[cls]
            
            # Sample k_shot + query_size examples without replacement
            selected_examples = random.sample(examples, k_shot + query_size)
            
            # Split into support and query
            support_examples = selected_examples[:k_shot]
            query_examples = selected_examples[k_shot:]
            
            # Add to support set
            for example in support_examples:
                support_set.append(example.numpy())
                support_labels.append(i)
            
            # Add to query set
            for example in query_examples:
                query_set.append(example.numpy())
                query_labels.append(i)
        
        # Convert to numpy arrays
        support_set = np.stack(support_set)
        support_labels = np.array(support_labels)
        query_set = np.stack(query_set)
        query_labels = np.array(query_labels)
        
        return support_set, support_labels, query_set, query_labels
    
    # Functions to generate episodes for training and testing
    def generate_train_episode(n_way, k_shot, query_size):
        return generate_episode(n_way, k_shot, query_size, train_data_by_class)
    
    def generate_test_episode(n_way, k_shot, query_size):
        return generate_episode(n_way, k_shot, query_size, test_data_by_class)
    
    # Train the model
    train_matching_networks(
        model,
        optimizer,
        generate_train_episode,
        generate_test_episode,
        n_way=5,
        k_shot=1,
        query_size=15,
        num_episodes_train=50000,
        num_episodes_val=100,
        val_interval=1000,
        device=device
    )


# Example for miniImageNet dataset
def mini_imagenet_example():
    # This would require miniImageNet data handling
    # Implementation would be similar to Omniglot but with RGB images
    pass


if __name__ == "__main__":
    # Run Omniglot example
    omniglot_example()