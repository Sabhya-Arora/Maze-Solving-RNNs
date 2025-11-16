import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
EMBEDDING_DIMENSION = 128
HIDDEN_DIMENSION = 512
NUMBER_OF_RNN_LAYERS = 2
OPTIMIZER = "adam"
VOCAB_SIZE = 46
PAD_IDX = 0

class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Encoder RNN (unidirectional)
        self.encoder_rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Decoder RNN input = embedding + context
        self.decoder_rnn = nn.RNN(
            input_size=embedding_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Bahdanau attention parameters
        self.W_a = nn.Linear(hidden_dim, hidden_dim)
        self.U_a = nn.Linear(hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1)

        # Output layer takes [s_t ; c_t] → vocab
        self.output_layer = nn.Linear(2 * hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

    # -----------------------------------------------------------
    # Encoder
    # -----------------------------------------------------------
    def encoder(self, src_tokens):
        embedded = self.embedding(src_tokens)
        outputs, h_n = self.encoder_rnn(embedded)
        return outputs, h_n

    # -----------------------------------------------------------
    # One decoding step
    # -----------------------------------------------------------
    def decoder_step(self, prev_token, prev_state, encoder_outputs, src_mask=None):
        B, T, H = encoder_outputs.shape

        # 1. embed previous token
        emb = self.embedding(prev_token)  # (B, E)

        # 2. attention
        s_prev = prev_state[-1]  # (B, H)
        dec_part = self.W_a(s_prev).unsqueeze(1).expand(B, T, H)
        enc_part = self.U_a(encoder_outputs)
        e = torch.tanh(dec_part + enc_part)
        e = self.v_a(e).squeeze(-1)  # (B, T)
        
        # Apply mask to ignore padding positions
        if src_mask is not None:
            e = e.masked_fill(src_mask == 0, float('-inf'))
        
        attn = torch.softmax(e, dim=1)  # (B, T)

        # 3. context
        context = torch.bmm(attn.unsqueeze(1), encoder_outputs).squeeze(1)

        # 4. RNN input
        rnn_input = torch.cat([emb, context], dim=1).unsqueeze(1)

        # 5. decoder RNN step
        rnn_output, new_state = self.decoder_rnn(rnn_input, prev_state)
        s_t = rnn_output.squeeze(1)

        # 6. output
        combined = torch.cat([s_t, context], dim=1)
        logits = self.output_layer(combined)

        return logits, new_state, attn

    # -----------------------------------------------------------
    # Full decoding loop
    # -----------------------------------------------------------
    def decode_train(self, encoder_outputs, initial_state, target_tokens, teacher_forcing_ratio, src_mask=None):
        B, Ty = target_tokens.shape
        outputs = []

        prev_token = target_tokens[:, 0]
        state = initial_state

        for t in range(1, Ty):
            logits, state, _ = self.decoder_step(prev_token, state, encoder_outputs, src_mask)
            outputs.append(logits)

            if torch.rand(1).item() < teacher_forcing_ratio:
                prev_token = target_tokens[:, t]
            else:
                prev_token = logits.argmax(dim=1)

        return torch.stack(outputs, dim=1)

    # -----------------------------------------------------------
    # Forward function
    # -----------------------------------------------------------
    def forward(self, src, target, teacher_forcing_ratio=0.5):
        # Create mask for source padding
        src_mask = (src != self.pad_idx).float()  # (B, T_src)
        
        encoder_outputs, h_n = self.encoder(src)
        return self.decode_train(encoder_outputs, h_n, target, teacher_forcing_ratio, src_mask)
    
    def decode_inference(self, encoder_outputs, initial_state, max_len=50, sos_token=1, eos_token=2, src_mask=None):
        B = encoder_outputs.size(0)
        outputs = []

        prev_token = torch.full((B,), sos_token, dtype=torch.long, device=encoder_outputs.device)
        state = initial_state

        for _ in range(max_len):
            logits, state, _ = self.decoder_step(prev_token, state, encoder_outputs, src_mask)
            next_token = logits.argmax(dim=1)
            outputs.append(next_token)

            # stop if ALL sequences emitted EOS
            if (next_token == eos_token).all():
                break

            prev_token = next_token

        # shape = (B, T_generated)
        return torch.stack(outputs, dim=1)
    
    def fit(self, train_loader, val_loader=None, epochs=20, learning_rate=1e-4, 
            teacher_forcing_ratio=0.5, device='cuda', optimizer_type='adam'):
        """
        Train the model using cross-entropy loss with proper padding handling.
        
        Args:
            train_loader: DataLoader with (src, target) batches
            val_loader: Optional validation DataLoader
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            teacher_forcing_ratio: Probability of using teacher forcing
            device: Device to train on ('cuda' or 'cpu')
            optimizer_type: Type of optimizer ('adam' or 'sgd')
        
        Returns:
            Dictionary with training history (losses, accuracies)
        """
        self.to(device)
        
        # Setup optimizer
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        
        # Cross-entropy loss with ignore_index for padding
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (src, target) in enumerate(train_loader):
                src = src.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                # Output shape: (B, Ty-1, vocab_size)
                outputs = self.forward(src, target, teacher_forcing_ratio)
                
                # Reshape for loss computation
                # outputs: (B, Ty-1, vocab_size) -> (B * (Ty-1), vocab_size)
                # target[:, 1:]: (B, Ty-1) -> (B * (Ty-1))
                outputs_flat = outputs.reshape(-1, self.vocab_size)
                targets_flat = target[:, 1:].reshape(-1)
                
                # Compute loss (padding tokens are automatically ignored)
                loss = criterion(outputs_flat, targets_flat)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                
                # Calculate accuracy (excluding padding tokens)
                predictions = outputs_flat.argmax(dim=1)
                mask = targets_flat != self.pad_idx
                train_correct += (predictions[mask] == targets_flat[mask]).sum().item()
                train_total += mask.sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            
            # Validation phase
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for src, target in val_loader:
                        src = src.to(device)
                        target = target.to(device)
                        
                        outputs = self.forward(src, target, teacher_forcing_ratio=1.0)
                        
                        outputs_flat = outputs.reshape(-1, self.vocab_size)
                        targets_flat = target[:, 1:].reshape(-1)
                        
                        loss = criterion(outputs_flat, targets_flat)
                        val_loss += loss.item()
                        
                        predictions = outputs_flat.argmax(dim=1)
                        mask = targets_flat != self.pad_idx
                        val_correct += (predictions[mask] == targets_flat[mask]).sum().item()
                        val_total += mask.sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
                
                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(val_accuracy)
                
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            else:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        
        return history


# ===============================================================
# Utility functions for padding and data loading
# ===============================================================

def pad_sequences(sequences, pad_idx=0, max_len=None):
    """
    Pad a list of sequences to the same length.
    
    Args:
        sequences: List of sequences (lists or tensors)
        pad_idx: Padding token index
        max_len: Maximum length to pad to (if None, uses longest sequence)
    
    Returns:
        Padded tensor of shape (batch_size, max_len)
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if isinstance(seq, list):
            seq = torch.tensor(seq, dtype=torch.long)
        
        # Pad or truncate
        if len(seq) < max_len:
            padding = torch.full((max_len - len(seq),), pad_idx, dtype=torch.long)
            seq = torch.cat([seq, padding])
        else:
            seq = seq[:max_len]
        
        padded.append(seq)
    
    return torch.stack(padded)


def collate_fn(batch, pad_idx=0):
    """
    Collate function for DataLoader to handle variable-length sequences.
    
    Args:
        batch: List of (src, target) tuples
        pad_idx: Padding token index
    
    Returns:
        Batched and padded (src, target) tensors
    """
    src_batch, target_batch = zip(*batch)
    
    # Convert to tensors if needed
    src_tensors = [torch.tensor(seq, dtype=torch.long) if isinstance(seq, list) else seq 
                   for seq in src_batch]
    target_tensors = [torch.tensor(seq, dtype=torch.long) if isinstance(seq, list) else seq 
                      for seq in target_batch]
    
    # Pad sequences
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=pad_idx)
    target_padded = pad_sequence(target_tensors, batch_first=True, padding_value=pad_idx)
    
    return src_padded, target_padded


class MazeDataset(Dataset):
    """
    Dataset class for maze sequences.
    """
    def __init__(self, src_sequences, target_sequences):
        """
        Args:
            src_sequences: List of source sequences (input mazes)
            target_sequences: List of target sequences (paths)
        """
        assert len(src_sequences) == len(target_sequences)
        self.src_sequences = src_sequences
        self.target_sequences = target_sequences
    
    def __len__(self):
        return len(self.src_sequences)
    
    def __getitem__(self, idx):
        return self.src_sequences[idx], self.target_sequences[idx]


def create_data_loader(src_sequences, target_sequences, batch_size=32, 
                       shuffle=True, pad_idx=0):
    """
    Create a DataLoader with proper padding for variable-length sequences.
    
    Args:
        src_sequences: List of source sequences
        target_sequences: List of target sequences
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        pad_idx: Padding token index
    
    Returns:
        DataLoader object
    """
    dataset = MazeDataset(src_sequences, target_sequences)
    
    # Create custom collate function with the specified pad_idx
    collate = lambda batch: collate_fn(batch, pad_idx=pad_idx)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                     collate_fn=collate)
