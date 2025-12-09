import torch
import random
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
import re
import matplotlib.pyplot as plt
import numpy as np
import uuid


BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
EMBEDDING_DIMENSION = 128
HIDDEN_DIMENSION = 512
NUMBER_OF_RNN_LAYERS = 2
OPTIMIZER = "adam"
VOCAB_SIZE = 48 # +1 PAD +1 SOS
PAD_IDX = 0

VOCAB = ['<PAD>', '<SOS>', '<ADJLIST_START>', '<ADJLIST_END>', '<PATH_START>', '<PATH_END>', ';', '<-->', '<ORIGIN_START>', '<ORIGIN_END>', '<TARGET_START>', '<TARGET_END>']
for i in range(6):
    for j in range(6):
        VOCAB.append(f'({i},{j})')

token_to_id = {token: idx for idx, token in enumerate(VOCAB)}
id_to_token = {idx: token for idx, token in enumerate(VOCAB)}
PAD_IDX = token_to_id['<PAD>']
SOS_IDX = token_to_id['<SOS>']

def parse_sequence(seq_str):
    """
    Parse a sequence string and convert tokens to IDs.
    Handles brackets, commas, and quotes.
    
    Args:
        seq_str: String representation of a list (e.g., "['token1', 'token2']")
        token_to_id: Dictionary mapping tokens to IDs
    
    Returns:
        List of token IDs
    """
    # Parse the string as a Python list
    seq = ast.literal_eval(seq_str)
    
    # Convert tokens to IDs
    token_ids = []
    for token in seq:
        if token in token_to_id:
            token_ids.append(token_to_id[token])
        else:
            print(f"Warning: Unknown token '{token}', skipping...")
    
    return token_ids
    
def load_and_parse_data(df):
    """
    Load data and convert to token ID sequences.
    
    Args:
        df: DataFrame containing the data
    
    Returns:
        src_sequences: List of source sequences (token IDs)
        target_sequences: List of target sequences (token IDs)
    """
    # Parse sequences
    src_sequences = []
    
    for idx, row in df.iterrows():
        src_seq = parse_sequence(row['input_sequence'])
        
        src_sequences.append(src_seq)
    
    print(f"Loaded {len(src_sequences)} sequences")
    print(f"Vocabulary size: {len(token_to_id)}")
    print(f"Sample source sequence (first 10 tokens): {src_sequences[0][:10]}")
    
    return src_sequences


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
        rnn_input = torch.cat([emb, context], dim=1).unsqueeze(1) # y_(t' - 1) and c_t' given

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
        B, Ty = target_tokens.shape # target tokens has meaningful + pad
        outputs = []

        prev_token = torch.full((B,), SOS_IDX, dtype=torch.long, device=target_tokens.device)

        state = initial_state

        for t in range(0, Ty):
            logits, state, _ = self.decoder_step(prev_token, state, encoder_outputs, src_mask)
            outputs.append(logits) # first token is meaningful and not sos

            if torch.rand(1).item() < teacher_forcing_ratio:
                prev_token = target_tokens[:, t] # after path end, this is pad
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
    
    def fit(self, train_loader, test_loader=None, epochs=20, learning_rate=1e-4,
        teacher_forcing_ratio=0.5, device='cuda', optimizer_type='adam'):

        def compute_sequence_accuracy(pred_seq, target_seq, pad_idx):
            mask = target_seq != pad_idx
            pred_clean = pred_seq[mask]
            target_clean = target_seq[mask]
            if pred_clean.shape[0] != target_clean.shape[0]:
                return 0
            return int(torch.equal(pred_clean, target_clean))
    
        self.to(device)
    
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
    
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
    
        history = {
            'train_loss': [],
            'test_loss': [],
            'train_token_acc': [],
            'train_seq_acc': [],
            'test_token_acc': [],
            'test_seq_acc': [],
        }
    
        for epoch in range(epochs):
    
            # -------------------------
            # TRAINING
            # -------------------------
            self.train()
            train_loss = 0
            train_token_correct = 0
            train_token_total = 0
            train_seq_correct = 0
            train_seq_total = 0
    
            for src, target in train_loader:
                src, target = src.to(device), target.to(device)
    
                optimizer.zero_grad()
    
                outputs = self.forward(src, target, teacher_forcing_ratio) # T_y logits may contain path end followed by garbage
                # outputs: (B, Ty, vocab)
    
                B, Ty, V = outputs.shape
                outputs_flat = outputs.reshape(-1, V)
                targets_flat = target.reshape(-1)
    
                loss = criterion(outputs_flat, targets_flat)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5)
                optimizer.step()
    
                train_loss += loss.item()
    
                # Token accuracy
                pred_tokens = outputs_flat.argmax(dim=1)
                mask = targets_flat != self.pad_idx
                train_token_correct += (pred_tokens[mask] == targets_flat[mask]).sum().item()
                train_token_total += mask.sum().item()
    
                # Sequence accuracy
                pred_seqs = outputs.argmax(dim=2)  # (B, Ty)
                for i in range(B):
                    train_seq_correct += compute_sequence_accuracy(pred_seqs[i], target[i], self.pad_idx)
                    train_seq_total += 1
    
            avg_train_loss = train_loss / len(train_loader)
            train_token_acc = 100 * train_token_correct / train_token_total
            train_seq_acc = 100 * train_seq_correct / train_seq_total
    
            history['train_loss'].append(avg_train_loss)
            history['train_token_acc'].append(train_token_acc)
            history['train_seq_acc'].append(train_seq_acc)
    
            # -------------------------
            # TEST / VALIDATION
            # -------------------------
            if test_loader:
                self.eval()
                test_loss = 0
                test_token_correct = 0
                test_token_total = 0
                test_seq_correct = 0
                test_seq_total = 0
    
                with torch.no_grad():
                    for src, target in test_loader:
                        src, target = src.to(device), target.to(device)
    
                        outputs = self.forward(src, target, teacher_forcing_ratio=0)
                        B, Ty, V = outputs.shape
    
                        outputs_flat = outputs.reshape(-1, V)
                        targets_flat = target.reshape(-1)
    
                        loss = criterion(outputs_flat, targets_flat)
                        test_loss += loss.item()
    
                        # Token acc
                        pred_tokens = outputs_flat.argmax(dim=1)
                        mask = targets_flat != self.pad_idx
                        test_token_correct += (pred_tokens[mask] == targets_flat[mask]).sum().item()
                        test_token_total += mask.sum().item()
    
                        # Sequence acc
                        pred_seqs = outputs.argmax(dim=2)
                        for i in range(B):
                            test_seq_correct += compute_sequence_accuracy(pred_seqs[i], target[i], self.pad_idx)
                            test_seq_total += 1
    
                avg_test_loss = test_loss / len(test_loader)
                test_token_acc = 100 * test_token_correct / test_token_total
                test_seq_acc = 100 * test_seq_correct / test_seq_total
    
                history['test_loss'].append(avg_test_loss)
                history['test_token_acc'].append(test_token_acc)
                history['test_seq_acc'].append(test_seq_acc)
    
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.3f}, Token Acc: {train_token_acc:.2f}%, Seq Acc: {train_seq_acc:.2f}% | "
                      f"Test Loss: {avg_test_loss:.3f}, Token Acc: {test_token_acc:.2f}%, Seq Acc: {test_seq_acc:.2f}%")
    
            else:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {avg_train_loss:.3f}, Token Acc: {train_token_acc:.2f}%, Seq Acc: {train_seq_acc:.2f}%")
    
        return history


def rnn_attention_caller(input_data, model_path, device='cuda'):
    """
    Runs inference using pretrained RNN-attention model.
    - input_data: pd.DataFrame with 'input_sequence' column (stringified token lists)
    - model_path: path to saved checkpoint
    - device: 'cuda' or 'cpu'
    Returns:
      pd.DataFrame with 'output_path' column containing predicted token lists trimmed at first EOS.
    """
    # --------------------------------------------------
    # 1. Load and parse sequences
    # --------------------------------------------------
    # load_and_parse_data returns (src_sequences, target_sequences)
    src_sequences = load_and_parse_data(input_data)
    # dummy targets (not used in inference)
    dummy_tgt = [[SOS_IDX] for _ in src_sequences]

    loader = create_data_loader(
        src_sequences=src_sequences,
        target_sequences=dummy_tgt,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pad_idx=PAD_IDX
    )

    # --------------------------------------------------
    # 2. Load model and checkpoint (robust)
    # --------------------------------------------------
    model = Seq2SeqAttentionModel(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIMENSION,
        hidden_dim=HIDDEN_DIMENSION,
        num_layers=NUMBER_OF_RNN_LAYERS,
        pad_idx=PAD_IDX
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)

    # Accept several common names for the saved state-dict
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ('model', 'model_state_dict', 'state_dict'):
            if key in ckpt:
                state_dict = ckpt[key]
                break
        # if none of the above, it might already be a state-dict stored as the outer dict
        if state_dict is None:
            # Heuristic: if ckpt looks like a state_dict (tensor values), try to load it directly
            try:
                model.load_state_dict(ckpt)
                state_dict = None  # already loaded successfully
            except Exception:
                state_dict = None
    else:
        # ckpt is not a dict (rare) — try to use it directly
        state_dict = ckpt

    if state_dict is not None:
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model state_dict from checkpoint: {e}")

    model.eval()

    # --------------------------------------------------
    # 3. Find EOS id to trim predicted sequences
    # --------------------------------------------------
    # Prefer path-specific EOS tokens used in this project:
    eos_id = 2
    # If no candidate found, eos_id stays None and trimming won't occur.

    # --------------------------------------------------
    # 4. Perform inference and trim at first EOS
    # --------------------------------------------------
    predictions = []

    with torch.no_grad():
        for src, _ in loader:
            src = src.to(device)

            src_mask = (src != PAD_IDX).float()

            encoder_outputs, h_n = model.encoder(src)
            pred_ids = model.decode_inference(
                encoder_outputs,
                h_n,
                max_len=60,
                sos_token=SOS_IDX,
                eos_token=(eos_id if eos_id is not None else None),
                src_mask=src_mask
            )  # (B, T) tensor of token ids (or list/tensor)
            # print()
            # ensure tensor -> cpu list
            if isinstance(pred_ids, torch.Tensor):
                batch_preds = pred_ids.cpu().tolist()
            else:
                # if decode_inference returned numpy array / list
                batch_preds = [list(x) for x in pred_ids]

            # Convert IDs → tokens and trim at first eos (exclude EOS)
            for seq in batch_preds:
                # print(seq)
                if eos_id is not None and eos_id in seq:
                    first = seq.index(eos_id)
                    seq_trimmed = seq[:first]
                else:
                    seq_trimmed = seq
                first = seq.index(token_to_id['<PATH_END>']) if token_to_id['<PATH_END>'] in seq else len(seq)-1
                seq_trimmed = seq[:first+1]

                tokens = [id_to_token.get(int(tid), '<UNK>') for tid in seq_trimmed]
                predictions.append(tokens)

    # --------------------------------------------------
    # 5. Store predictions in DataFrame (preserve order)
    # --------------------------------------------------
    # safety: ensure predictions length matches rows
    if len(predictions) != len(input_data):
        if len(predictions) < len(input_data):
            predictions.extend([[]] * (len(input_data) - len(predictions)))
        else:
            predictions = predictions[:len(input_data)]

    out_df = input_data.copy()
    out_df['output_path'] = predictions
    return out_df
