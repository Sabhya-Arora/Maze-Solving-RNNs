import math
import ast
import pandas as pd
from collections import Counter
from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from time import time

D_MODEL = 128
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1, batch_first: bool = False):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)   # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            seq_len = x.size(1)
            pos = self.pe[:seq_len].transpose(0, 1)
            x = x + pos
        else:
            seq_len = x.size(0)
            x = x + self.pe[:seq_len]
        return self.dropout(x)


###############################################################################
# Transformer Encoder–Decoder
###############################################################################
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, D_MODEL, padding_idx=PAD_IDX)
        self.pos_encoder = PositionalEncoding(D_MODEL, dropout=DROPOUT, batch_first=True)
        self.pos_decoder = PositionalEncoding(D_MODEL, dropout=DROPOUT, batch_first=True)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, NUM_LAYERS)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, NUM_LAYERS)

        self.output_layer = nn.Linear(D_MODEL, vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # masking for autoregressive decoding
    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    @staticmethod
    def make_padding_mask(seq):
        return seq == PAD_IDX

    def forward(self, src, tgt): # uses teacher forcing inside decoder returns unnormalized logits
        """
        src: (B,S)
        tgt: (B,T) - tgt[:,0] must be SOS, computing next tokens
        """
        B, S = src.shape
        _, T = tgt.shape

        src_emb = self.embedding(src) * math.sqrt(D_MODEL)
        tgt_emb = self.embedding(tgt) * math.sqrt(D_MODEL)

        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_decoder(tgt_emb)

        src_pad_mask = self.make_padding_mask(src)
        tgt_pad_mask = self.make_padding_mask(tgt)
        tgt_mask = self.generate_square_subsequent_mask(T, src.device)

        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        return self.output_layer(out)

    # Greedy decoding for inference/test
    def greedy_decode(self, src, max_len=200):
        """
        Autoregressive greedy decode that returns:
          - sequences : list[list[int]]  (predicted token ids, trimmed at EOS)
          - logits    : Tensor (B, T_out, V)  raw logits per step (not softmaxed)
          - lengths   : Tensor (B,) lengths (number of predicted tokens per sample, excludes EOS)
        Use these logits + lengths to compute CE loss during validation/test.
        """
        self.eval()
        device = src.device
        src = src.to(device)
        B = src.size(0)

        src_emb = self.embedding(src) * math.sqrt(D_MODEL)
        src_emb = self.pos_encoder(src_emb)
        src_pad_mask = self.make_padding_mask(src)

        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        ys = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        sequences = [[] for _ in range(B)]
        logits_list = []  # list of (B, V) tensors for each step

        for _ in range(max_len):
            tgt_emb = self.embedding(ys) * math.sqrt(D_MODEL)
            tgt_emb = self.pos_decoder(tgt_emb)

            T = ys.size(1)
            tgt_mask = self.generate_square_subsequent_mask(T, device)
            tgt_pad_mask = self.make_padding_mask(ys)

            out = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask
            )

            step_logits = self.output_layer(out[:, -1, :])  # (B, V)
            logits_list.append(step_logits)

            next_token = step_logits.argmax(dim=-1)  # (B,)

            # append to sequences for unfinished examples
            for idx in range(B):
                if not finished[idx]:
                    sequences[idx].append(next_token[idx].item())

            # mark finished when EOS predicted
            finished |= (next_token == EOS_IDX)

            # append tokens for next iteration
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

            if finished.all():
                break

        # if no steps produced
        if len(logits_list) == 0:
            V = self.output_layer.weight.size(0)
            return sequences, torch.zeros((B, 0, V), device=device), torch.zeros(B, dtype=torch.long, device=device)

        # Stack logits_list -> (B, T_out, V)
        # each element of logits_list is (B, V); stacking with dim=1 yields (B, T_out, V)
        logits = torch.stack(logits_list, dim=1)

        # lengths: number of predicted tokens per sequence (exclude EOS)
        lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long, device=device)

        # For convenience, trim sequences at EOS if present (already done above by stopping append after EOS)
        # but ensure sequences don't contain EOS token (we appended token ids as output of argmax)
        for i in range(B):
            if EOS_IDX in sequences[i]:
                first = sequences[i].index(EOS_IDX)
                sequences[i] = sequences[i][:first]

        return sequences, logits, lengths

def parse_sequence(seq_str):
    return ast.literal_eval(seq_str)



def build_vocab(src, tgt):
    counter = Counter()
    for s in src: counter.update(s)
    for t in tgt: counter.update(t)

    tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
    for tok, _ in counter.most_common():
        if tok not in tokens:
            tokens.append(tok)

    tok2id = {tok: i for i, tok in enumerate(tokens)}
    id2tok = {i: tok for tok, i in tok2id.items()}
    return tok2id, id2tok



def convert_to_ids(list_of_seqs, tok2id):
    return [[tok2id.get(tok, PAD_IDX) for tok in seq] for seq in list_of_seqs]



class MazeDataset(Dataset):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return torch.tensor(self.src[idx]), torch.tensor(self.tgt[idx])



def collate(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD_IDX)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX)
    return src, tgt


def train_one_epoch(model, loader, optimizer, criterion, epoch=None):
    """
    Train for one epoch with teacher forcing.
    Returns: (avg_loss, seq_accuracy_pct, token_accuracy_pct)
    """
    model.train()
    tot_loss = 0.0
    preds_seq, gold_seq = [] , []

    desc = f"Epoch {epoch}" if epoch is not None else "Training"
    loader_iter = tqdm(loader, desc=desc, unit="batch", leave=False)

    for src, tgt in loader_iter:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        inp = tgt[:, :-1]   # teacher-forcing inputs (include SOS)
        out = tgt[:, 1:]    # targets (no SOS)

        optimizer.zero_grad()
        logits = model(src, inp)  # expected shape: (batch, seq_len, vocab)
        loss = criterion(logits.reshape(-1, logits.size(-1)), out.reshape(-1))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += loss.item()

        # predicted tokens (per-position) from logits
        preds = logits.argmax(dim=-1)          # (batch, seq_len)
        preds_seq.extend(preds.cpu().tolist()) # list of lists
        gold_seq.extend(tgt.cpu().tolist())    # keep full target (includes SOS / PAD / EOS)

        loader_iter.set_postfix(loss=loss.item())

    avg_loss = tot_loss / len(loader)
    seq_acc = seq_accuracy(preds_seq, gold_seq)
    tok_acc = token_accuracy(preds_seq, gold_seq)

    return avg_loss, seq_acc, tok_acc

def _trim_to_eos(seq_tensor, eos_idx):
    """Return tensor sliced up to (but not including) the first EOS token, if any."""
    eos_positions = (seq_tensor == eos_idx).nonzero(as_tuple=True)[0]
    if eos_positions.numel() > 0:
        return seq_tensor[: eos_positions[0]]
    return seq_tensor


def seq_accuracy(pred_seqs, target_seqs, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX):
    correct = 0
    total = len(pred_seqs)
    print(pred_seqs[0], target_seqs[0])
    for p, t in zip(pred_seqs, target_seqs):
        # convert python lists → tensors if needed
        p = torch.tensor(p) if not torch.is_tensor(p) else p
        t = torch.tensor(t) if not torch.is_tensor(t) else t

        # remove PAD tokens from target
        mask = t != pad_idx
        target_clean = t[mask]

        # remove leading SOS token from target if present (predictions do not include SOS)
        if target_clean.numel() > 0 and target_clean[0].item() == sos_idx:
            target_clean = target_clean[1:]

        target_clean = _trim_to_eos(target_clean, eos_idx)
        p = _trim_to_eos(p, eos_idx)

        # exact sequence match
        if p.numel() != target_clean.numel():
            continue

        if torch.equal(p, target_clean):
            correct += 1

    return 100 * correct / max(1, total)


def token_accuracy(pred_seqs, target_seqs, pad_idx=PAD_IDX, sos_idx=SOS_IDX, eos_idx=EOS_IDX):
    """
    Token-level accuracy:
      correct_tokens / total_target_tokens (excluding PAD and initial SOS).
    Predictions trimmed at EOS; tokens beyond prediction length count as incorrect.
    Returns percentage (0-100).
    """
    correct = 0
    total = 0

    for p, t in zip(pred_seqs, target_seqs):
        p = torch.tensor(p) if not torch.is_tensor(p) else p
        t = torch.tensor(t) if not torch.is_tensor(t) else t

        # clean target: remove PAD tokens
        mask = t != pad_idx
        target_clean = t[mask]

        # remove leading SOS if present
        if target_clean.numel() > 0 and target_clean[0].item() == sos_idx:
            target_clean = target_clean[1:]

        target_clean = _trim_to_eos(target_clean, eos_idx)
        p = _trim_to_eos(p, eos_idx)

        tgt_len = target_clean.numel()
        if tgt_len == 0:
            continue

        total += tgt_len

        # count matches up to min length
        minlen = min(p.numel(), tgt_len)
        if minlen > 0:
            correct += int((p[:minlen] == target_clean[:minlen]).sum().item())

        # tokens in target beyond prediction length are incorrect (we don't add anything for them)

    return 100 * correct / max(1, total)

def evaluate(model, loader, criterion):
    model.eval()
    tot_loss = 0
    preds_seq, gold_seq = [], []  # list of list

    with torch.no_grad():
        loader_iter = tqdm(loader, desc="Evaluating", unit="batch", leave=False)
        for src, tgt in loader_iter:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)

            inp = tgt[:, :-1]
            out = tgt[:, 1:]



            # gen, logits, _ = model.greedy_decode(src, max_len=out.size(1))
            logits = model(src, inp)
            loss = criterion(logits.reshape(-1, logits.size(-1)), out.reshape(-1))
            tot_loss += loss.item()

            # get gen from greedy decoding
            gen = model.greedy_decode(src, max_len=out.size(1))[0]
            preds_seq.extend(gen)
            gold_seq.extend(tgt.cpu().tolist())

    avg_loss = tot_loss / len(loader)
    seq_acc = seq_accuracy(preds_seq, gold_seq)
    tok_acc = token_accuracy(preds_seq, gold_seq)
    return avg_loss, seq_acc, tok_acc


def transformer_attention_caller(input_data: pd.DataFrame,
                                 model_path: str,
                                 device: str = 'cuda',
                                 batch_size: int = BATCH_SIZE,
                                 max_len: int = 200):
    """
    Runs inference using pretrained Transformer-attention model saved with:
      torch.save({"model": model.state_dict(), "tok2id": tok2id, "id2tok": id2tok, "vocab_size": vocab_size}, SAVE_PATH)

    input_data: pd.DataFrame with 'src' column (stringified lists) OR 'input_sequence'
    model_path: path to saved checkpoint
    device: 'cuda' or 'cpu' (defaults to DEVICE)
    Returns: pd.DataFrame with 'output_path' column (list of token strings per row)
    """
    if device is None:
        device = DEVICE

    # pick source column
    if 'src' in input_data.columns:
        src_col = 'src'
    elif 'input_sequence' in input_data.columns:
        src_col = 'input_sequence'
    else:
        raise ValueError("input_data must contain a 'src' or 'input_sequence' column containing stringified token lists.")

    # parse input sequences
    src_seqs = [parse_sequence(s) for s in input_data[src_col].tolist()]

    # load checkpoint (map to device)
    ckpt = torch.load(model_path, map_location=device)

    # Try to recover tok2id/id2tok/vocab_size/state_dict from checkpoint
    tok2id = None
    id2tok = None
    vocab_size = None
    state_dict = None

    if isinstance(ckpt, dict):
        # expected keys based on your save: "model", "tok2id", "id2tok", "vocab_size"
        if 'tok2id' in ckpt and 'id2tok' in ckpt:
            tok2id = ckpt['tok2id']
            id2tok = ckpt['id2tok']
        if 'vocab_size' in ckpt:
            vocab_size = ckpt['vocab_size']
        # state dict keys (accept several common names)
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            # maybe the checkpoint itself *is* a state dict
            state_dict = ckpt

    else:
        # ckpt might be a plain state_dict
        state_dict = ckpt

    # Fallback: build vocab from source sequences (if checkpoint didn't include it)
    if tok2id is None or id2tok is None:
        tok2id, id2tok = build_vocab(src_seqs, [])
        vocab_size = vocab_size or len(tok2id)
        print("Warning: checkpoint had no tok2id/id2tok — built vocab from input data (may mismatch model).")

    vocab_size = vocab_size or len(tok2id)

    # Important: ensure PAD/SOS/EOS indices align with the checkpoint mapping.
    # Update module-level constants so make_padding_mask / greedy_decode use the same indices.
    pad_idx = tok2id.get(PAD_TOKEN, PAD_IDX)
    sos_idx = tok2id.get(SOS_TOKEN, SOS_IDX)
    eos_idx = tok2id.get(EOS_TOKEN, EOS_IDX)

    # update global constants used by the model class and helpers
    globals()['PAD_IDX'] = pad_idx
    globals()['SOS_IDX'] = sos_idx
    globals()['EOS_IDX'] = eos_idx

    # Convert source tokens -> ids (unknown tokens -> pad_idx)
    src_ids = [[tok2id.get(tok, pad_idx) for tok in seq] for seq in src_seqs]

    # DataLoader (dummy targets, preserve order)
    dataset = MazeDataset(src_ids, [[sos_idx] for _ in src_ids])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Instantiate model with vocab_size and load weights
    model = TransformerSeq2Seq(vocab_size=vocab_size).to(device)

    # load state dict into model (best-effort)
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        # helpful message if shapes/mapping mismatch
        raise RuntimeError(f"Failed to load model state_dict from checkpoint: {e}")

    model.eval()

    # run greedy decode and convert ids -> tokens
    predictions = []
    with torch.no_grad():
        for src_batch, _ in loader:
            src_batch = src_batch.to(device)
            seqs, _, _ = model.greedy_decode(src_batch, max_len=max_len)  # seqs: list[list[int]]
            for seq in seqs:
                # convert ids to tokens, trim at EOS if present
                if eos_idx in seq:
                    first = seq.index(eos_idx)
                    seq_trimmed = seq[:first]
                else:
                    seq_trimmed = seq
                tokens = [id2tok.get(i, PAD_TOKEN) for i in seq_trimmed]
                predictions.append(tokens)

    # ensure we have prediction per-row
    if len(predictions) != len(input_data):
        # pad/truncate (shouldn't happen under normal circumstances)
        if len(predictions) < len(input_data):
            predictions.extend([[]] * (len(input_data) - len(predictions)))
        else:
            predictions = predictions[:len(input_data)]

    out_df = input_data.copy()
    out_df['output_path'] = predictions
    return out_df
