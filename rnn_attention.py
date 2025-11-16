import torch
import torch.nn as nn

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
EMBEDDING_DIMENSION = 128
HIDDEN_DIMENSION = 512
NUMBER_OF_RNN_LAYERS = 2
OPTIMIZER = "adam"
VOCAB_SIZE = 46

class Seq2SeqAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

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
    def decoder_step(self, prev_token, prev_state, encoder_outputs):
        B, T, H = encoder_outputs.shape

        # 1. embed previous token
        emb = self.embedding(prev_token)  # (B, E)

        # 2. attention
        s_prev = prev_state[-1]  # (B, H)
        dec_part = self.W_a(s_prev).unsqueeze(1).expand(B, T, H)
        enc_part = self.U_a(encoder_outputs)
        e = torch.tanh(dec_part + enc_part)
        e = self.v_a(e).squeeze(-1)
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
    def decode(self, encoder_outputs, initial_state, target_tokens, teacher_forcing_ratio):
        B, Ty = target_tokens.shape
        outputs = []

        prev_token = target_tokens[:, 0]
        state = initial_state

        for t in range(1, Ty):
            logits, state, _ = self.decoder_step(prev_token, state, encoder_outputs)
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
        encoder_outputs, h_n = self.encoder(src)
        return self.decode(encoder_outputs, h_n, target, teacher_forcing_ratio)
