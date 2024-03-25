import torch
from torch.nn import functional as F
from .transformer import Transformer, TransformerAdvanced
from .masks import create_padding_mask, create_padding_casual_mask, create_padding_rectangle_mask

class SupervoiceGPT(torch.nn.Module):
    def __init__(self, config):
        super(SupervoiceGPT, self).__init__()
        self.config = config
        self.n_input_tokens = config.tokenizer.vocab_size
        self.n_output_tokens = 1024 + 3 # 1024 phonemes + 3 special tokens

        # Embeddings
        self.input_embedding = torch.nn.Embedding(self.n_input_tokens, self.config.gpt.n_dim)
        torch.nn.init.normal_(self.input_embedding.weight, mean=0.0, std=0.02)
        self.output_embeddings = []
        for i in range(config.gpt.code_dim):
            d = self.config.gpt.n_dim // self.config.gpt.code_dim if self.config.gpt.code_mode == "combine" else self.config.gpt.n_dim
            self.output_embeddings.append(torch.nn.Embedding(self.n_output_tokens, d))
            torch.nn.init.normal_(self.output_embeddings[i].weight, mean=0.0, std=0.02)
        self.output_embeddings = torch.nn.ModuleList(self.output_embeddings)

        # Encoder Transformer
        self.encoder = Transformer(

            # Architecture
            n_heads = self.config.gpt.n_heads,
            n_layers = self.config.gpt.n_layers,
            n_dim = self.config.gpt.n_dim,
            n_dim_head = self.config.gpt.n_dim_head,
            n_dim_ffn = self.config.gpt.n_dim_ffn,

            # Dropout
            att_dropout = 0,
            ffn_dropout = 0.1,

            # Positional embedding
            position_embedding = 'alibi'
        )

        # Decoder Transformer
        self.decoder = TransformerAdvanced(
            
            # Architecture
            n_heads = self.config.gpt.n_heads,
            n_layers = self.config.gpt.n_layers,
            n_dim = self.config.gpt.n_dim,
            n_dim_head = self.config.gpt.n_dim_head,
            n_dim_ffn = self.config.gpt.n_dim_ffn,

            # Dropout
            att_dropout = 0,
            ffn_dropout = 0.1,

            # Positional embedding
            position_embedding = 'alibi'
        )

        # Prediction heads
        self.prediction_heads = []
        for i in range(config.gpt.code_dim):
            d = self.config.gpt.n_dim // self.config.gpt.code_dim if self.config.gpt.code_mode == "combine" else self.config.gpt.n_dim
            self.prediction_heads.append(torch.nn.Linear(d, self.n_output_tokens, bias=False))
        self.prediction_heads = torch.nn.ModuleList(self.prediction_heads)

        # Weight sharing
        for i in range(config.gpt.code_dim):
            self.output_embeddings[i].weight = self.prediction_heads[i].weight

    def forward(self, *,

        # Inputs (text)
        input, 
        input_lengths = None, 

        # Outputs
        output_tokens, 
        output_lengths = None, 
        
        # Target
        target_tokens = None, 
    ):

        # Check input
        assert len(input.size()) == 2, 'Input tensor shape should be [batch_size, sequence_length]'
        assert len(output_tokens.size()) == 3, 'Output tensor shape should be [batch_size, sequence_length, code]'
        assert input.size(0) == output_tokens.size(0), 'Input and output batch size should be the same'
        assert output_tokens.size(2) == self.config.gpt.code_dim, 'Output code dimension should be the same as the model'

        # Create input mask for self-attention which is useful for training on variable length sequences
        if input_lengths is None:
            input_lengths = torch.tensor([input.size(1)] * input.size(0), device = input.device)
        input_mask = create_padding_mask(input_lengths, input.size(1), device = input.device).unsqueeze(1)

        # Create output mask for self-attention which is useful for training on variable length sequences
        if output_lengths is None:
            output_lengths = torch.tensor([output_tokens.size(1)] * output_tokens.size(0), device = output_tokens.device)
        output_mask = create_padding_casual_mask(output_lengths, output_tokens.size(1), device = output_tokens.device).unsqueeze(1)

        # Create input-output masks for cross-attention which is useful for training on variable length sequences
        input_output_mask = create_padding_rectangle_mask(output_lengths, input_lengths, output_tokens.size(1), input.size(1), device = input.device).unsqueeze(1)

        # Input Embeddings
        input_embedded = self.input_embedding(input)
        
        # Output embeddings
        if self.config.gpt.code_mode == "mix":
            output_embedded = self.output_embeddings[0](output_tokens[:, :, 0])
            for i in range(1, self.config.gpt.code_dim):
                output_embedded += self.output_embeddings[i](output_tokens[:, :, i])
        elif self.config.gpt.code_mode == "combine":
            output_embedded = self.output_embeddings[0](output_tokens[:, :, 0])
            for i in range(1, self.config.gpt.code_dim):
                output_embedded = torch.cat((output_embedded, self.output_embeddings[i](output_tokens[:, :, i])), dim = -1)

        # Run an encoder
        latents = self.encoder(input_embedded, mask = input_mask)

        # Run an decoder
        decoded = self.decoder(latents, output_embedded, x_mask = input_mask, y_mask = output_mask, xy_mask = input_output_mask)

        # Run prediction head
        predicted = []
        for i in range(self.config.gpt.code_dim):
            if self.config.gpt.code_mode == "mix":
                predicted.append(self.prediction_heads[i](decoded))
            elif self.config.gpt.code_mode == "combine":
                predicted.append(self.prediction_heads[i](decoded[:, :, i * (self.config.gpt.n_dim // self.config.gpt.code_dim):(i + 1) * (self.config.gpt.n_dim // self.config.gpt.code_dim)]))

        # Compute loss if targets are provided
        if target_tokens is not None:
            loss = torch.tensor(0, device = input.device, dtype = torch.float32)
            for i in range(self.config.gpt.code_dim):
                loss += F.cross_entropy(predicted[i].view(-1, predicted[i].size(-1)), target_tokens[:, :, i].view(-1), ignore_index = 0)
            return predicted, loss

        return predicted

    @torch.no_grad()
    def generate(self, input, tokenizer, max_new_tokens = 128, temperature=1.0, top_k=None, deterministic = False, device="cpu"):
        ctx_input = torch.tensor([tokenizer.sequence_begin_token_id] + tokenizer.encode(input) + [tokenizer.sequence_end_token_id], device = device)
        ctx_output_tokens = torch.tensor([[tokenizer.sequence_begin_token_id, 0, 0, 0]], device = device)
        for _ in range(max_new_tokens):
            
            # Forward the model to get the logits for the index in the sequence
            logits = self(input = ctx_input.unsqueeze(0), output_tokens = ctx_output_tokens.unsqueeze(0))
            # print(logits.shape)
            
            # Pluck the logits at the final step and scale by desired temperature
            for i in range(self.config.gpt.code_dim):
                logits[i] = logits[i][:, -1, :] / temperature

            # Truncate the logits to only having generate tokens
            # logits = logits[:, :self.n_generate_tokens]

            # Remove padding values
            for i in range(self.config.gpt.code_dim):
                logits[i][:, 0] = -float('Inf')
                logits[i][:, tokenizer.sequence_begin_token_id] = -float('Inf')
                if i != 0:
                    logits[i][:, tokenizer.sequence_end_token_id] = -float('Inf')
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                for i in range(self.config.gpt.code_dim):
                    v, _ = torch.topk(logits[i], min(top_k, logits[i].size(-1)))
                    logits[i][logits[i] < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs = []
            for i in range(self.config.gpt.code_dim):
                probs.append(F.softmax(logits[i], dim=-1))
            
            # Sample from the distribution
            idx_next_token = []
            if deterministic:
                for i in range(self.config.gpt.code_dim):
                    idx_next_token.append(torch.argmax(probs[i], dim=-1))
            else:
                for i in range(self.config.gpt.code_dim):
                    idx_next_token.append(torch.multinomial(probs[i], num_samples=1)[0])
            idx_next_token = torch.cat(idx_next_token, dim=0)

            # Stop Tokens
            if idx_next_token[0] == tokenizer.sequence_end_token_id:
                break

            # Append Context
            ctx_output_tokens = torch.cat((ctx_output_tokens, idx_next_token.unsqueeze(0) - 3), dim=0)

        # Post-process
        return ctx_output_tokens.cpu()[1:,:]

    # def predict_next(self, input, output_tokens, output_durations, tokenizer, top_k = 10, device = "cpu"):

    #     # Context
    #     ctx_input = torch.tensor([tokenizer.sequence_begin_token_id] + tokenizer.encode(input) + [tokenizer.sequence_end_token_id], device = device).unsqueeze(0)
    #     ctx_output_tokens = torch.tensor([tokenizer.sequence_begin_token_id] + tokenizer.encode_phonemes(output_tokens), device = device).unsqueeze(0)
    #     ctx_output_durations = torch.tensor([0] + output_durations, device = device).unsqueeze(0)

    #     # Predict next token
    #     logits_token, logits_duration = self(input = ctx_input, output_tokens = ctx_output_tokens, output_durations = ctx_output_durations)
    #     logits_token.squeeze_(0)
    #     logits_duration.squeeze_(0)
    #     logits_token = logits_token[-1, :]
    #     logits_duration = logits_duration[-1, :]

    #     # Probabilities
    #     probs_token = F.softmax(logits_token, dim=-1)

    #     # Get top k
    #     probs_token, indices = torch.topk(probs_token, top_k)
        
    #     return probs_token.cpu().tolist(), tokenizer.decode_phonemes(indices.cpu().tolist())

    # def encode(self, *, input):
    #     input = input.unsqueeze(0)

    #     # Embeddings
    #     input_embedded = self.input_embedding(input)

    #     # Run an encoder
    #     return self.encoder(input_embedded).squeeze(0)