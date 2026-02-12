import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel
from massspecgym.models.tokenizers import SpecialTokensBaseTokenizer, SmilesBPETokenizer
from massspecgym.definitions import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from massspecgym.models.base import Stage
from massspecgym.data import MassSpecDataModule, RetrievalDataset
from massspecgym.data.transforms import SpecTokenizer


# 1. ENCODER

class FourierFeatures(nn.Module):
    """
    Implements Fourier Feature mapping for high-frequency coordinate transformation.
    """
    def __init__(self, output_dim, sigma=1.0):
        super().__init__()
        self.num_freqs = output_dim // 2
        self.register_buffer('B', torch.randn(self.num_freqs) * sigma)
    
    def forward(self, x):
        # Normalization to prevent extreme values during projection
        projected = 2 * math.pi * torch.clamp(x / 1000.0, 0, 2) * self.B
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class PeakEncoder(nn.Module):
    """
    Transformer-based Encoder for Mass Spectrometry peaks.
    
    Key Components:
    - Fourier Features for m/z representation.
    - Linear projection for log-intensity.
    - Learnable Rank Embeddings to prioritize high-intensity peaks.
    - Precursor m/z injection for conditioning.
    """
    def __init__(self, d_model=256, nhead=4, num_layers=2, dropout=0.1, max_peaks=1000, use_rank_emb=True):
        super().__init__()
        # Features for m/z and linear projection for log-intensity
        self.mz_enc = FourierFeatures(d_model // 2, sigma=10.0)
        self.int_enc = nn.Linear(1, d_model - (d_model // 2))
        
        # Positional/Rank Embedding
        self.use_rank_emb = use_rank_emb
        self.max_peaks = max_peaks
        if self.use_rank_emb:
            self.rank_emb = nn.Embedding(max_peaks, d_model)
        
        # Projection for the precursor m/z
        self.precursor_proj = nn.Linear(1, d_model)
        
        # Layer normalization for stability
        self.input_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, spec, precursor_mz, src_key_padding_mask=None):
        mz = spec[:, :, 0:1]
        intensity = spec[:, :, 1:2]
        B_size, N_peaks, _ = spec.shape
        
        mz_emb = self.mz_enc(mz)
        int_emb = self.int_enc(torch.log1p(torch.clamp(intensity, 0, 1e6))) # Use log1p and clamp to avoid log(0) or extremely large values
        
        x = torch.cat([mz_emb, int_emb], dim=-1)
        
        # Add Rank Embeddings
        if self.use_rank_emb:
            positions = torch.arange(N_peaks, device=spec.device).unsqueeze(0).expand(B_size, -1)
            positions = positions.clamp(max=self.max_peaks - 1)
            x = x + self.rank_emb(positions)
        
        # Precursor Injection
        if precursor_mz is not None:
            if precursor_mz.dim() == 1:
                precursor_mz = precursor_mz.unsqueeze(-1)
            prec_norm = torch.clamp(precursor_mz.float() / 1000.0, 0, 2)
            prec_emb = self.precursor_proj(prec_norm).unsqueeze(1)
            x = x + prec_emb
        
        x = self.input_norm(x)
        
        memory = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return memory

 
# 2. DECODER

class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings for the SMILES sequence.
    Allows the model to understand the order of atoms in the string.
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.encoding = nn.Embedding(max_len, d_model)
        self.register_buffer("positions", torch.arange(max_len))
    
    def forward(self, x):
        seq_len = x.size(1)
        clamped_positions = self.positions[:seq_len].clamp(max=self.encoding.num_embeddings - 1)
        return self.encoding(clamped_positions.unsqueeze(0))


class AutoregressiveDecoder(nn.Module):
    """
    Standard Transformer Decoder.
    Predicts the next SMILES token based on the Encoder memory and previous tokens.
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1, max_len=200):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = LearnedPositionalEncoding(d_model, max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = x + self.pos_encoder(tgt)
        x = self.dropout(x)
        
        output = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.output_proj(output)


# 3. LIGHTNING MODULE

class SimpleDeNovoTransformer(DeNovoMassSpecGymModel):
    """
    Main PyTorch Lightning Module for De Novo Molecular Generation.
    
    Integrates:
    - PeakEncoder (Spectral processing)
    - AutoregressiveDecoder (SMILES generation)
    - Training logic (Teacher Forcing, Loss calculation)
    - Inference logic (Beam Search)
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        smiles_tokenizer: SpecialTokensBaseTokenizer,
        dropout: float = 0.1,
        max_smiles_len: int = 150,
        lr: float = 5e-4,
        peak_dropout_p: float = 0.1,
        label_smoothing: float = 0.1,
        warmup_ratio: float = 0.1,
        beam_size: int = 5,
        length_penalty_alpha: float = 0.8,
        max_peaks: int = 1000,
        top_ks: list = None,
        *args,
        **kwargs
    ):
        if top_ks is None:
            top_ks = [1, 5]
        super().__init__(top_ks=top_ks, *args, **kwargs)
        self.save_hyperparameters(ignore=['smiles_tokenizer'])
        
        self.smiles_tokenizer = smiles_tokenizer
        self.vocab_size = smiles_tokenizer.get_vocab_size()
        self.pad_id = smiles_tokenizer.token_to_id(PAD_TOKEN)
        self.sos_id = smiles_tokenizer.token_to_id(SOS_TOKEN)
        self.eos_id = smiles_tokenizer.token_to_id(EOS_TOKEN)
        self.max_len = max_smiles_len
        self.lr = lr
        
        self.peak_dropout_p = peak_dropout_p
        self.warmup_ratio = warmup_ratio
        self.beam_size = beam_size
        self.length_penalty_alpha = length_penalty_alpha
        
        self.max_k_eval = max(top_ks) if top_ks else 1
        
        self.encoder = PeakEncoder(
            d_model, nhead, num_encoder_layers, dropout,
            max_peaks=max_peaks, use_rank_emb=True
        )
        self.decoder = AutoregressiveDecoder(
            self.vocab_size, d_model, nhead, num_decoder_layers,
            dropout, max_smiles_len
        )
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id,
            label_smoothing=label_smoothing
        )
    
    def generate_src_padding_mask(self, spec):
        return spec.abs().sum(dim=-1) == 0
    
    def generate_causal_mask(self, sz, device):
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
    
    def _augment_spectrum(self, spec):
        """Applies Peak Dropout only during training to improve robustness."""
        if self.training and self.peak_dropout_p > 0:
            is_content = (spec.abs().sum(dim=-1) > 0).float()
            dropout_mask = torch.bernoulli(torch.full_like(is_content, self.peak_dropout_p))
            final_mask = dropout_mask * is_content
            spec = spec * (1 - final_mask.unsqueeze(-1))
        return spec
    
    def forward(self, batch):
        spec = batch["spec"]
        precursor_mz = batch.get("precursor_mz", None)
        mols = batch["mol"]
        
        spec = self._augment_spectrum(spec)
        
        encoded_mols = self.smiles_tokenizer.encode_batch(mols)
        tgt_ids = torch.tensor([e.ids for e in encoded_mols], device=self.device)
        
        tgt_in = tgt_ids[:, :-1]
        tgt_out = tgt_ids[:, 1:]
        
        src_mask = self.generate_src_padding_mask(spec)
        tgt_pad_mask = (tgt_in == self.pad_id)
        tgt_causal_mask = self.generate_causal_mask(tgt_in.size(1), self.device)
        
        memory = self.encoder(spec, precursor_mz, src_key_padding_mask=src_mask)
        logits = self.decoder(
            tgt_in, memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask
        )
        
        return logits, tgt_out
    
    def step(self, batch, stage: Stage = Stage.NONE):
        logits, tgt_out = self.forward(batch)
        
        loss = self.criterion(logits.reshape(-1, self.vocab_size), tgt_out.reshape(-1))
        
        if torch.isnan(loss):
            print(f"NaN detected in loss at stage {stage}")
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        mols_pred = None
        if stage not in self.log_only_loss_at_stages:
            mols_pred = self.decode_smiles(batch)
        
        return dict(loss=loss, mols_pred=mols_pred)
    
    def decode_smiles(self, batch):
        """
        Executes Beam Search decoding with Early Stopping.
        
        Strategy:
        1. Maintains top-k sequences (beams) at each step.
        2. Prunes low-probability paths.
        3. Stops when enough hypotheses (k) are finished or max length is reached.
        """
        spec = batch["spec"]
        precursor_mz = batch.get("precursor_mz", None)
        batch_size = spec.size(0)
        beam_size = self.beam_size
        device = self.device
        required_k = self.max_k_eval
        
        src_mask = self.generate_src_padding_mask(spec)
        
        with torch.inference_mode():
            # Encoder forward pass
            memory = self.encoder(spec, precursor_mz, src_key_padding_mask=src_mask)
            memory = memory.repeat_interleave(beam_size, dim=0)
            src_mask = src_mask.repeat_interleave(beam_size, dim=0)
            
            # Initialize Beams
            ys = torch.full((batch_size * beam_size, 1), self.sos_id, dtype=torch.long, device=device)
            beam_scores = torch.zeros((batch_size, beam_size), device=device)
            beam_scores[:, 1:] = float('-1e9') # Only the first beam starts with 0 score
            beam_scores = beam_scores.view(-1)
            
            finished_beams = [[] for _ in range(batch_size)]
            
            for step in range(self.max_len):
                # EARLY STOPPING CHECK
                finished_counts = torch.tensor([len(fb) for fb in finished_beams], device=device)
                has_enough_hyps = (finished_counts >= required_k)
                
                active_scores_view = beam_scores.view(batch_size, beam_size)
                has_active_beams = (active_scores_view > -1e8).any(dim=1)
                
                needs_work = (~has_enough_hyps) & has_active_beams
                if not needs_work.any():
                    break
                
                # Decoder Step
                tgt_mask = self.generate_causal_mask(ys.size(1), device)
                out = self.decoder(ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
                logits = out[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                
                next_scores = log_probs + beam_scores.unsqueeze(-1)
                next_scores = next_scores.view(batch_size, -1)
                
                topk_scores, topk_indices = next_scores.topk(beam_size, dim=1)
                beam_indices = torch.div(topk_indices, self.vocab_size, rounding_mode='floor')
                token_indices = topk_indices % self.vocab_size
                
                batch_offset = torch.arange(batch_size, device=device).unsqueeze(1) * beam_size
                global_beam_indices = batch_offset + beam_indices
                
                prev_ys = ys[global_beam_indices.view(-1)]
                new_tokens = token_indices.view(-1, 1)
                ys = torch.cat([prev_ys, new_tokens], dim=1)
                beam_scores = topk_scores.view(-1)
                
                # EOS Check
                current_tokens = token_indices.view(-1)
                is_eos = (current_tokens == self.eos_id)
                
                if is_eos.any():
                    eos_indices = torch.nonzero(is_eos, as_tuple=True)[0]
                    for idx in eos_indices:
                        batch_idx = idx.item() // beam_size
                        score = beam_scores[idx].item()
                        seq = ys[idx].tolist()
                        # Length Penalty application
                        lp = ((5 + len(seq)) / 6) ** self.length_penalty_alpha
                        final_score = score / lp
                        finished_beams[batch_idx].append((final_score, seq))
                        beam_scores[idx] = float('-1e9') # Invalidate this beam
            
            # 3. Process final results
            decoded_smiles_batch = []
            for i in range(batch_size):
                # Collect beams that didn't finish
                for j in range(beam_size):
                    idx = i * beam_size + j
                    if beam_scores[idx] > float('-1e8'):
                        seq = ys[idx].tolist()
                        lp = ((5 + len(seq)) / 6) ** self.length_penalty_alpha
                        score = beam_scores[idx].item() / lp
                        finished_beams[i].append((score, seq))
                
                finished_beams[i].sort(key=lambda x: x[0], reverse=True)
                max_k = self.max_k_eval
                top_hyps = finished_beams[i][:max_k]
                
                batch_preds = []
                for _, seq in top_hyps:
                    try:
                        s = self.smiles_tokenizer.decode(seq, skip_special_tokens=True)
                    except:
                        s = ""
                    batch_preds.append(s)
                
                # Fill with empty string if no hypotheses found or fewer than max_k
                if len(batch_preds) == 0:
                    batch_preds = [""] * max_k
                elif len(batch_preds) < max_k:
                    # Fill with the last valid prediction or empty string
                    fill_value = batch_preds[-1] if batch_preds else ""
                    batch_preds += [fill_value] * (max_k - len(batch_preds))
                
                decoded_smiles_batch.append(batch_preds)
            
            return decoded_smiles_batch
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        
        if hasattr(self, 'trainer') and self.trainer is not None:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = 1000
        
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]


# 4. EXECUTION

if __name__ == "__main__":
    pl.seed_everything(42)
    
    dataset = RetrievalDataset(
        spec_transform=SpecTokenizer(n_peaks=200),
        mol_transform=lambda x: str(x)
    )
    
    data_module = MassSpecDataModule(
        dataset=dataset,
        batch_size=32,  
        num_workers=4 
    )
    
    smiles_tokenizer = SmilesBPETokenizer(max_len=150)
    
    model = SimpleDeNovoTransformer(
        d_model=256,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        smiles_tokenizer=smiles_tokenizer,
        lr=3e-4, 
        peak_dropout_p=0.05, 
        label_smoothing=0.05,
        warmup_ratio=0.1,
        beam_size=5,
        length_penalty_alpha=0.6,
        max_peaks=1000,
        top_ks=[1, 5]
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_denovo",
        filename="denovo_model_10_epochs",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )
    
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=10,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, early_stop_callback],
        accumulate_grad_batches=2,
        precision=16 if torch.cuda.is_available() else 32 
    )
    
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
    
    print("Starting testing...")
    trainer.test(model, datamodule=data_module, ckpt_path="best")