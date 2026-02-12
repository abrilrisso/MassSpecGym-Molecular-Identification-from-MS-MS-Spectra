import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from massspecgym.data import RetrievalDataset, MassSpecDataModule
from massspecgym.data.transforms import SpecTokenizer, MolFingerprinter
from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel

# 1. ARCHITECTURE COMPONENTS

class FourierFeatures(nn.Module):
    """
    Fourier Features for m/z positional encoding.
    Maps low-dimensional input coordinates (m/z) to a higher-dimensional feature space to help the Transformer capture high-frequency patterns in the spectral data.
    """
    def __init__(self, output_dim, sigma=1.0): 
        super().__init__()
        self.num_freqs = output_dim // 2
        self.register_buffer('B', torch.randn(self.num_freqs) * sigma)

    def forward(self, x):
        projected = 2 * math.pi * (x / 1000.0) * self.B
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class SpectralTransformerEncoder(nn.Module):
    """
    Spectral Transformer architecture for embedding Mass Spectrometry data.
    Components:
    - Fourier Features for m/z encoding.
    - Linear layer for log-transformed intensity.
    - Global conditioning using the precursor m/z.
    - Transformer layers for peak interaction.
    - Global attention pooling to generate a single molecular fingerprint embedding.
    """
    def __init__(self, d_model=256, nhead=4, num_layers=2, out_channels=4096, dropout=0.1):
        super().__init__()
        self.fourier_dim = d_model // 2
        self.mz_enc = FourierFeatures(output_dim=self.fourier_dim, sigma=10.0)
        self.int_enc = nn.Linear(1, d_model - self.fourier_dim)
        self.precursor_proj = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attention_pool = nn.Linear(d_model, 1) # Attention Pooling
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, out_channels)
        )

    def forward(self, x_spec, precursor_mz):
        mz = x_spec[:, :, 0:1]
        intensity = torch.log1p(x_spec[:, :, 1:2]) # Log-transform for stability
        mz_emb = self.mz_enc(mz)
        int_emb = self.int_enc(intensity)
        peak_embs = torch.cat([mz_emb, int_emb], dim=-1)
        
        if precursor_mz.dim() > 1: 
            precursor_mz = precursor_mz.squeeze(-1)
        precursor_mz_norm = precursor_mz.unsqueeze(-1).float() / 1000.0
        prec_feat = self.precursor_proj(precursor_mz_norm)
        
        x = peak_embs + prec_feat.unsqueeze(1)
        
        x_out = self.transformer(x)
        
        attn_weights = torch.softmax(self.attention_pool(x_out), dim=1)
        global_repr = torch.sum(x_out * attn_weights, dim=1)
        
        return self.head(global_repr)

# 2. LIGHTNING MODULE

class MyRetrievalTransformer(RetrievalMassSpecGymModel):
    """
    LightningModule for Spectral Retrieval optimized for Hit Rate.
    
    This model utilizes InfoNCE loss to maximize the cosine similarity between the predicted spectral embedding and the correct molecular candidate.
    """
    def __init__(self, d_model=256, nhead=4, num_layers=2, out_channels=4096, lr=1e-4, temp=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.model = SpectralTransformerEncoder(d_model, nhead, num_layers, out_channels)

    def forward(self, x, precursor_mz):
        return self.model(x, precursor_mz)

    def step(self, batch: dict, stage: Stage) -> dict:
        """
        Implements the training step using InfoNCE Loss.
        """
        x = batch["spec"]
        precursor_mz = batch["precursor_mz"]
        candidates = batch["candidates_mol"]
        batch_ptr = batch["batch_ptr"] # Number of candidates per query in the batch
        labels = batch["labels"]

        fp_pred = self.forward(x, precursor_mz) # Generate predicted fingerprints
        
        fp_pred = F.normalize(fp_pred, p=2, dim=-1) # L2 Normalization for Cosine Similarity calculation
        candidates = F.normalize(candidates, p=2, dim=-1)

        fp_pred_repeated = fp_pred.repeat_interleave(batch_ptr, dim=0)
        cos_sim = (fp_pred_repeated * candidates).sum(dim=-1)
        
        logits = cos_sim / self.hparams.temp
        
        # InfoNCELoss
        batch_indices = torch.arange(len(batch_ptr), device=logits.device).repeat_interleave(batch_ptr)
        exp_logits = torch.exp(logits)
        denominators = torch.zeros(len(batch_ptr), device=logits.device, dtype=logits.dtype)
        denominators.scatter_add_(0, batch_indices, exp_logits)
        log_denominators = torch.log(denominators + 1e-10)
        
        pos_logits = logits[labels.bool()]
        loss = (log_denominators - pos_logits).mean()

        return {"loss": loss, "scores": cos_sim}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

# 3. EXECUTION

if __name__ == "__main__":
    print(f"\n{'='*60}\n>>> MODEL: RETRIEVAL (HIT RATE OPTIMIZATION) \n{'='*60}\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--d_model", type=int, default=256)
    args = parser.parse_args()

    pl.seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    dataset = RetrievalDataset(
        spec_transform=SpecTokenizer(n_peaks=200),
        mol_transform=MolFingerprinter(fp_size=4096)
    )
    data_module = MassSpecDataModule(
        dataset=dataset, 
        batch_size=args.batch_size, 
        num_workers=8
    )
    
    model = MyRetrievalTransformer(d_model=args.d_model, out_channels=4096)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_retrieval",
        filename="best_retrieval_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )

    trainer = Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=args.epochs,
        callbacks=[
            TQDMProgressBar(), 
            EarlyStopping(monitor="val_loss", patience=5, mode="min"), 
            checkpoint_callback
        ],
        precision="bf16-mixed"
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")