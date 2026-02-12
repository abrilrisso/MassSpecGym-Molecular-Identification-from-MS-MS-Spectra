import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from torchmetrics.classification import BinaryF1Score
from massspecgym.data import RetrievalDataset, MassSpecDataModule
from massspecgym.data.transforms import SpecTokenizer, MolFingerprinter

# 1. LOSS FUNCTIONS and ARCHITECTURAL COMPONENTS

class FocalBCEWithLogits(nn.Module):
    """
    Focal Loss implementation for multi-label classification (fingerprints).
    Addresses class imbalance by down-weighting well-classified examples and focusing the model on hard-to-learn bits.
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        # Standard binary cross entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)  # probability of being classified to the true class
        # Apply the focal factor
        loss = ((1.0 - pt) ** self.gamma) * bce 
        
        return loss.mean() if self.reduction == "mean" else loss


class FourierFeatures(nn.Module):
    """
    Fourier Features for m/z encoding.
    Maps low-dimensional input coordinates (m/z) to a higher-dimensional feature space to help the Transformer capture high-frequency patterns in the spectral data.
    """
    def __init__(self, output_dim, sigma=1.0): 
        super().__init__()
        self.num_freqs = output_dim // 2
        self.register_buffer('B', torch.randn(self.num_freqs) * sigma)

    def forward(self, x):
        # Normalize m/z values and project using random frequencies
        projected = 2 * math.pi * (x / 1000.0) * self.B
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class SpectralTransformerEncoder(nn.Module):
    """
    Main Spectral Encoder using a Transformer Architecture.
    Steps:
    1. Embeds peak m/z using Fourier Features and intensities using linear layers.
    2. Incorporates precursor m/z as a global conditioning feature.
    3. Processes peak sequences via Transformer layers.
    4. Aggregates information using an Attention Pooling mechanism.
    """
    def __init__(self, d_model=256, nhead=4, num_layers=2, out_channels=4096, dropout=0.1):
        super().__init__()
        self.mz_enc = FourierFeatures(d_model // 2, sigma=10.0)
        self.int_enc = nn.Linear(1, d_model - (d_model // 2))
        self.precursor_proj = nn.Linear(1, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.attention_pool = nn.Linear(d_model, 1) # Attention Pooling
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(d_model, out_channels)
        )

    def forward(self, x_spec, precursor_mz):
        mz_emb = self.mz_enc(x_spec[:, :, 0:1])
        int_emb = self.int_enc(torch.log1p(x_spec[:, :, 1:2]))
        peak_embs = torch.cat([mz_emb, int_emb], dim=-1)
        
        # Precursor Conditioning
        if precursor_mz.dim() > 1: 
            precursor_mz = precursor_mz.squeeze(-1)
        prec_emb = self.precursor_proj(precursor_mz.unsqueeze(-1).float() / 1000.0).unsqueeze(1)
        
        x = peak_embs + prec_emb
        
        x_out = self.transformer(x)
        
        attn_weights = torch.softmax(self.attention_pool(x_out), dim=1)
        global_repr = torch.sum(x_out * attn_weights, dim=1) 
        
        return self.head(global_repr)

# 2. LIGHTNING MODULE

class FingerprintPredictor(pl.LightningModule):
    """
    LightningModule for Molecular Fingerprint Prediction from MS/MS spectra.
    
    This module handles training, validation, and testing logic, focusing on the Instance-wise F1 Score as the primary performance metric.
    """
    def __init__(self, d_model=256, nhead=4, num_layers=2, out_channels=4096, lr=1e-4, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = SpectralTransformerEncoder(d_model, nhead, num_layers, out_channels)
        self.loss_fn = FocalBCEWithLogits(gamma=2.0)
        self.val_f1 = BinaryF1Score(multidim_average='samplewise')
        self.test_f1 = BinaryF1Score(multidim_average='samplewise')

    def forward(self, x, precursor_mz):
        return self.model(x, precursor_mz)

    def step(self, batch):
        x, precursor_mz = batch["spec"], batch["precursor_mz"]
        fp_logits = self.forward(x, precursor_mz)
        fp_true = batch["mol"].to(dtype=fp_logits.dtype) 
        loss = self.loss_fn(fp_logits, fp_true)
        return loss, fp_logits, fp_true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self.step(batch)
        self.log("val_loss", loss, on_epoch=True)
        self.val_f1.update(torch.sigmoid(logits), targets.long()) 
        return loss

    def on_validation_epoch_end(self):
        self.log("val_f1", self.val_f1.compute().mean(), prog_bar=True) 
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        loss, logits, targets = self.step(batch)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.test_f1.update(torch.sigmoid(logits), targets.long())

    def on_test_epoch_end(self):
        self.log("test_f1", self.test_f1.compute().mean(), prog_bar=True)
        self.test_f1.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)

# 3. EXECUTION

if __name__ == "__main__":
    print(f"\n{'='*60}\n>>> MODEL: FINGERPRINT PREDICTION (INSTANCE-WISE F1)\n{'='*60}\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--lr", type=float, default=5e-4)
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
    
    model = FingerprintPredictor(
        d_model=args.d_model, 
        out_channels=4096, 
        lr=args.lr
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_fingerprint",
        filename="best_f1_model",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        verbose=True
    )

    trainer = Trainer(
        accelerator="gpu", 
        devices=1, 
        max_epochs=args.epochs,
        callbacks=[
            TQDMProgressBar(), 
            EarlyStopping(monitor="val_f1", patience=5, mode="max"), 
            checkpoint_callback
        ],
        precision="bf16-mixed"
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module, ckpt_path="best")