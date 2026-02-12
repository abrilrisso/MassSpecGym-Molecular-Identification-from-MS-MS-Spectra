# MassSpecGym: Molecular Identification from MS/MS Spectra

This repository contains a professional implementation of a **Transformer-based pipeline** for molecular identification using tandem mass spectrometry (MS/MS) data. This project was developed as part of the **MassSpecGym** benchmark, focusing on mapping high-dimensional spectral data to chemical structures.


## 🧪 Project Overview
Mass spectrometry-based metabolomics faces a massive "dark matter" problem: thousands of detected signals remain unidentified. This project implements two cutting-edge computational strategies to solve this:

1. **Retrieval Task**: Learning a joint latent space between spectra and chemical fingerprints to rank true molecules within a candidate database.
2. **De Novo Task**: A Seq2Seq approach that "translates" spectral peaks directly into SMILES strings (molecular text representations).

## 🚀 Technical Implementation

### Spectral Transformer Encoder
Unlike traditional CNNs or RNNs, I implemented a **Spectral Transformer** that treats mass spectra as sets of peaks. Key architectural features include:
* **Fourier Features**: Captures high-precision mass differences (isotopes).
* **Log-Intensity Scaling**: Prevents dominant peaks from overshadowing smaller, informative ones.
* **Precursor Injection**: Adds global context (parent mass) to every spectral token.
* **Attention Pooling**: Learns to dynamically weight the most informative fragmentation peaks.


### Retrieval Strategy
I trained two specialized models for the retrieval task:
* **The Ranker**: Optimized via **Contrastive Loss (InfoNCE)** to push the ground truth molecule to the top of the candidate list.
* **The Reconstructor**: Optimized via **Focal Loss** to handle sparse molecular fingerprints and reconstruct chemical substructures.

### De Novo Generation
* **Architecture**: Encoder-Decoder Transformer.
* **Training**: Implemented **Teacher Forcing** for stable gradient convergence.
* **Inference**: Utilized **Beam Search** to explore multiple molecular paths and avoid error propagation in SMILES string generation.

## 📊 Results & Leaderboard Performance

My models demonstrate strong competitive performance, consistently outperforming the official MLP baselines and approaching State-of-the-Art (SOTA) results.

### Task 1: Retrieval (Ranking & Reconstruction)
| Model | Hit@1 (%) | Hit@10 (%) | Hit@20 (%) | Fingerprint F1 |
| :--- | :---: | :---: | :---: | :---: |
| **MLP Baseline** | 5.20 | 22.40 | 33.10 | 0.210 |
| **The Reconstructor (Ours)** | 6.50 | 25.80 | 35.20 | **0.283** |
| **The Ranker (Ours)** | **8.38** | **31.10** | **42.00** | 0.245 |
| **MS-Transformer (SOTA)** | 12.40 | 45.30 | 58.20 | 0.310 |

> **Key Takeaway**: The **Ranker** achieved a **61% improvement in Hit@1** over the MLP baseline, while the **Reconstructor** achieved the highest structural fidelity (F1: 0.283).

### Task 2: De Novo Molecular Generation
| Model | Top-1 Tanimoto | Accuracy (Exact) |
| :--- | :---: | :---: |
| MLP Baseline | 0.070 | 0.0% |
| **De Novo Decoder (Ours)** | **0.108** | 0.0% |
| MS-Transformer (SOTA) | 0.220 | 2.1% |

*Note: While De Novo exact matching is an extreme challenge (0.0% accuracy), our model achieved a **Top-1 Tanimoto Similarity of 0.108**, proving that the model learns relevant structural patterns and outperforms standard baselines.*

## 📁 Repository Structure
* `retrieval_notebook.ipynb`: Data preprocessing, Transformer training with InfoNCE/Focal Loss, and retrieval evaluation.
* `de_novo_notebook.ipynb`: Implementation of the Seq2Seq Transformer, SMILES tokenization, and Beam Search inference.
* `Slides_AdvancedAI.pdf`: Detailed presentation of the methodology and experimental results.

## 🛠️ Requirements
* Python 3.11+
* PyTorch
* RDKit (for molecular fingerprinting and SMILES validation)
* Pandas, NumPy, Matplotlib

---
**Author:** Abril Risso Matas  
**Framework:** MassSpecGym Benchmark
