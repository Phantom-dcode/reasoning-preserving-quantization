# Reasoning-Preserving Low-Bit Quantization for Edge Language Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.44.2-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Reproducible on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

Sensitivity-Aware Mixed Precision, Verification-Gated Fallback, and Reasoning-Calibrated Post-Training Quantization for preserving chain-of-thought reasoning under aggressive low-bit model compression.

## 📋 Problem Statement

Post-training quantization (PTQ) to 4-bit is the standard technique for deploying language models on edge hardware. While general-purpose surveys treat low-bit quantization as a "solved" compression problem on standard NLP benchmarks, **multi-step reasoning tasks (mathematics, logic) degrade disproportionately**. 

**The critical finding:** quantized reasoning models don't self-correct—they generate *longer* chains of thought without becoming more correct, indicating that quantization damages the reasoning process itself.

**The gap:** existing PTQ methods (AWQ, GPTQ, SmoothQuant, bitsandbytes NF4) calibrate against generic text perplexity. None target *reasoning-fidelity preservation* as an explicit objective, allocate precision based on reasoning-propagation sensitivity, or provide runtime mechanisms to detect and correct degraded reasoning traces.

## 🎯 Solution: Three Complementary Contributions

### Contribution 1: Reasoning-Sensitivity-Aware Mixed Precision
- Measure per-layer sensitivity via lightweight **noise-injection probe** on calibration reasoning traces
- Identify which transformer blocks matter most for propagating chain-of-thought state
- Selectively keep only top-quartile layers at 8-bit; quantize the rest to 4-bit
- **Benefit:** recover accuracy with minimal memory overhead vs. uniform 4-bit

### Contribution 2: Chain-of-Thought Verification-Gated Fallback
- Generate answers with the cheap 4-bit model by default
- Apply free, rule-based **arithmetic-consistency verifier** to CoT output
- Escalate only flagged queries to higher-precision re-run
- **Benefit:** keep average-case cost near 4-bit while providing safety net for degraded reasoning

### Contribution 3: Reasoning-Trace-Calibrated Quantization
- Compute activation quantization statistics from **multi-step reasoning chains**, not generic text
- Align quantization error objective with the activation distributions reasoning actually exercises
- Demonstrate via controlled activation-clamp comparison (identical bit-width, varying calibration content)
- **Benefit:** preserve fidelity in the activation regions that reasoning depends on

## ✨ Key Features

- ✅ **Fully reproducible on free-tier Google Colab** (T4 GPU, ~2–3 hours end-to-end)
- ✅ **No manual data uploads** — all datasets loaded programmatically from HuggingFace Hub
- ✅ **Open-weight, edge-deployable model** — Qwen2.5-1.5B-Instruct (production-ready architecture)
- ✅ **Modular design** — each contribution tested independently and combined
- ✅ **Comprehensive evaluation** — accuracy, CoT length, memory footprint, escalation rate, Pareto tradeoff analysis
- ✅ **Ablation study** — sensitivity-probe layer-budget tradeoff analysis
- ✅ **Transparent limitations** — explicit threats-to-validity discussion

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- GPU (T4 or better; free-tier Colab works)
- Internet access (data from HuggingFace Hub)

### Installation & Execution

**Option 1: Google Colab (Recommended)**
1. Open `2_reasoning_preserving_quantization.ipynb` in [Google Colab](https://colab.research.google.com)
2. Click **Runtime → Change runtime type → GPU**
3. Run all cells top-to-bottom (no setup needed — pip installs are in Cell 2)
4. Results (CSV tables, Pareto plot, ablation data) saved to `/content/` automatically

**Option 2: Local Environment**
```bash
git clone https://github.com/yourusername/reasoning-preserving-quantization.git
cd reasoning-preserving-quantization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -U transformers==4.44.2 datasets==2.21.0 accelerate==0.33.0 \
    bitsandbytes==0.43.1 sentencepiece==0.2.0 evaluate==0.4.2

# Run notebook (requires jupyter)
jupyter notebook 2_reasoning_preserving_quantization.ipynb
```

## 📊 Dataset & Model

| Property | Value |
|----------|-------|
| **Benchmark** | GSM8K (Cobbe et al., 2021) — grade-school multi-step math word problems |
| **Model** | Qwen2.5-1.5B-Instruct (open-weight, HuggingFace) |
| **Eval split size** | 60 examples (modest for free-tier; raise for full runs) |
| **Calibration sets** | 40 reasoning traces + 40 generic questions (train split, disjoint) |
| **Quantization baseline** | bitsandbytes NF4 (double-quantized, representative of production PTQ) |

## 📖 Notebook Structure

| Section | Purpose |
|---------|---------|
| **0. Environment** | Install exact library versions (transformers, bitsandbytes, etc.) |
| **1. Data Acquisition** | Load GSM8K from HuggingFace, split into eval/calibration/control sets |
| **2. Model Stack** | Define loaders for FP16, 4-bit NF4, and 8-bit configurations |
| **3. Baselines** | Evaluate full-precision FP16 and uniform 4-bit NF4 on eval set |
| **4. Contribution 1** | Sensitivity probe → rank layers → mixed-precision allocation → evaluate |
| **5. Contribution 2** | Arithmetic-consistency verifier → gate higher-precision fallback → evaluate |
| **6. Contribution 3** | Compute reasoning-calibrated vs. generic-calibrated activation ranges → evaluate both |
| **7. Combined System** | Stack all three contributions → evaluate with verification gate |
| **8. Results Table** | Aggregate all results → pandas DataFrame → export CSV and Pareto plot |
| **9. Ablation** | Layer-budget sensitivity — vary fraction of high-precision layers, measure accuracy tradeoff |
| **10. Threats to Validity** | Document known limitations and approximations |

## 📈 Evaluation Metrics

- **Exact-match accuracy (%)** — percentage of 60 eval questions answered correctly
- **Mean chain-of-thought length (tokens)** — checks for "thinks longer, not better" degradation
- **Approximate parameter memory (GB)** — reported parameter count × element size (note: actual on-disk 4-bit is smaller due to packing)
- **Generation time (seconds)** — mean time per query (Contribution 2 includes fallback re-run time if escalated)
- **Verification escalation rate (%)** — fraction of queries requiring higher-precision re-run (Contribution 2)
- **Pareto accuracy-vs-memory frontier** — scatter plot showing all configurations' tradeoff envelope

## 🔧 How to Customize

### Adjust evaluation set size
```python
N_EVAL = 100  # change from 60 to larger value
```

### Adjust calibration set size
```python
N_CALIB = 50  # change from 40
```

### Change the model
```python
MODEL_ID = "meta-llama/Llama-2-7b-hf"  # or any HF model supporting CausalLM
```

### Adjust layer-protection fraction (Contribution 1)
```python
K_HIGH_PRECISION = len(sensitivities) // 3  # protect top 33% instead of 25%
```

### Adjust verifier tolerance (Contribution 2)
```python
if result is None or abs(result - c) > 0.05:  # tolerance from 1e-2 to 0.05
    return False
```

### Adjust activation clip percentile (Contribution 3)
```python
ranges[name] = torch.quantile(all_vals, 0.99)  # 99th instead of 99.5th percentile
```

## 📄 Manuscript & Paper

- **Title:** "Reasoning-Preserving Low-Bit Quantization for Edge Language Models: Sensitivity-Aware Mixed Precision, Verification-Gated Fallback, and Reasoning-Calibrated Post-Training Quantization"
- **File:** `2_manuscript_reasoning_preserving_quantization.docx` (companion to this notebook)
- **Status:** Methods and threats-to-validity finalized; results section (Section 5) populated by running this notebook end-to-end

## ⚠️ Known Limitations & Threats to Validity

1. **Mixed precision via post-load re-cast, not fused kernels**
   - The notebook proves the accuracy effect by casting selected layers to fp16 after 4-bit loading
   - This does NOT deliver the inference-speed benefit a production fused mixed-precision kernel would
   - For deployment, implement via library-native per-layer bit-width assignment (AWQ, GPTQ)

2. **Activation-clamping as calibration proxy**
   - Contribution 3 uses forward-pass activation statistics to derive clipping ranges, not a full GPTQ/AWQ scale-and-zero-point recalibration
   - The direction and existence of the reasoning-calibration effect should transfer to full recalibration
   - Magnitude should be re-validated with production PTQ tooling before sign-off

3. **Verifier coverage limits**
   - Only catches reasoning errors expressed as explicit `A op B = C` statements
   - Internally consistent but incorrect derivations will pass (false negatives)
   - Optimistic default if no arithmetic statements are found

4. **Small model and eval set**
   - Results on Qwen2.5-1.5B and 60 examples for free-tier reproducibility
   - Reasoning-degradation literature shows effect severity varies by model family, size, and training regime
   - Validate on additional models (7B+) and benchmarks (MATH-500, GPQA-Diamond, LiveCodeBench) before generalizing

## 📁 Project Structure

```
.
├── 2_reasoning_preserving_quantization.ipynb  # Main executable notebook (34 cells)
├── 2_manuscript_reasoning_preserving_quantization.docx  # Companion manuscript
├── README.md                                   # This file
├── LICENSE                                     # MIT License
├── requirements.txt                            # Pip package list
└── outputs/                                    # Generated files (after running notebook)
    ├── quantization_reasoning_results.csv     # Main results table
    ├── pareto_accuracy_memory.png            # Accuracy-vs-memory scatter plot
    └── ablation_mixed_precision_budget.csv   # Layer-budget ablation table
```

## 🔍 Results (Run the notebook to generate)

Once executed, the notebook generates:

| File | Contents | Purpose |
|------|----------|---------|
| `quantization_reasoning_results.csv` | Accuracy, CoT length, memory for all 7 configurations | Main results table (Section 5 of manuscript) |
| `pareto_accuracy_memory.png` | Scatter plot of accuracy (y) vs. memory footprint (x) | Accuracy-memory tradeoff visualization |
| `ablation_mixed_precision_budget.csv` | k_layers, frac_8bit, accuracy for budget points {0, 0.125, 0.25, 0.5} | Contribution 1 layer-budget sensitivity |

## 🛠️ Requirements

```
transformers==4.44.2
datasets==2.21.0
accelerate==0.33.0
bitsandbytes==0.43.1
sentencepiece==0.2.0
evaluate==0.4.2
torch>=2.0
numpy
pandas
matplotlib
```

See `requirements.txt` for exact pinned versions.

## 📚 Related Work

This project directly addresses the gap between:

1. **Edge-deployment literature** (surveys on edge LLMs, PTQ frameworks) — treats quantization as a solved general-purpose problem
2. **Reasoning-degradation literature** (recent empirical studies) — documents severe reasoning loss under aggressive quantization but proposes no deployment-ready mitigation

Key citations:
- Cobbe et al. (2021) — GSM8K benchmark introduction
- Lin et al. (2024) — AWQ (activation-aware weight quantization)
- Frantar et al. (2023) — GPTQ (accurate post-training quantization)
- "Quantization Meets Reasoning" (2025) — reasoning-specific quantization analysis
- "Quantization Hurts Reasoning?" (2025) — empirical study on reasoning degradation

## 🤝 Contributing

Contributions welcome! Please:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/your-contribution`)
3. Test your changes on Colab or locally
4. Submit a pull request with a clear description

**Ideas for extension:**
- Adapt sensitivity probe to other reasoning benchmarks (MATH-500, GPQA-Diamond)
- Test on larger models (7B+) to validate small-model findings
- Implement true fused mixed-precision kernels for production deployment
- Extend verifier to catch more classes of reasoning errors
- Integrate with GPTQ/AWQ for full PTQ recalibration on reasoning traces

## ✉️ Citation

If you use this work, please cite:

```bibtex
@article{logesh2025reasoning_preserving_quantization,
  author = {[Your Name]},
  title = {Reasoning-Preserving Low-Bit Quantization for Edge Language Models: Sensitivity-Aware Mixed Precision, Verification-Gated Fallback, and Reasoning-Calibrated Post-Training Quantization},
  journal = {[Journal/Conference Name]},
  year = {2025},
  note = {Companion notebook: https://github.com/yourusername/reasoning-preserving-quantization}
}
```

## 📧 Contact

- **Author:** [Your Name]
- **Email:** [your.email@example.com]
- **GitHub:** [@yourusername](https://github.com/yourusername)
- **Institution:** [Your University/Organization]

## 📜 License

This project is licensed under the MIT License — see `LICENSE` file for details.

---

## 🎓 Quick Links

| Resource | Link |
|----------|------|
| **Notebook** | `2_reasoning_preserving_quantization.ipynb` (this repo) |
| **Manuscript** | `2_manuscript_reasoning_preserving_quantization.docx` (this repo) |
| **Dataset (GSM8K)** | https://huggingface.co/datasets/openai/gsm8k |
| **Model (Qwen2.5-1.5B)** | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |
| **Transformers Docs** | https://huggingface.co/docs/transformers |
| **bitsandbytes Docs** | https://github.com/TimDettmers/bitsandbytes |

---

**Last updated:** 2025-01-17  
**Status:** Reproducible on free-tier Colab ✅ | Results section pending ⏳
