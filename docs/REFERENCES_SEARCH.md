# Additional References — Search Results

## Batch 1: QLoRA/LoRA Fundamentals + Effective Rank + PEFT

### 1. Hu et al. (2022) — LoRA original
- **Title:** Low-Rank Adaptation of Large Language Models
- **Venue:** ICLR 2022
- **DOI/arXiv:** arXiv:2106.09685
- **Key idea:** Freezes pretrained weights, injects trainable rank decomposition matrices (B×A) into each Transformer layer. Reduces trainable params by 10000× with minimal quality loss.
- **Our use:** Foundation of our experimental protocol. We use QLoRA (4-bit quantized LoRA) and vary rank systematically.

### 2. Dettmers et al. (2023) — QLoRA
- **Title:** QLoRA: Efficient Finetuning of Quantized LLMs
- **Venue:** NeurIPS 2023
- **DOI/arXiv:** arXiv:2305.14314
- **Key idea:** 4-bit NormalFloat quantization + LoRA adapters. Finetunes 65B model on single 48GB GPU preserving full 16-bit performance. Introduces Double Quantization and Paged Optimizers.
- **Our use:** Exact method used in all QLoRA experiments. We use NF4 + double quantization.

### 3. Aghajanyan et al. (2021) — Intrinsic Dimensionality
- **Title:** Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
- **Venue:** ACL 2021
- **DOI/arXiv:** arXiv:2012.13255
- **Key idea:** Pretrained models have very low intrinsic dimension for fine-tuning. 200 trainable params (random projected) achieve 90% of full performance on MRPC. Larger models have lower intrinsic dim.
- **Our use:** Theoretical motivation for why low-rank adaptation works, and why effective rank matters as a predictor of regime transitions.

### 4. Roy & Vetterli (2007) — Effective Rank Definition
- **Title:** The Effective Rank: A Measure of Effective Dimensionality
- **Venue:** European Signal Processing Conference (EUSIPCO) 2007
- **Key idea:** Defines effective rank as exp(H(σ)) where H is Shannon entropy of normalized singular values. Smooth, continuous measure of intrinsic dimensionality that decreases with correlation.
- **Our use:** Exact definition we use for effective rank of B@A matrices. Primary predictor within each backbone.

### 5. Biderman et al. (2024) — LoRA Learns Less and Forgets Less
- **Title:** LoRA Learns Less and Forgets Less
- **Venue:** TMLR 2024 (Featured Certification)
- **DOI/arXiv:** arXiv:2405.09673
- **Key idea:** LoRA underperforms FFT in target domain but preserves base model's out-of-domain performance. LoRA mitigates forgetting more than regularization techniques. Maintains diverse generations.
- **Our use:** Direct predecessor. We extend "forgets less" to a pressure-dependent regime map showing WHEN it fails.

### 6. Liu et al. (2024) — DoRA
- **Title:** DoRA: Weight-Decomposed Low-Rank Adaptation
- **Venue:** ICML 2024
- **DOI/arXiv:** arXiv:2402.09353
- **Key idea:** Decomposes pretrained weight into magnitude + direction. LoRA for directional updates. Bridges gap between LoRA and full fine-tuning.
- **Our use:** Future work reference (alternative PEFT method with different pressure profile).

### 7. Adapala et al. (2025) — Anti-Ouroboros Effect
- **Title:** The Anti-Ouroboros Effect: Emergent Resilience in Large Language Models from Recursive Selective Feedback
- **Venue:** arXiv preprint, 2025
- **DOI/arXiv:** arXiv:2509.10509
- **Key idea:** Selective feedback mechanism in recursive training can REVERSE degradation (not just slow it). Gemma 2B on summarization task shows improvement. Contrasts with model collapse predictions.
- **Our use:** Related work. They use cumulative + filter (selective feedback). We use replace-without-filter (harshest protocol) and still find homeostasis at low rank.

---

## Batch 2: Recursive/Iterative Training + Self-Consuming Loops + Data Contamination

### 8. Shumailov et al. (2024) — Model Collapse (Nature)
- **Title:** AI models collapse when trained on recursively generated data
- **Venue:** Nature, Vol. 631, pp. 755-759, July 2024
- **DOI:** 10.1038/s41586-024-07566-y | arXiv:2305.17493
- **Key idea:** Indiscriminate use of model-generated content in training causes irreversible defects. Tails of original distribution disappear. Demonstrated in VAEs, GMMs, and LLMs.
- **Our use:** Primary reference for model collapse. We identify the pressure regimes where their predictions hold.

### 9. Alemohammad et al. (2024) — Self-Consuming Models Go MAD
- **Title:** Self-Consuming Generative Models Go MAD
- **Venue:** ICLR 2024
- **DOI/arXiv:** arXiv:2307.01850
- **Key idea:** Autophagous (self-consuming) loop degrades quality and diversity. Coined "Model Autophagy Disorder (MAD)." Without fresh real data, quality/diversity degrade each generation.
- **Our use:** Broader context for recursive training problem. Our QLoRA low-rank setting avoids MAD-like degradation.

### 10. Villalobos et al. (2024) — Will We Run Out of Data?
- **Title:** Will we run out of data? Limits of LLM scaling based on human-generated data
- **Venue:** ICML 2024 (Proc. Mach. Learn. Res. 235)
- **DOI/arXiv:** arXiv:2211.04325
- **Key idea:** Stock of public human text ~300 trillion tokens. Will be exhausted between 2026-2032. Motivates synthetic data usage and its risks.
- **Our use:** Introduction motivation. Why synthetic data pipelines are increasingly necessary, making recursive stability critical.

### 11. Dey & Donoho (2024) — Universality of π²/6
- **Title:** Universality of the π²/6 Pathway in Avoiding Model Collapse
- **Venue:** arXiv preprint, 2024
- **DOI/arXiv:** arXiv:2410.22812
- **Key idea:** Generalizes Gerstgrasser's accumulation result to exponential family models. Characterizes the universal growth schedule needed.
- **Our use:** Theoretical context. Our work is on a different axis (update pressure vs data accumulation).

### 12. Web contamination stats (for Introduction)
- 74.2% of newly published webpages contain AI-generated material (Ahrefs 2025 study)
- 30-40% of active web corpus is now synthetic (large-scale text analyses)
- Source: cited in arXiv:2511.05535 "A Computational Perspective on Model Collapse"
- **Our use:** Introduction framing, motivating urgency of the problem.

## Batch 3: TriviaQA Dataset + Factual Evaluation + Knowledge Retention

### 13. Joshi et al. (2017) — TriviaQA
- **Title:** TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
- **Venue:** ACL 2017
- **DOI/arXiv:** arXiv:1705.03551 | ACL Anthology P17-1147
- **Key idea:** 650K+ question-answer-evidence triples. 95K QA pairs authored by trivia enthusiasts with independently gathered evidence documents. Multiple valid answer aliases per question.
- **Our use:** Our evaluation dataset. 2000 train, 200 eval, K0 subset of 78 items. Public dataset satisfying EAAI replicability requirement.

### 14. Luo et al. (2023/2024) — Systematic Assessment of Factual Knowledge in LLMs
- **Title:** Systematic Assessment of Factual Knowledge in Large Language Models
- **Venue:** arXiv:2310.11638 (EMNLP 2024 likely)
- **Key idea:** Existing QA benchmarks have limited factual coverage. Proposes systematic probing beyond generic domains.
- **Our use:** Context for why factual QA (TriviaQA) is a valid but limited probe of knowledge retention.

### 15. Chen et al. (2024) — FACT-BENCH
- **Title:** Towards a Holistic Evaluation of LLMs on Factual Knowledge Recall
- **Venue:** arXiv:2404.16164
- **Key idea:** Benchmark covering 20 domains, 134 property types, 3 answer types, different popularity levels. Benchmarks 31 models from 10 families.
- **Our use:** Related evaluation methodology. Our K0 metric is simpler (single-domain factoid) but longitudinal across generations.

### 16. Luo et al. (2024) — Catastrophic Forgetting in LLM Tuning
- **Title:** Revisiting Catastrophic Forgetting in Large Language Model Tuning
- **Venue:** EMNLP 2024 Findings
- **DOI:** ACL Anthology 2024.findings-emnlp.249
- **Key idea:** Investigates causes of catastrophic forgetting during fine-tuning. Relevant to distinguishing our recursive degradation from classical catastrophic forgetting.
- **Our use:** Discussion. Our phenomenon is NOT classical catastrophic forgetting (same task, recursive), but the forgetting literature provides mechanistic context.

### 17. Huang et al. (2023) — Catastrophic Forgetting During Continual Fine-tuning
- **Title:** An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning
- **Venue:** arXiv:2308.08747
- **Key idea:** CF generally observed in 1B-7B models. Larger models forget MORE intensely (larger initial performance gap). LoRA partially mitigates.
- **Our use:** Connects to Biderman's finding. Our recursive protocol is a special case of continual fine-tuning (same task, synthetic data).

## Batch 4: Output Diversity Metrics + Distributional Drift + Text Degeneration

### 18. Li et al. (2016) — Distinct-n Metric
- **Title:** A Persona-Based Neural Conversation Model
- **Venue:** ACL 2016
- **DOI:** ACL Anthology P16-1094
- **Key idea:** Introduced distinct-1 and distinct-2 as diversity metrics (ratio of unique n-grams to total n-grams). Widely adopted for measuring generation diversity.
- **Our use:** Primary diversity metric. We track distinct-1 across generations to detect lexical homogenization in degradative regimes.

### 19. McCarthy & Jarvis (2010) — MTLD
- **Title:** MTLD, vocd-D, and HD-D: A Validation Study of Sophisticated Approaches to Lexical Diversity Assessment
- **Venue:** Behavior Research Methods, 42(2), 381-392
- **DOI:** 10.3758/BRM.42.2.381 | PubMed 20479170
- **Key idea:** MTLD = mean length of sequential word strings maintaining a criterion TTR. Robust to text length (unlike raw TTR). Validated as strongest LD index.
- **Our use:** Secondary diversity metric. MTLD collapse (2673→745) in r=128 reveals filler verbosity phenotype.

### 20. Holtzman et al. (2020) — Neural Text Degeneration
- **Title:** The Curious Case of Neural Text Degeneration
- **Venue:** ICLR 2020
- **DOI/arXiv:** arXiv:1904.09751
- **Key idea:** Maximum likelihood decoding leads to repetitive, degenerate text. Nucleus (top-p) sampling solves by truncating unreliable tail. Established that generation quality ≠ likelihood maximization.
- **Our use:** Context for why output drift matters. Even without degenerate decoding, recursive training can induce similar homogenization at the fine-tuning level.

### 21. Guo et al. (2024) — Large Language Models Suffer From Their Own Output
- **Title:** Large Language Models Suffer From Their Own Output
- **Venue:** arXiv:2311.16822
- **Key idea:** Self-consuming training initially improves quality/diversity but inevitably degenerates in diversity after a few generations. Rate depends on proportion of real vs generated data.
- **Our use:** Directly related. We show that low-rank constrains this degeneration (homeostatic regime), while high-rank permits it.

### 22. Zhang et al. (2025) — Output Diversity Collapse in Post-Training
- **Title:** Where Does Output Diversity Collapse in Post-Training?
- **Venue:** arXiv:2604.16027 (2025)
- **Key idea:** Post-trained models produce less varied outputs. Diversity collapse undermines inference-time scaling. Identifies WHERE in the pipeline diversity is lost.
- **Our use:** Related phenomenon. Our three-regime taxonomy (homeostatic / distributionally degraded / factually degradative) is a recursive-training analog.

## Batch 5: Models + Practical Governance + Training Stability

### 23. Qwen Team (2024) — Qwen 2.5 Technical Report
- **Title:** Qwen2.5 Technical Report
- **Venue:** arXiv:2412.15115 (December 2024)
- **Key idea:** Open-source LLM family from Alibaba. Sizes from 0.5B to 72B. Improved pre-training and post-training over Qwen2. Competitive with proprietary models.
- **Our use:** Primary backbone model (Qwen 2.5 1.5B-Instruct). Cite for architecture details.

### 24. Gemma Team (2025) — Gemma 3 Technical Report
- **Title:** Gemma 3 Technical Report
- **Venue:** arXiv:2503.19786 (March 2025)
- **Key idea:** Multimodal Gemma family. 1B to 27B parameters. Vision, 128K context, 140+ languages. Trained with distillation. Consumer hardware target.
- **Our use:** Secondary backbone (Gemma 3 1B IT). Cross-backbone validation with 10× lower threshold.

### 25. EU AI Act (2024) — Regulatory context
- **Key fact:** Entered into force August 2024. Article 10 requires documented data governance for high-risk AI.
- **Our use:** Discussion/motivation. Recursive training stability is relevant to compliance with data governance requirements for AI systems that use synthetic training data.

### 26. Synthetic data contamination stats (from multiple sources)
- 74.2% of newly published webpages contain AI-generated material (Ahrefs 2025)
- 30-40% of active web corpus is now synthetic (multiple 2024-2025 analyses)
- Models will exhaust public human text between 2026-2032 (Villalobos et al. 2024)
- **Our use:** Introduction framing for urgency of recursive training governance.

---

## Summary: Total References Identified (this search)

| # | Author | Year | Topic | Status |
|---|---|---|---|---|
| 1 | Hu et al. | 2022 | LoRA | ICLR paper |
| 2 | Dettmers et al. | 2023 | QLoRA | NeurIPS paper |
| 3 | Aghajanyan et al. | 2021 | Intrinsic dim | ACL paper |
| 4 | Roy & Vetterli | 2007 | Effective rank | EUSIPCO paper |
| 5 | Biderman et al. | 2024 | LoRA forgets less | TMLR paper |
| 6 | Liu et al. | 2024 | DoRA | ICML paper |
| 7 | Adapala et al. | 2025 | Anti-Ouroboros | arXiv preprint |
| 8 | Shumailov et al. | 2024 | Model collapse | Nature |
| 9 | Alemohammad et al. | 2024 | MAD | ICLR paper |
| 10 | Villalobos et al. | 2024 | Data limits | ICML paper |
| 11 | Dey & Donoho | 2024 | π²/6 | arXiv preprint |
| 12 | Joshi et al. | 2017 | TriviaQA | ACL paper |
| 13 | Luo et al. | 2023 | Factual assessment | arXiv/EMNLP |
| 14 | Chen et al. | 2024 | FACT-BENCH | arXiv |
| 15 | Luo et al. | 2024 | CF revisited | EMNLP Findings |
| 16 | Huang et al. | 2023 | CF empirical | arXiv |
| 17 | Li et al. | 2016 | Distinct-n | ACL paper |
| 18 | McCarthy & Jarvis | 2010 | MTLD | BRM journal |
| 19 | Holtzman et al. | 2020 | Text degeneration | ICLR paper |
| 20 | Guo et al. | 2024 | Self-consuming LLMs | arXiv |
| 21 | Zhang et al. | 2025 | Diversity collapse | arXiv |
| 22 | Qwen Team | 2024 | Qwen 2.5 | arXiv |
| 23 | Gemma Team | 2025 | Gemma 3 | arXiv |

**Plus from existing KB (already have PDFs):**
- Seddik 2024 (statistical LM collapse)
- Xu 2025 (probabilistic model collapse)
- Gerstgrasser 2024 (accumulation)
- Zibakhsh/ForTIFAI 2024 (TCE loss)
- Yi 2025 (verification retraining)
- Keisha 2025 (knowledge collapse)
- Dohmatob 2025 (strong model collapse)

**Total reference pool: ~30+ papers covering all sections.**
