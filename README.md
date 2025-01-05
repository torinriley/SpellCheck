# **Spell-Correction Model**

A character-level sequence-to-sequence model for spelling correction, designed to transform misspelled words into their corrected forms. This implementation uses a Transformer-based architecture for accurate and efficient correction.

---

## **Features**
- Character-level tokenization for handling single-word corrections.
- Encoder-decoder Transformer architecture for flexible sequence modeling.
- Trained on synthetic misspelled-to-corrected word pairs.
- Supports real-time inference with beam search decoding for improved predictions.

---

## Dataset

| **Dataset**       | **Description**                                           |
|------------------|-----------------------------------------------------------|
| [Spell-Correction v1](https://huggingface.co/datasets/torinriley/spell-correction)   | Spell Correction dataset on hugging face     |
