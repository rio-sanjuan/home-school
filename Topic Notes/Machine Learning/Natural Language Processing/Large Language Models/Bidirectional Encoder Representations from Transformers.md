Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning model designed for natural language understanding (NLU) tasks. It was introduced by Google in 2018 in the paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" by #JacobDevlin.

BERT revolutionized Natural Language Processing (NLP) by
1. Introducing a bidirectional understanding of language
2. Reducing the amount of labeled data needed for specific tasks
3. Achieving state-of-the-art results across numerous NLP benchmarks

It remains one of the foundational models in NLP and a building block for many modern advancements in the field.
## Key Features

1. **Bidirectional Contextual Understanding**: Unlike traditional language models that read text in one direction, BERT reads text in both directions simultaneously. This bidirectional approach helps BERT capture the context of words based on both preceding and following words in a sentence.
2. **Transformer Architecture**: BERT is based on the [[Transformer]] architecture, specifically the [[Transformer#Encoder]] part. The Transformer uses [[Attention]] mechanisms to focus on different parts of the input sequence and understand relationships between words regardless of their position in the text.
3. **Pre-training and Fine-tuning**: BERT is pre-trained on large amounts of unlabeled text using two self-supervised tasks. After pre-training, BERT can be fine-tuned on specific tasks (e.g. sentiment analysis, question answering) using labeled data
	1. [[Masked Language Model]]
	2. [[Next Sentence Prediction]]

## Variants

### RoBERTa

Robustly optimized version of BERT with better pre-training strategies
### DistilBERT

A lightweight and faster version of BERT with reduced size
### ALBERT

A more memory-efficient BERT with parameter sharing
### [[Generative Pre-trained Transformer]]

Focuses on text generation rather than understanding, based on a unidirectional [[Transformer]].
### BART and T5

Sequence-to-sequence models for text generation and translation tasks
