# Our solution to LLMs4Subjects (SemEval'25 task #5) 🚀
This repo contains both files from LLMs4Subjects' original repository and our files 📂

## Authors
- [Vincenzo Avantaggiato](https://github.com/VincenzoAvantaggiato)
- [Michele Cazzola](https://github.com/MicheleCazzola)
- [Andrea Delli](https://github.com/RonPlusSign)

## General information
**Course**: `Large Language Models` (`Polytechnic of Turin`)  
**Academic Year**: 2024-25. Developed between January and February 2025.  
**Teachers**: Flavio Giobergia, Riccardo Coppola.  
**Topic**: development of a LLM-based method to perform authomatic subject tagging of technical documents from Leibniz University’s Technical Library (TIBKAT).  

## Our approach
We explore three different approaches of increasing complexity:

1. **Embedding-based retrieval**: Generate embeddings for both documents and tags using an encoder LLM, compute cosine similarity, and assign the top-k highest scoring tags. 🧩

2. **Fine-tuned embedding model**: Fine-tune a transformer-based encoder and apply the previous embedding-based retrieval approach. 🔧

3. **Binary classification model**: Train a binary MLP that, given a document and a tag, predicts a similarity score, then selects the top-k highest scoring tags. 🧠

## Repository Structure 🗂️
In the following, we'll report only the files we created/modified:

```
├── images/                          # Directory containing images used in the project 🖼️
├── results/                         # Directory where results are stored 📊
├── main.ipynb                       # Main Jupyter notebook for running experiments 📓
├── embedding_similarity_tagging.py  # Script for tagging using embedding similarity 🏷️
├── finetune_sentence_transformer.py # Script for fine-tuning a sentence transformer model 🔧
├── binary_classifier.py             # Script for defining the multi-layer perceptron 🧠
├── binary_mlp.py                    # Script for training the multi-layer perceptron 🏋️
├── performances.py                  # Script for evaluating model performances 📈
├── plots.py                         # Script for generating plots 📉
├── README.md                        # Project documentation 📃
├── requirements.txt                 # List of dependencies required for the project 📋
```

## Best results obtained 🏆
The best results are obtained by fine-tuning for three epochs using MultipleNegativesRankingLoss 🔄
| k  | Precision (%) | Recall (%) | F1 Score (%) |
|----|---------------|------------|--------------|
| 5  | 9.74          | 21.70      | 13.28        |
| 10 | 6.30          | 27.12      | 10.14        |
| 15 | 4.73          | 29.75      | 8.11         |
| 20 | 3.87          | 32.16      | 6.88         |
| 25 | 3.30          | 33.89      | 5.98         |
| 30 | 2.91          | 35.68      | 5.36         |
| 35 | 2.61          | 36.99      | 4.86         |
| 40 | 2.36          | 37.83      | 4.43         |
| 45 | 2.16          | 38.92      | 4.08         |
| 50 | 1.99          | 39.76      | 3.87         |
