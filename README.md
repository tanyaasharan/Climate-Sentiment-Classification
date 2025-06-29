# Climate-Sentiment-Classification
Classifying corporate climate disclosures into Risk, Opportunity, or Neutral sentiments using traditional and deep learning NLP models.
## Project Overview 
This project tackles the task of sentiment classification within the climate finance domain. It uses a labeled dataset of corporate disclosures to determine whether a paragraph expresses a risk, an opportunity, or a neutral stance towards climate issues. The project explores a range of models from Naïve Bayes to CNNs and transformer-based TinyBERT.

## Data 
* Source: HuggingFace - climatebert/climate_sentiment
* Format: CSV, with columns: "sentence" and "label" (risk, opportunity, neutral)

## Models Trained 
| Model       | Type          | Description                                   |
| ----------- | ------------- | --------------------------------------------- |
| Naïve Bayes | Baseline      | CountVectorizer + Multinomial NB              |
| CNN         | Deep Learning | 1D CNN with dropout and ReLU for feature maps |
| TinyBERT    | Transformer   | Fine-tuned on domain-specific climate text    |

## Libraries Required
```
scikit-learn
nltk
torch / torchvision
transformers (Hugging Face)
pandas / numpy
matplotlib / seaborn
```
Install dependencies with:
```
pip install -r requirements.txt
```

## Methodology 
1. Preprocessing: tokenization, lowercasing, punctuation removal, stopword filtering
2. Naïve Bayes: evaluated with unigrams and bigrams using TF features
3. CNN: trained with 1D convolutions and max-pooling layers
4. TinyBERT: fine-tuned with huggingface/transformers Trainer API

## Results
| Model       | Accuracy |
| ----------- | -------- |
| Naïve Bayes | 0.80     |
| CNN         | 0.75     |
| TinyBERT    | 0.78     |

Naïve Bayes with bigram filtering performed the best, showing that lightweight models can compete with transformers on small, domain-specific datasets.

## Future Work
1. Fine-tune DistilBERT or RoBERTa on larger climate corpora
2. Incorporate domain knowledge into embeddings
