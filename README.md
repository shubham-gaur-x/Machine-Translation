
# Hindi-English Machine Translation: A Comparative Study of Pre-trained and Custom Transformer Models

## Introduction
This project explores and compares two transformer-based approaches for English-to-Hindi machine translation:  
1. A **custom transformer model** built from scratch.  
2. A **pre-trained transformer model** fine-tuned for the same task.

The study evaluates the performance of both models using the BLEU metric, providing insights into the effectiveness of model customization versus leveraging pre-trained solutions for translation tasks.

---

## Features
- **Custom Transformer Model**: Built using PyTorch with an encoder-decoder architecture for English-to-Hindi translation.  
- **Pre-Trained Model**: Utilizes the Helsinki-NLP/opus-mt-en-hi model from Hugging Face, fine-tuned on the same dataset.  
- **Dataset**: English-Hindi parallel corpus from ManyThings.org.  
- **Evaluation**: Comparative analysis using BLEU scores, sentence-level visualization, and token-level accuracy.  

---

## Dataset
- **Source**: [ManyThings.org Parallel Corpus](http://www.manythings.org/anki/).  
- **Size**: 3,061 sentence pairs.  
- **Preprocessing**:  
  - Tokenization using Helsinki-NLP/opus-mt-en-hi tokenizer.  
  - Padding and truncation to a maximum length of 128 tokens.  
  - Label masking for padded tokens.

---

## Project Structure
```
shakespeare_qa_model/
├── data/
│   ├── english_hindi.txt       # Dataset file
├── models/
│   ├── custom_transformer.py   # Custom transformer model implementation
│   ├── pre_trained_model.py    # Pre-trained transformer model setup
├── notebooks/
│   ├── analysis.ipynb          # Analysis and visualizations
├── scripts/
│   ├── train_custom.py         # Training script for custom model
│   ├── evaluate_models.py      # BLEU score evaluation
├── qa_app.py                   # Streamlit app for interactive testing
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## Setup and Installation

### Prerequisites
- Python 3.8 or higher.
- Install dependencies using pip:  
  ```bash
  pip install -r requirements.txt
  ```

### Run the Project
1. **Train the Custom Model**:  
   ```bash
   python scripts/train_custom.py
   ```
2. **Evaluate Models**:  
   ```bash
   python scripts/evaluate_models.py
   ```
3. **Launch the Streamlit App**:  
   ```bash
   streamlit run qa_app.py
   ```

---

## Results
- **Custom Model BLEU Score**: 84.79  
- **Pre-trained Model BLEU Score**: 0.14  

### Observations:
- The **custom model** outperforms the pre-trained model, highlighting the importance of dataset-specific training.  
- Pre-trained models are better suited for scenarios with extensive training data or multilingual tasks.

---

## Visualization
- **Sentence-Level BLEU Score Comparison**  
- **Token-Level Accuracy**  
- **Translation Examples**  

---

## Business Insights
1. **Custom Models**: Provide high accuracy for domain-specific tasks but require adequate data and resources.  
2. **Pre-trained Models**: Offer quick deployment with a strong baseline for general-purpose translation tasks.

---

## Future Work
- Incorporate larger and more diverse datasets for improved custom model performance.  
- Fine-tune pre-trained models for better domain-specific accuracy.  
- Explore hybrid approaches combining custom and pre-trained models.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- [ManyThings.org Parallel Corpus](http://www.manythings.org/anki/) for providing the dataset.  
- [Hugging Face](https://huggingface.co/) for the pre-trained transformer model.  
