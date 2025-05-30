# 🧠 Document Categorization using Knowledge Distillation (BERT → Custom CNN)

This project demonstrates an efficient **document categorization pipeline** using a **knowledge distillation** approach, where a high-capacity **BERT teacher model** guides a lightweight **custom CNN student model**. Our goal is to retain most of BERT’s accuracy while significantly reducing model size and inference time—making the solution deployment-friendly.

---

## 🚀 Overview

- **Task**: Multi-class document classification
- **Teacher Model**: Pre-trained BERT (fine-tuned on the task)
- **Student Model**: Lightweight CNN architecture
- **Distillation Technique**: Custom distillation loss combining soft targets and hard labels
- **Motivation**: Deploy BERT-level accuracy in low-resource environments using a smaller student model

---

## 📁 Project Structure

```
document-distillation/
│
├── data/                      # Dataset files and preprocessing scripts
│   └── preprocess.py
│
├── models/
│   ├── teacher_bert.py        # BERT fine-tuning model
│   └── student_cnn.py         # Custom CNN model
│
├── distillation/
│   └── loss.py                # Custom distillation loss implementation
│
├── train/
│   ├── train_teacher.py       # Fine-tune the teacher BERT
│   ├── train_student.py       # Train student with distillation
│   └── utils.py               # Utilities for training, logging, evaluation
│
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 📊 Methodology

### 🔹 Teacher Model: BERT
- Uses `bert-base-uncased` from HuggingFace Transformers
- Fine-tuned on the classification dataset
- Provides both **hard labels** and **soft logits** as supervision

### 🔹 Student Model: Custom CNN
- A simple CNN with:
  - Embedding layer
  - Multiple convolutional filters
  - Global max pooling
  - Fully connected output layer
- Extremely parameter-efficient

### 🔹 Custom Distillation Loss

The loss function blends:

```
Total Loss = α * CrossEntropy(student_logits, true_labels) + 
             (1 - α) * KLDiv(student_logits, teacher_logits / T)
```

Where:
- `α` is the weight balancing between distillation and classification loss
- `T` is the temperature scaling factor
- `KLDiv` encourages the student to mimic the softened output distribution of the teacher

---

## 🔧 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/yourusername/document-distillation.git
cd document-distillation

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📈 Training

### 1. Train the BERT Teacher
```bash
python train/train_teacher.py
```

### 2. Train the CNN Student using Distillation
```bash
python train/train_student.py
```

Optional flags:
- `--alpha`: weight for cross-entropy loss
- `--temperature`: temperature for softening logits

---

## 🧪 Evaluation

Both training scripts provide accuracy, precision, recall, and F1-score after each epoch. You can also run standalone evaluation:

```bash
python train/eval.py --model student
```

---

## 🧠 Key Benefits

- 🔍 **90%+ compression** in model size
- ⚡ **Faster inference** suitable for edge devices
- 📚 Retains most of the teacher’s accuracy with fewer resources
- 🧪 Fully customizable distillation pipeline

---

## 📌 TODO

- [ ] Add support for different student architectures (LSTM, Transformer Lite)
- [ ] Integrate TensorBoard for training visualizations
- [ ] Export student model to ONNX / TensorRT for deployment

---

## 📚 References

- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [TextCNN (Yoon Kim, 2014)](https://arxiv.org/abs/1408.5882)

---

## 💡 Citation

If this project helped you, consider citing it or giving it a ⭐️ on GitHub!
