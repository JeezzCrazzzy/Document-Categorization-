from transformers import TFBertForSequenceClassification

def get_teacher_model(num_labels):
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    return model
