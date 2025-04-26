def evaluate_student_model(model_student, dataset, label_encoder=None):
    """
    Evaluates the student model on a given dataset, prints metrics, and shows accuracy plots.

    Args:
        model_student: Trained student model (Keras model).
        dataset: tf.data.Dataset with (text, label) batches.
        label_encoder: Optional LabelEncoder for class names.
    
    Returns:
        accuracy: Float, overall accuracy.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    all_preds, all_labels = [], []

    for batch_texts, batch_labels in dataset:
        logits = model_student(batch_texts, training=False)
        preds = tf.argmax(logits, axis=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(batch_labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = np.mean(all_preds == all_labels)
    print(f"\n✅ Evaluation Accuracy: {acc:.4f}\n")

    # Classification Report
    if label_encoder:
        class_names = [str(cls) for cls in label_encoder.classes_]
        print("📊 Classification Report:\n")
        print(classification_report(all_labels, all_preds, target_names=class_names))
    else:
        print("📊 Classification Report:\n")
        print(classification_report(all_labels, all_preds))
        class_names = [str(i) for i in np.unique(all_labels)]


    # 🔹 Plot: Per-class Accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(10, 4))
    plt.bar(class_names, per_class_accuracy, color='skyblue')
    plt.title("Per-Class Accuracy")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 🔹 Plot: Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return acc
