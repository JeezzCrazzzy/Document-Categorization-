import tensorflow as tf
from transformers import TFBertForSequenceClassification
from config import *
from models.teacher import get_teacher_model
from models.student import get_student_model
from utils.preprocess import load_and_prepare_data, get_tokenizer, get_vectorizer
from utils.train_utils import distillation_loss, evaluate

x_train, x_val, y_train, y_val, label_encoder = load_and_prepare_data()
tokenizer = get_tokenizer()
vectorizer = get_vectorizer(x_train)

train_tokens = tokenizer(list(x_train), truncation=True, padding=True, max_length=MAX_LENGTH)
val_tokens = tokenizer(list(x_val), truncation=True, padding=True, max_length=MAX_LENGTH)
train_dataset_teacher = tf.data.Dataset.from_tensor_slices((dict(train_tokens), y_train)).shuffle(1000).batch(BATCH_SIZE)
val_dataset_teacher = tf.data.Dataset.from_tensor_slices((dict(val_tokens), y_val)).batch(BATCH_SIZE)

teacher_model = get_teacher_model(num_labels=NUM_CLASSES)
teacher_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5))
teacher_model.fit(train_dataset_teacher, epochs=1)

teacher_model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

from models.student import get_student_model
student_model = get_student_model(vectorizer, len(vectorizer.get_vocabulary()), NUM_CLASSES)

lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=DECAY_STEPS,
    end_learning_rate=END_LR
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

train_dataset_student = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
val_dataset_student = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE)

@tf.function
def train_step(texts, labels, teacher_logits):
    with tf.GradientTape() as tape:
        student_logits = student_model(texts, training=True)
        loss = distillation_loss(labels, student_logits, teacher_logits)
    grads = tape.gradient(loss, student_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
    return loss

for epoch in range(EPOCHS):
    total_loss = 0
    for step, (texts, labels) in enumerate(train_dataset_student):
        decoded = [t.decode("utf-8") for t in texts.numpy()]
        teacher_inputs = tokenizer(decoded, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="tf")
        teacher_logits = teacher_model(teacher_inputs).logits
        loss = train_step(texts, labels, teacher_logits)
        total_loss += loss.numpy()
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.numpy():.4f}")
    val_acc = evaluate(student_model, val_dataset_student)
    print(f"Epoch {epoch + 1} - Avg Loss: {total_loss / len(train_dataset_student):.4f}, Val Acc: {val_acc:.4f}")
