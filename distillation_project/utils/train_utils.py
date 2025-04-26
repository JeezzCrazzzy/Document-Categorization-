import tensorflow as tf
from config import *

def distillation_loss(y_true, student_logits, teacher_logits, temperature=TEMPERATURE, alpha=ALPHA):
    student_soft = tf.nn.log_softmax(student_logits / temperature, axis=1)
    teacher_soft = tf.nn.softmax(teacher_logits / temperature, axis=1)
    soft_loss = tf.keras.losses.KLDivergence()(teacher_soft, student_soft) * (temperature ** 2)
    hard_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, student_logits)
    return alpha * soft_loss + (1 - alpha) * hard_loss

def evaluate(model, dataset):
    total, correct = 0, 0
    for batch_x, batch_y in dataset:
        preds = tf.argmax(model(batch_x, training=False), axis=1)
        correct += tf.reduce_sum(tf.cast(preds == batch_y, tf.int32)).numpy()
        total += batch_y.shape[0]
    return correct / total
