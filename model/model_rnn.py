from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def ctc_loss_func(y_true, y_pred):
    """CTC loss function with proper data types"""
    # Convert y_true to int32 if it's not already
    y_true = tf.cast(y_true, tf.int32)
    
    batch_len = tf.shape(y_true)[0]
    
    # Calculate actual lengths (non-zero elements)
    label_length = tf.reduce_sum(tf.cast(y_true > 0, tf.int32), axis=1)
    input_length = tf.fill([batch_len], tf.shape(y_pred)[1])
    
    # Use sparse tensor format for CTC
    indices = tf.where(y_true > 0)
    values = tf.gather_nd(y_true, indices)
    dense_shape = tf.cast(tf.shape(y_true), tf.int64)
    
    sparse_labels = tf.SparseTensor(
        indices=tf.cast(indices, tf.int64),
        values=tf.cast(values, tf.int32),
        dense_shape=dense_shape
    )
    
    loss = tf.nn.ctc_loss(
        labels=sparse_labels,
        logits=y_pred,
        label_length=None,  # Will be inferred from sparse tensor
        logit_length=tf.cast(input_length, tf.int32),
        blank_index=33,
        logits_time_major=False
    )
    
    return tf.reduce_mean(loss)

def build_model(input_dim, output_dim):
    # Input layer
    input_data = Input(name='input', shape=(None, input_dim))
    
    # RNN layers
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(input_data)
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
    
    # Dense layers
    x = TimeDistributed(Dense(256, activation='relu'))(x)
    
    # Output layer (no softmax - CTC handles this)
    y_pred = TimeDistributed(Dense(output_dim, activation='softmax', name='dense2'))(x)
    
    # Create model
    model = Model(inputs=input_data, outputs=y_pred)
    
    # Compile with CTC loss
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=ctc_loss_func)
    
    return model