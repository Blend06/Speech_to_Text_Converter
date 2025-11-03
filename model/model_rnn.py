from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional

def build_model(input_dim, output_dim):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True, input_shape=(None, input_dim))),
        TimeDistributed(Dense(128, activation='relu')),
        TimeDistributed(Dense(output_dim, activation='softmax'))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model