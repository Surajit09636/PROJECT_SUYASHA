import subprocess

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import json
from datetime import datetime
from transformers import GPT2Tokenizer
from datasets import load_dataset

MEMORY_FILE = "memory.json"

if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump([], f)

def save_to_memory(user_input, ai_response):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(MEMORY_FILE, "r+") as f:
        memory = json.load(f)
        memory.append({"timestamp": timestamp, "user_input": user_input, "ai_response": ai_response})
        f.seek(0)
        json.dump(memory, f, indent=4)

def retrive_memory(query=None, max_result = 5, max_results=None):
    with open (MEMORY_FILE, "r") as f:
        memory = json.load(f)
        if query:
            return [item for item in memory if query.lower() in item["user_input"].lower()][:max_result]
        return memory[-max_results:]

def model(input_shape, output_units, input_Shape=None):
    model = Sequential([
        Dense(128, input_shape=input_Shape, activation='relu'),
        Dropout(0.2),
        Dense(64, activition='relu'),
        Dropout(0.2),
        Dense(output_units, activation='softmax')
    ])
    model.compile(optimizer = Adam(learning_rate=0.001),
                  loss = 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    return model

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epslion = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epslion =1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropuot2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(input_dim = maxlen, output_dim = embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start = 0, limit = maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def create_transformer_model(maxlen, vocab_size, embed_size, num_heads, ff_dim, embed_dim=None):
    inputs = tf.keras.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(20, activation = "relu")(x)
    outputs = tf.keras.layers.Dense(vocab_size, activation = "softmax")(x)
    model = tf.keras.model(inputs=inputs, outputs=outputs)
    return model

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def load_custom_datasets():
    dataset = load_dataset("openwebtext", split="train[:1%]")
    def preprocess_data(example):
        return tokenizer(example["text"], truncation = True, padding = "max_length", max_length = 128)
    return dataset.map(preprocess_data, batched = True)

def log_to_notepad(content):
    log_file = "test_log.txt"
    with open(log_file, "a") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{timestamp}\n")
        file.write(content + "\n\n")
    subprocess.Popen(["notepad.exe", log_file])

def save_model(model, model_name):
    model.save(model_name)
    print(f"Model saved as {model_name}")

def load_model(model_name):
    if os.path.exists(model_name):
        return tf.keras.models.load_model(model_name)
    else:
        print(f"No model found at {model_name}. please train first.")
        return None

def generate_response_with_memory(prompt, transformer_model=None, prediction=None):
    relevant_memory = retrive_memory(query=prompt, max_results=3)
    context = "\n".join([f"User : {m['user_input']}, AI: {m['ai_response']}" for m in relevant_memory])
    full_prompt = f"Context: {context}\nUser: {prompt}\nAI:"
    input_ids = tokenizer.encode(full_prompt, return_tensors = "tf")
    predection = transformer_model.predict(input_ids)
    predected_token = tf.argmax(prediction[0], axis=-1).numpy()
    response = tokenizer.decode(predected_token)
    save_to_memory(prompt, response)
    return response

if __name__ == "__main__":
    transformer_model = load_model("personal_ai_model")
    if transformer_model:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            response = generate_response_with_memory(user_input)
            log_to_notepad(f"User: {user_input}\nAI: {response}")
            print(f"AI: {response}")
