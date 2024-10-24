import streamlit as st
import torch
from torch import nn
import re
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to preprocess text
def preprocess_text(text):
    text = re.sub('\n', '.', text)
    text = re.sub('[^a-zA-Z0-9 /.]', '', text)
    text = text.lower()
    sentences = text.split('.')
    word_sequences = [sentence.strip().split() for sentence in sentences if sentence.strip()]
    return word_sequences

from collections import Counter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open ("temp.txt",'r') as data:
    text = data.read()

def preprocess_text(text):
    text = re.sub('\n', '.', text)
    text = re.sub('[^a-zA-Z0-9 /.]', '', text)

    text = text.lower()
    sentences = text.split('.')
    word_sequences = [sentence.strip().split() for sentence in sentences if sentence.strip()]
    return word_sequences

p_text = preprocess_text(str(text))
# Assume p_text is preprocessed text data (list of word sequences)
all_words = [word for sentence in p_text for word in sentence]
word_counts = Counter(all_words)
vocab = sorted(word_counts)
vocab_size = len(vocab)

# Word-to-index and index-to-word mappings
word_to_index = {word: idx for idx, word in enumerate(vocab)}
word_to_index["."] = 0
index_to_word = {idx: word for word, idx in word_to_index.items()}

# def create_word_pairs(sequences, context_length, word_to_index):
#     inputs, outputs = [], []

#     for sentence in sequences:
#         context = [word_to_index["."]] * context_length
#         for word in sentence + ["."]:
#             inputs.append([word_to_index.get(w, 0) for w in context])
#             outputs.append(word_to_index.get(word, 0))
#             context.pop(0)
#             context.append(word)
#     return inputs, outputs

# # Convert word sequences to word-index sequences
# inputs, outputs = create_word_pairs(p_text, context_length=4, word_to_index=word_to_index)

# Convert to tensors
# X = torch.tensor(inputs).to(device)
# Y = torch.tensor(outputs).to(device)

# word_to_index = {"example": 1, "word": 2, ".": 0}  # replace with your vocab
# index_to_word = {idx: word for word, idx in word_to_index.items()}

# Streamlit app title
st.title("Next Word Prediction")

# Text area for input text
input_text = st.text_area("Enter Text", height=200, placeholder="Type or paste your text here...")

# Dropdown for hyperparameters
context_length = st.selectbox("Context Length", options=[4, 8], index=0)
embedding_dim = st.selectbox("Embedding Dimension", options=[256, 1024], index=0)
activation_function = st.selectbox("Activation Function", options=["ReLU", "Softmax"], index=0)
random_seed = st.selectbox("Random Seed", options=[42, 2024], index=0)
vocab_size = 20068 
# Load pre-trained model based on hyperparameters
def load_model(context_length, embedding_dim, activation_function, random_seed):
    model = NextWord(context_length,vocab_size,embedding_dim,10,activation_function).to(device)
    # model = NextWord(context_length, len(word_to_index), embedding_dim, 10,activation_function).to(device)
    model_path = f"models/model_{context_length}_{embedding_dim}_{activation_function}_{random_seed}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# Define the model class
# class NextWord(nn.Module):
#     def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
#         super().__init__()
#         self.emb = nn.Embedding(vocab_size, emb_dim)
#         self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
#         self.lin2 = nn.Linear(hidden_size, vocab_size)

#     def forward(self, x):
#         x = self.emb(x)
#         x = x.view(x.shape[0], -1)
#         x = torch.sin(self.lin1(x))  # Using sin activation for demo purposes
#         x = self.lin2(x)
#         return x

class NextWord(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation_func="ReLU"):
        super(NextWord, self).__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, vocab_size)
        
        if activation_func == "ReLU":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        x = self.activation(self.lin1(x))
        x = self.lin2(x)
        return x

# Generate next word
def generate_word(text, model, index_to_word, word_to_index, block_size, max_len=15):
    context = [word_to_index.get(w.lower(), 0) for w in text.split()]
    if len(context)<block_size:
        context = [0]*(block_size-len(context))+context
    else:
        context = context[:block_size]
    len(context)
    sentence = []
    print(context)
    for _ in range(max_len):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        # print(ix)
        # word = index_to_word.get(ix, ".")
        word = index_to_word[ix]
        context.pop(0)
        context.append(ix)
        # print(word)
        
        if word == '.':
            break
        sentence.append(word)
        context = context[1:] + [ix]
    
    return sentence

if st.button("Predict Next Word"):
    # Load the model based on selected hyperparameters

    model = load_model(context_length, embedding_dim, activation_function, random_seed)

    # Generate next word
    predicted_sentence = generate_word(input_text, model, index_to_word, word_to_index, context_length)
    
    # Display predicted sentence
    print(predicted_sentence)
    st.write("### Predicted Next Words")
    st.write(f" ".join(predicted_sentence))
