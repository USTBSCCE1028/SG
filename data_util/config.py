import os

root_dir = os.path.expanduser("~")
# path
data_path = "./data/atis/"
vocab_path = data_path + "vocab.txt"
model_save_sent_acc = "./ckpt/best_sent_acc/"
model_save_intent_acc = "./ckpt/best_intent_acc/"
model_save_slot_f1 = "./ckpt/best_slot_f1/"
model_path = "atis_model.bin"

# model hyperparameters
hidden_dim = 768
emb_dim = 300
emb_dorpout = 0.8
lstm_dropout = 0.5
attention_dropout = 0.1
num_attention_heads = 8

# hyperparameters
# max_len = 32
max_len = 32
lr_scheduler_gama = 0.5
batch_size = 32
epoch = 50
seed = 12
lr = 1e-5
eps = 1e-12
use_gpu = True

