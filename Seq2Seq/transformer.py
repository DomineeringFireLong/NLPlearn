import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
# 自定义分词器
def get_tokenizer():
    """简单的分词器，按空格分割句子"""
    return lambda x: x.split()

# 构建词汇表
def build_vocab(data, tokenizer, min_freq=1, specials=["<unk>", "<pad>", "<sos>", "<eos>"]):
    counter = Counter()
    for src, tgt in data:
        counter.update(tokenizer(src))
        counter.update(tokenizer(tgt))
    
    # 按词频排序
    sorted_by_freq = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    # 构建词汇表
    vocab = {word: idx for idx, word in enumerate(specials)}
    for word, freq in sorted_by_freq:
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    # 添加特殊符号
    vocab["<unk>"] = 0
    vocab["<pad>"] = 1
    vocab["<sos>"] = 2
    vocab["<eos>"] = 3
    
    return vocab

# 将句子转换为索引
def text_to_indices(text, vocab, tokenizer):
    return [vocab.get(token, vocab["<unk>"]) for token in tokenizer(text)]

# 自定义数据集
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, tokenizer):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_indices = [self.src_vocab["<sos>"]] + text_to_indices(src, self.src_vocab, self.tokenizer) + [self.src_vocab["<eos>"]]
        tgt_indices = [self.tgt_vocab["<sos>"]] + text_to_indices(tgt, self.tgt_vocab, self.tokenizer) + [self.tgt_vocab["<eos>"]]
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)

# 数据加载器
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=1, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=1, batch_first=True)
    return src_batch, tgt_batch

# 多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        original_size_attention = scaled_attention.view(batch_size, -1, self.d_model)
        
        output = self.dense(original_size_attention)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(k.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        
        return output, attention_weights

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
    
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 编码层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
    
    def forward(self, x, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

# 解码层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)
        self.dropout3 = nn.Dropout(rate)
    
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)
        
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3, attn_weights_block1, attn_weights_block2

# 编码器
class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding)
        
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(rate)
    
    def forward(self, x, mask):
        seq_len = x.size(1)
        
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)
        
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)
        
        return x

# 解码器
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding)
        
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(rate)
    
    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.size(1)
        attention_weights = {}
        
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoding(x)
        
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        
        return x, attention_weights

# Transformer
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        
        self.final_layer = nn.Linear(d_model, target_vocab_size)
    
    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)
        
        dec_output, attention_weights = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)
        
        return final_output, attention_weights

# 翻译函数
def translate(model, src_sentence, src_vocab, tgt_vocab, tokenizer, max_length=50):
    model.eval()
    
    # 将源句子转换为索引
    src_indices = torch.tensor([src_vocab["<sos>"]] + text_to_indices(src_sentence, src_vocab, tokenizer) + [src_vocab["<eos>"]], dtype=torch.long).unsqueeze(0)
    
    # 初始化目标序列
    tgt_indices = torch.tensor([[tgt_vocab["<sos>"]]], dtype=torch.long)
    
    for _ in range(max_length):
        # 预测下一个词
        with torch.no_grad():
            output, _ = model(src_indices, tgt_indices, None, None, None)
            next_token = output.argmax(dim=-1)[:, -1].item()
        
        # 添加到目标序列
        tgt_indices = torch.cat([tgt_indices, torch.tensor([[next_token]], dtype=torch.long)], dim=-1)
        
        # 如果遇到 <eos>，停止生成
        if next_token == tgt_vocab["<eos>"]:
            break
    
    # 将索引转换为句子
    tgt_sentence = " ".join([list(tgt_vocab.keys())[list(tgt_vocab.values()).index(idx)] for idx in tgt_indices.squeeze().tolist()])
    return tgt_sentence

# 训练函数
def train(model, dataloader, optimizer, criterion, num_epochs):
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for src, tgt in dataloader:
            # 前向传播
            optimizer.zero_grad()
            output, _ = model(src, tgt[:, :-1], None, None, None)
            
            # 计算损失
            loss = criterion(output.view(-1, model.decoder.embedding.num_embeddings), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return train_losses

def plot_loss(train_losses):
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.show()


# 主程序
if __name__ == "__main__":
# 示例数据
    data = [
        ("The quick brown fox jumps over the lazy dog", "Le rapide renard brun saute par-dessus le chien paresseux"),
        ("I love machine learning", "J'adore l'apprentissage automatique"),
        ("Hello world", "Bonjour le monde"),
        ("How are you", "Comment ça va"),
        ("Good morning", "Bonjour"),
        ("Good night", "Bonne nuit")
    ]

    # 分词器
    tokenizer = get_tokenizer()

    # 构建词汇表
    src_vocab = build_vocab(data, tokenizer)
    tgt_vocab = build_vocab(data, tokenizer)

    # 数据集和数据加载器
    dataset = TranslationDataset(data, src_vocab, tgt_vocab, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # 初始化模型
    transformer = Transformer(
        num_layers=2, d_model=128, num_heads=8, dff=512,
        input_vocab_size=len(src_vocab), target_vocab_size=len(tgt_vocab),
        pe_input=1000, pe_target=1000
    )

    # 损失函数（忽略填充部分的损失）
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])

    # 优化器
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001)

    # 训练
    num_epochs = 300
    train_losses = train(transformer, dataloader, optimizer, criterion, num_epochs)

    # 可视化训练损失
    plot_loss(train_losses)

    # 多次翻译测试
    test_sentences = [
        "The quick brown fox jumps over the lazy dog",
        "I love machine learning",
        "Hello world",
        "How are you",
        "Good morning",
        "Good night"
    ]

    for src_sentence in test_sentences:
        translated_sentence = translate(transformer, src_sentence, src_vocab, tgt_vocab, tokenizer)
        print(f"Source: {src_sentence}")
        print(f"Translated: {translated_sentence}")
        print("-" * 50)