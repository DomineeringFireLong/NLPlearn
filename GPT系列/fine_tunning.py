import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pre_train import SimpleGPT  # 导入预训练模型

# 电影评论情感分析数据集
class MovieReviewDataset(Dataset):
    def __init__(self, reviews, labels, vocab_size, context_length):
        self.reviews = reviews
        self.labels = labels
        self.vocab_size = vocab_size
        self.context_length = context_length
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = self.reviews[idx]
        tokens = [ord(c) % self.vocab_size for c in review]
        
        if len(tokens) > self.context_length:
            tokens = tokens[:self.context_length]
        else:
            tokens = tokens + [0] * (self.context_length - len(tokens))
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 修正后的情感分析模型
class SentimentGPT(nn.Module):
    def __init__(self, pretrained_model, num_classes=2):
        super(SentimentGPT, self).__init__()
        self.gpt = pretrained_model
        
        # 冻结GPT参数 (可选)
        for param in self.gpt.parameters():
            param.requires_grad = False
            
        # 获取嵌入维度 - 从token嵌入层获取
        embed_size = self.gpt.token_embedding.embedding_dim
        
        # 添加分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_size, 128),  # 输入维度与GPT嵌入维度一致
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # 获取GPT的隐藏表示
        # x形状: (batch_size, seq_len)
        gpt_output = self.gpt(x)  # (batch_size, seq_len, vocab_size)
        
        # 关键修改：从token嵌入层获取中间表示，而不是直接使用输出logits
        # 获取最后一层Transformer的输出
        with torch.no_grad():
            # 获取词嵌入和位置嵌入
            B, T = x.shape
            pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
            tok_emb = self.gpt.token_embedding(x)  # (B,T,embed_size)
            pos_emb = self.gpt.position_embedding(pos)  # (1,T,embed_size)
            x_emb = tok_emb + pos_emb
            
            # 通过所有Transformer层获取中间表示
            for block in self.gpt.transformer_blocks:
                x_emb = block(x_emb)
            
            # 获取最后的隐藏状态（未经过最后的线性层）
            hidden_states = self.gpt.ln_f(x_emb)  # (B,T,embed_size)
        
        # 取序列的第一个token的表示作为整个序列的表示
        pooled_output = hidden_states[:, 0, :]  # (batch_size, embed_size)
        
        # 分类预测
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)
        return logits

def load_pretrained_model(vocab_size, embed_size, num_heads, num_layers, context_length):
    model = SimpleGPT(vocab_size, embed_size, num_heads, num_layers, context_length)
    model.load_state_dict(torch.load('pretrained_gpt.pth'))
    return model

def fine_tune():
    # 超参数设置
    vocab_size = 5000
    embed_size = 128
    num_heads = 4
    num_layers = 4
    context_length = 64
    
    batch_size = 8  # 修改为8以匹配错误信息中的batch_size
    lr = 1e-4
    num_epochs = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    reviews = [
        "This movie was fantastic! I loved every minute of it.",
        "Terrible acting and boring plot. Would not recommend.",
        "The cinematography was beautiful but the story was weak.",
        "One of the best films I've seen this year!",
        "I fell asleep halfway through. So dull.",
        "The characters were so well developed and the story was engaging.",
        "Waste of time and money. The worst movie of the year.",
        "A masterpiece that will stand the test of time."
    ]
    labels = [1, 0, 0, 1, 0, 1, 0, 1]
    
    dataset = MovieReviewDataset(reviews, labels, vocab_size, context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    pretrained_model = load_pretrained_model(vocab_size, embed_size, num_heads, num_layers, context_length)
    model = SentimentGPT(pretrained_model).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.2f}"
            })
        
        epoch_loss /= len(dataloader)
        epoch_acc = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    
    plt.tight_layout()
    plt.show()
    
    test_reviews = [
        "I absolutely loved this film!",
        "This was the worst movie ever made.",
        "The acting was mediocre but the story was compelling.",
        "A truly unforgettable cinematic experience."
    ]
    
    model.eval()
    with torch.no_grad():
        for review in test_reviews:
            tokens = [ord(c) % vocab_size for c in review]
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
            else:
                tokens = tokens + [0] * (context_length - len(tokens))
            
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            
            sentiment = "Positive" if predicted.item() == 1 else "Negative"
            print(f"Review: '{review[:50]}...' | Sentiment: {sentiment}")

if __name__ == '__main__':
    fine_tune()