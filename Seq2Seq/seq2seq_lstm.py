import torch
import torch.nn as nn
import torch.optim as optim
import random

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(input_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
    
    def forward(self, input_seq):
        # 输入序列嵌入
        embedded = self.embedding(input_seq)
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_size)
        
        # LSTM前向传播
        outputs, (hidden, cell) = self.lstm(embedded, (h0, c0))
        
        # 返回最后的隐藏状态和细胞状态
        return hidden, cell


# 解码器
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, embedding_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(output_size, embedding_dim)
        
        # LSTM层
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        
        # 全连接层，用于输出词汇表的概率分布
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_seq, hidden, cell):
        # 输入序列嵌入
        embedded = self.embedding(input_seq)
        
        # LSTM前向传播
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # 通过全连接层生成输出
        predictions = self.fc(outputs)
        
        # 返回预测结果和新的隐藏状态、细胞状态
        return predictions, hidden, cell
    

# Seq2Seq模型
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # 编码器编码输入序列
        hidden, cell = self.encoder(src)
        
        # 解码器的输入是目标序列的第一个单词（通常是<SOS>）
        input = trg[:, 0].unsqueeze(1)
        
        # 存储解码器的输出
        outputs = []
        
        # 逐步解码
        for t in range(1, trg.size(1)):
            # 解码器生成输出
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # 存储输出
            outputs.append(output)
            
            # 决定是否使用教师强制
            teacher_force = random.random() < teacher_forcing_ratio
            
            # 下一个输入是目标序列的下一个单词或解码器生成的单词
            input = trg[:, t].unsqueeze(1) if teacher_force else output.argmax(2)
        
        # 将输出序列拼接起来
        outputs = torch.cat(outputs, dim=1)
        
        return outputs


# 数据准备
def prepare_data(src_sentence, tgt_sentence, src_vocab, tgt_vocab):
    src_seq = [src_vocab[word] for word in src_sentence.split()] + [src_vocab['<EOS>']]
    tgt_seq = [tgt_vocab['<SOS>']] + [tgt_vocab[word] for word in tgt_sentence.split()] + [tgt_vocab['<EOS>']]
    return torch.tensor(src_seq, dtype=torch.long).unsqueeze(0), torch.tensor(tgt_seq, dtype=torch.long).unsqueeze(0)

# 训练函数
def train(model, src, trg, optimizer, criterion, teacher_forcing_ratio=0.5):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    output = model(src, trg, teacher_forcing_ratio)
    
    # 计算损失
    loss = criterion(output.view(-1, output.size(-1)), trg[:, 1:].contiguous().view(-1))
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 测试函数
def translate(model, src, src_vocab, tgt_vocab, max_len=10):
    model.eval()
    with torch.no_grad():
        # 编码器编码输入序列
        hidden, cell = model.encoder(src)
        
        # 解码器的输入是<SOS>
        input = torch.tensor([[tgt_vocab['<SOS>']]], dtype=torch.long)
        
        # 存储翻译结果
        translated_sentence = []
        
        # 逐步解码
        for _ in range(max_len):
            # 解码器生成输出
            output, hidden, cell = model.decoder(input, hidden, cell)
            
            # 选择概率最高的单词
            top_word = output.argmax(2)
            
            # 如果生成<EOS>则停止
            if top_word.item() == tgt_vocab['<EOS>']:
                break
            
            # 存储生成的单词
            translated_sentence.append(list(tgt_vocab.keys())[list(tgt_vocab.values()).index(top_word.item())])
            
            # 下一个输入是生成的单词
            input = top_word
        
        return ' '.join(translated_sentence)

if __name__ == "__main__":
    # 英文词汇表
    src_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'I': 3, 'love': 4, 'China': 5}
    
    # 中文词汇表
    tgt_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '我': 3, '爱': 4, '中国': 5}
    
    # 模型参数
    input_size = len(src_vocab)
    output_size = len(tgt_vocab)
    hidden_size = 128
    embedding_dim = 64
    num_layers = 1
    
    # 初始化模型
    encoder = Encoder(input_size, hidden_size, embedding_dim, num_layers)
    decoder = Decoder(output_size, hidden_size, embedding_dim, num_layers)
    model = Seq2Seq(encoder, decoder)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 准备数据
    src_sentence1 = "I love China"
    tgt_sentence1 = "我 爱 中国"
    src1, trg1 = prepare_data(src_sentence1, tgt_sentence1, src_vocab, tgt_vocab)
    
    src_sentence2 = "I love"
    tgt_sentence2 = "我 爱"
    src2, trg2 = prepare_data(src_sentence2, tgt_sentence2, src_vocab, tgt_vocab)
    
    # 训练模型
    num_epochs = 200
    for epoch in range(num_epochs):
        loss1 = train(model, src1, trg1, optimizer, criterion, teacher_forcing_ratio=0.5)
        loss2 = train(model, src2, trg2, optimizer, criterion, teacher_forcing_ratio=0.5)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss1: {loss1:.4f}, Loss2: {loss2:.4f}')
    
    # 测试模型
    translated_sentence1 = translate(model, src1, src_vocab, tgt_vocab)
    print(f'Translated sentence1: {translated_sentence1}')

    translated_sentence2 = translate(model, src2, src_vocab, tgt_vocab)
    print(f'Translated sentence2: {translated_sentence2}')