import torch
import torch.nn as nn
import math
import librosa

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)  # query
        self.W_k = nn.Linear(d_model, d_model)  # key
        self.W_v = nn.Linear(d_model, d_model)  # value
        self.W_o = nn.Linear(d_model, d_model)  # output

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        self.pe = self.pe.to(x.device)
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class PhantomNet(nn.Module):
    def __init__(self, use_mode, feature_size, conv_projection, num_classes, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1):
        super(PhantomNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=10, stride=5)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2)
        self.conv4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=2)
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.conv7 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.use_mode = use_mode
        self.conv_projection = conv_projection
        self.num_classes = num_classes
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

        self.fcIntermidiate = nn.Linear(512, feature_size)
        self.positional_encoding = PositionalEncoding(feature_size, 10000)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(feature_size, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
        if self.conv_projection:
            self.convProjection = nn.Conv1d(feature_size, feature_size, kernel_size=128, stride=1)
        
        self.fc1 = nn.Linear(feature_size, feature_size)
        self.fc2 = nn.Linear(feature_size, 1, bias=True)

        
        if self.use_mode == 'spoof':
            #if there is a mismatch error, you will need to replace this input size.. currently working with 8 seconds samples
            #just multiply 95.760 * seconds the get this layer's input size
            #or I can just add another parameter to the model seq_length and input = seq_length * feature_size 
            self.fcSpoof = nn.Linear(286080, d_ff)
            self.fcFinal = nn.Linear(d_ff,self.num_classes)
            
        else:
            self.fcSpoof = None

    def forward(self, src):
        src = src.unsqueeze(1)
        src = self.gelu(self.conv1(src))
        src = self.gelu(self.conv2(src))
        src = self.gelu(self.conv3(src))
        src = self.gelu(self.conv4(src))
        src = self.gelu(self.conv5(src))
        src = self.gelu(self.conv6(src))
        src = self.gelu(self.conv7(src))
        src = src.permute(0, 2, 1)
        src = self.fcIntermidiate(src)
        src = src.permute(0, 2, 1)
        
        if self.conv_projection:
            src = self.gelu(self.convProjection(src))
        
        src = self.dropout(src)
        src = src.transpose(1, 2)
        src_embedded = self.dropout(self.positional_encoding(src))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, None)

        embeddings = self.fc1(enc_output)
        flatten_embeddings = self.flatten(embeddings)

        if self.use_mode == 'extractor':
            return embeddings
        elif self.use_mode == 'partialSpoof':
            return self.fc2(embeddings)
        elif self.use_mode == 'spoof':
            out_fcSpoof= self.fcSpoof(flatten_embeddings)
            output = self.fcFinal(out_fcSpoof)
           # output = self.sigmoid(self.fcSpoof(flatten_embeddings))
#            print(f"Model output shape: {output.shape}")
            return output
        else:
            raise ValueError('Wrong use mode of PhantomNet, please pick between extractor, partialSpoof, or spoof')




