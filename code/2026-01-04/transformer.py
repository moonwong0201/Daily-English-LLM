# day02_attention_core.py  ≤30 行空白模板
import torch, torch.nn as nn, math

# ===== 超参 =====
d_k = 64
d_v = 64
n_heads = 8           # 头数
d_model = 64

# ===== 1. 缩放点积注意力 =====
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            scores.masked_fill(mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


# ===== 2. 多头注意力 =====
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(d_model, d_k * n_heads)
        self.k = nn.Linear(d_model, d_k * n_heads)
        self.v = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(d_v * n_heads, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        residual, batch_size = x, x.size(0)
        q_s = self.q(x).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.k(x).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.v(x).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        context, attn_weights = ScaledDotProductAttention()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attn_weights


# ===== 3. FeedForward + Add&Norm =====
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_ff)
        self.l2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = self.l1(x)
        output = nn.functional.gelu(output)
        output = self.l2(output)
        output = self.norm(output + residual)
        return output


# ===== 4. Transformer Block =====
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.feed = FeedForward(d_model, d_model * 4)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        residual = x
        x = x + self.attn(self.norm1(x), mask)[0]
        x = x + self.feed(self.norm2(x))
        return x


# ===== 5. 推理 demo =====
if __name__ == "__main__":
    block = TransformerBlock(d_model=64, n_heads=8)
    x = torch.randn(1, 8, 64)          # batch=1, seq=8, dim=64
    out = block(x)
    assert out.shape == x.shape, "shape error"
    print("✅ TransformerBlock output shape:", out.shape)
