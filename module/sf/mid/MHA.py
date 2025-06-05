import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.q_proj=nn.Linear(embed_dim,embed_dim)
        self.k_proj=nn.Linear(embed_dim,embed_dim)
        self.v_proj=nn.Linear(embed_dim,embed_dim)
        self.head_dim = embed_dim//num_heads
        self.scale = 1 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if self.head_dim*num_heads !=embed_dim:
            raise ValueError('embed_dim cannot be evenly divided by num_heads')
    def split_head(self,x):
        bs,seq,_ = x.size()
        return x.view(bs,seq,self.num_heads,self.head_dim).transpose(1,2)
    def combine_head(self,x):
        bs,_,seq,_ = x.size()
        return x.transpose(1,2).contiguous().view(bs,seq,self.head_dim*self.num_heads)
    def causal_mask(self,x):
        bs,seq,_=x.size()
        mask=torch.tril(torch.ones(bs,self.num_heads,seq,seq))
        return mask
    def forward(self,input_x,pad_mask):
        Q=self.q_proj(input_x) # bt,seq,dim
        K=self.k_proj(input_x)
        V=self.v_proj(input_x)

        q=self.split_head(Q) #bt,num_heads,seq,dim
        k=self.split_head(K)
        v=self.split_head(V)
        pad_mask = pad_mask.unsqueeze(1).expand(-1,self.num_heads,-1).float() # bs,seq -> bs,num_heads,seq

        att_scores = torch.matmul(q,k.transpose(2,3))*self.scale

        causal_mask = self.causal_mask(input_x)
        if pad_mask is not None:
            # pad_mask=pad_mask.unsqueeze(2).expand(-1,-1,causal_mask.size(2),-1)
            pad_mask_1 = pad_mask.unsqueeze(-1) #bs,num_heads,seq,1
            pad_mask_2 = pad_mask.unsqueeze(2) # bs,num_heads,1,seq
            pad_mask = torch.matmul(pad_mask_1,pad_mask_2)
            combine_mask = causal_mask.bool() & pad_mask.bool()
        att_scores = att_scores.masked_fill(combine_mask.bool()==False,float('-inf'))
        att_weights = F.softmax(att_scores,dim=-1)

        # 检查是否存在全掩码行,当float('-inf')改成-2**32,也要注意全行为掩码的情况，也要强制把全掩码的行置为0
        all_inf_mask = (att_scores == float('-inf')).all(dim=-1, keepdim=True)
        att_weights = att_weights.masked_fill(all_inf_mask, 0.0)  # 强制赋0避免NaN

        output = torch.matmul(att_weights,v)
        output = self.combine_head(output)
        return output

if __name__ == '__main__':
    # 参数设置
    embed_dim = 8
    num_heads = 2
    batch_size = 2
    seq_len = 3

    # 创建模型
    model = MultiHeadAttention(embed_dim, num_heads)

    # 随机输入张量
    input_x = torch.randn(batch_size, seq_len, embed_dim)

    # 填充掩码（假设第二个样本的最后一个位置是填充）
    pad_mask = torch.tensor([
        [True, False, False],  # 第一个样本填充2
        [True, True, False]  # 第二个样本第三个位置是填充
    ],dtype=torch.bool)

    # 扩展维度并取反
    expanded_mask = pad_mask.unsqueeze(-1)
    mask = ~expanded_mask

    # 应用 masked_fill
    input_x = input_x.masked_fill(mask, 0)

    # 前向传播
    print('input_x:',input_x)
    print('pad_mask',pad_mask)
    output = model(input_x, pad_mask)
    print(output)



