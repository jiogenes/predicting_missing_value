import numpy as np
import pyreadstat
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

class TabularDataset(Dataset):
    def __init__(self, label_name, train=True):
        self.df = self.preprocess()
        self.cats, self.conts, self.label = self.get_data(label_name, train)

    def preprocess(self):
        exclude_columns = [
            'QKEY', 'INTERVIEW_START_W116', 'INTERVIEW_END_W116',
            'WEIGHT_W116', 'XW91NONRESP_W116', 'LANG_W116',
            'FORM_W116', 'DEVICE_TYPE_W116', 'XW78NONRESP_W116'
        ]

        df, meta = pyreadstat.read_sav("/data/jyji/ATP W116.sav")
        df = df.drop(exclude_columns, axis=1)

        def preprocess_column(column):
            # 고유값을 가져와서 정렬
            unique_values = sorted(column.dropna().unique())

            # 연속적인 숫자 찾기
            sequential_values = set()
            previous_value = None

            for value in unique_values:
                if previous_value is None or value == previous_value + 1:
                    sequential_values.add(value)
                previous_value = value

            # 연속적인 숫자는 그대로, 불연속적인 숫자와 NaN은 0으로 대체
            column = column.apply(lambda x: int(x) if x in sequential_values else 0)
            return column

        # 데이터프레임의 모든 컬럼에 대해 전처리 적용
        for col in df.columns:
            if col not in ['THERMTRUMP_W116', 'THERMBIDEN_W116']:
                df[col] = preprocess_column(df[col])

        return df


    def get_data(self, label_name, train=True):


        # 고유 값 추출
        unique_values = [value for value in self.df[label_name].unique()]

        # 샘플 인덱스 추출
        sample_indices = []
        total_samples_needed = 100
        samples_per_value = total_samples_needed // len(unique_values)

        for value in unique_values:
            available_samples = (self.df[label_name] == value).sum()
            n_samples = min(samples_per_value, available_samples)
            if n_samples > 0:
                sample_indices.extend(self.df[self.df[label_name] == value].sample(n=n_samples, random_state=1).index.tolist())
                total_samples_needed -= n_samples

        if total_samples_needed > 0:
            remaining_samples = self.df[~self.df.index.isin(sample_indices)][self.df[label_name].isin(unique_values)].sample(n=total_samples_needed, random_state=1).index.tolist()
            sample_indices.extend(remaining_samples)

        if len(sample_indices) < 100:
            raise ValueError(f"Not enough data to sample 100 rows with balanced ratio in '{label_name}'.")

        # 샘플된 데이터프레임 추출
        df_sampled = self.df.loc[sample_indices].copy()

        # Test dataset 설정 (샘플된 데이터)
        test_cats = df_sampled.drop(columns=['THERMTRUMP_W116', 'THERMBIDEN_W116', label_name]).fillna(0.0).replace([9.0, 99.0], 0.0).astype(int).values
        test_conts = df_sampled[['THERMTRUMP_W116', 'THERMBIDEN_W116']].fillna(0.0).replace([9.0, 99.0], 0.0).astype(float).values
        test_label = df_sampled[label_name].fillna(0.0).replace([9.0, 99.0], 0.0).astype(int).values

        # Train dataset 설정 (샘플되지 않은 데이터)
        df_remaining = self.df.drop(index=sample_indices).copy()
        train_cats = df_remaining.drop(columns=['THERMTRUMP_W116', 'THERMBIDEN_W116', label_name]).fillna(0.0).replace([9.0, 99.0], 0.0).astype(int).values
        train_conts = df_remaining[['THERMTRUMP_W116', 'THERMBIDEN_W116']].fillna(0.0).replace([9.0, 99.0], 0.0).astype(float).values
        train_label = df_remaining[label_name].fillna(0.0).replace([9.0, 99.0], 0.0).astype(int).values

        unique = [set(np.unique(train_cats[:, i]).tolist() + np.unique(test_cats[:, i]).tolist()) for i in range(train_cats.shape[1])]
        self.unique = {i: list(val) for i, val in enumerate(unique)}

        return (train_cats, train_conts, train_label) if train else (test_cats, test_conts, test_label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.cats[idx], self.conts[idx], self.label[idx]

class TabularDataset2(Dataset):
    def __init__(self, label_name, train=True, user_ratio=0.2):
        self.user_ratio = user_ratio
        self.df = self.preprocess()
        self.cats, self.conts, self.label = self.get_data(label_name, train)

    def preprocess(self):
        exclude_columns = [
            'QKEY', 'INTERVIEW_START_W116', 'INTERVIEW_END_W116',
            'WEIGHT_W116', 'XW91NONRESP_W116', 'LANG_W116',
            'FORM_W116', 'DEVICE_TYPE_W116', 'XW78NONRESP_W116'
        ]

        df, meta = pyreadstat.read_sav("/data/jyji/ATP W116.sav")
        df = df.drop(exclude_columns, axis=1)

        def preprocess_column(column):
            # 고유값을 가져와서 정렬
            unique_values = sorted(column.dropna().unique())

            # 연속적인 숫자 찾기
            sequential_values = set()
            previous_value = None

            for value in unique_values:
                if previous_value is None or value == previous_value + 1:
                    sequential_values.add(value)
                previous_value = value

            # 연속적인 숫자는 그대로, 불연속적인 숫자와 NaN은 0으로 대체
            column = column.apply(lambda x: int(x) if x in sequential_values else 0)
            return column

        # 데이터프레임의 모든 컬럼에 대해 전처리 적용
        for col in df.columns:
            if col not in ['THERMTRUMP_W116', 'THERMBIDEN_W116']:
                df[col] = preprocess_column(df[col])

        return df


    def get_data(self, label_name, train=True):
        n_users = int(len(self.df) * self.user_ratio) # 0.5
        missing_df = self.df.sample(n=n_users, random_state=42)
        missing_user_ids = missing_df.index
        if label_name == 'SATIS_W116' or label_name == 'POL1JB_W116':
            index_1 = missing_df[missing_df[label_name] == 1.0].sample(n=50, random_state=42).index.tolist()
            index_2 = missing_df[missing_df[label_name] == 2.0].sample(n=50, random_state=42).index.tolist()
            sample_indices = index_1 + index_2
        else:
            index_1 = missing_df[missing_df[label_name] == 1.0].sample(n=33, random_state=42).index.tolist()
            index_2 = missing_df[missing_df[label_name] == 2.0].sample(n=33, random_state=42).index.tolist()
            index_3 = missing_df[missing_df[label_name] == 3.0].sample(n=34, random_state=42).index.tolist()
            sample_indices = index_1 + index_2 + index_3

        # 샘플된 데이터프레임 추출
        df_sampled = self.df.loc[sample_indices].copy()

        # Test dataset 설정 (샘플된 데이터)
        test_cats = df_sampled.drop(columns=['THERMTRUMP_W116', 'THERMBIDEN_W116', label_name]).fillna(0.0).replace([9.0, 99.0], 0.0).astype(int).values
        test_conts = df_sampled[['THERMTRUMP_W116', 'THERMBIDEN_W116']].fillna(0.0).replace([9.0, 99.0], 0.0).astype(float).values
        test_label = df_sampled[label_name].fillna(0.0).replace([9.0, 99.0], 0.0).astype(int).values

        # Train dataset 설정 (샘플되지 않은 데이터)``
        missing_user_ids = [i for i in missing_user_ids if i not in sample_indices]
        self.df = self.df.drop(missing_user_ids, axis=0)
        df_remaining = self.df.drop(index=sample_indices).copy()
        train_cats = df_remaining.drop(columns=['THERMTRUMP_W116', 'THERMBIDEN_W116', label_name]).fillna(0.0).replace([9.0, 99.0], 0.0).astype(int).values
        train_conts = df_remaining[['THERMTRUMP_W116', 'THERMBIDEN_W116']].fillna(0.0).replace([9.0, 99.0], 0.0).astype(float).values
        train_label = df_remaining[label_name].fillna(0.0).replace([9.0, 99.0], 0.0).astype(int).values

        unique = [set(np.unique(train_cats[:, i]).tolist() + np.unique(test_cats[:, i]).tolist()) for i in range(train_cats.shape[1])]
        self.unique = {i: list(val) for i, val in enumerate(unique)}


        return (train_cats, train_conts, train_label) if train else (test_cats, test_conts, test_label)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.cats[idx], self.conts[idx], self.label[idx]


def ifnone(a, b):
    # From fastai.fastcore
    "`b` if `a` is None else `a`"
    return b if a is None else a


def _trunc_normal_(x, mean=0., std=1.):
    "Truncated normal initialization (approximation)"
    # From fastai.layers
    # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class _Embedding(nn.Embedding):
    "Embedding layer with truncated normal initialization"
    # From fastai.layers
    def __init__(self, ni, nf, std=0.01):
        super(_Embedding, self).__init__(ni, nf)
        _trunc_normal_(self.weight.data, std=std)


class SharedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, shared_embed=True, add_shared_embed=False, shared_embed_div=8):
        super().__init__()
        if shared_embed:
            if add_shared_embed:
                shared_embed_dim = embedding_dim
                self.embed = _Embedding(num_embeddings, embedding_dim)
            else:
                shared_embed_dim = embedding_dim // shared_embed_div
                self.embed = _Embedding(num_embeddings, embedding_dim - shared_embed_dim)
            self.shared_embed = nn.Parameter(torch.empty(1, 1, shared_embed_dim))
            _trunc_normal_(self.shared_embed.data, std=0.01)
            self.add_shared_embed = add_shared_embed
        else:
            self.embed = _Embedding(num_embeddings, embedding_dim)
            self.shared_embed = None

    def forward(self, x):
        out = self.embed(x).unsqueeze(1)
        # print(out.shape)
        if self.shared_embed is None: return out
        if self.add_shared_embed:
            out += self.shared_embed
        else:
            shared_embed = self.shared_embed.expand(out.shape[0], -1, -1)
            # print(shared_embed.shape)
            out = torch.cat((out, shared_embed), dim=-1)
        return out


class FullEmbeddingDropout(nn.Module):
    '''From https://github.com/jrzaurin/pytorch-widedeep/blob/be96b57f115e4a10fde9bb82c35380a3ac523f52/pytorch_widedeep/models/tab_transformer.py#L153'''
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        mask = x.new().resize_((x.size(1), 1)).bernoulli_(1 - self.dropout).expand_as(x) / (1 - self.dropout)
        return mask * x


class _MLP(nn.Module):
    def __init__(self, dims, bn=False, act=None, skip=False, dropout=0., bn_final=False):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for i, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = i >= (len(dims) - 2)
            if bn and (not is_last or bn_final): layers.append(nn.BatchNorm1d(dim_in))
            if dropout and not is_last:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim_in, dim_out))
            if is_last: break
            layers.append(ifnone(act, nn.ReLU()))
        self.mlp = nn.Sequential(*layers)
        self.shortcut = nn.Linear(dims[0], dims[-1]) if skip else None

    def forward(self, x):
        if self.shortcut is not None:
            return self.mlp(x) + self.shortcut(x)
        else:
            return self.mlp(x)


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k:int, res_attention:bool=False):
        super().__init__()
        self.d_k,self.res_attention = d_k,res_attention

    def forward(self, q, k, v, prev=None, attn_mask=None):

        # MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        scores = torch.matmul(q, k)                                    # scores : [bs x n_heads x q_len x q_len]

        # Scale
        scores = scores / (self.d_k ** 0.5)

        # Attention mask (optional)
        if attn_mask is not None:                                     # mask with shape [q_len x q_len]
            if attn_mask.dtype == torch.bool:
                scores.masked_fill_(attn_mask, float('-inf'))
            else:
                scores += attn_mask

        # SoftMax
        if prev is not None: scores = scores + prev

        attn = F.softmax(scores, dim=-1)                               # attn   : [bs x n_heads x q_len x q_len]

        # MatMul (attn, v)
        context = torch.matmul(attn, v)                                # context: [bs x n_heads x q_len x d_v]

        if self.res_attention: return context, attn, scores
        else: return context, attn


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_k:int, d_v:int, res_attention:bool=False):
        """Input shape:  Q, K, V:[batch_size (bs) x q_len x d_model], mask:[q_len x q_len]"""
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)

        self.W_O = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.res_attention = res_attention

        # Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            self.sdp_attn = _ScaledDotProductAttention(self.d_k, self.res_attention)
        else:
            self.sdp_attn = _ScaledDotProductAttention(self.d_k)


    def forward(self, Q, K, V, prev=None, attn_mask=None):

        bs = Q.size(0)

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            context, attn, scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, attn_mask=attn_mask)
        else:
            context, attn = self.sdp_attn(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len]

        # Concat
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # context: [bs x q_len x n_heads * d_v]

        # Linear
        output = self.W_O(context)                                                           # context: [bs x q_len x d_model]

        if self.res_attention: return output, attn, scores
        else: return output, attn                                                            # output: [bs x q_len x d_model]


class _TabEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 res_dropout=0.1, activation="gelu", res_attention=False):

        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = ifnone(d_k, d_model // n_heads)
        d_v = ifnone(d_v, d_model // n_heads)
        d_ff = ifnone(d_ff, d_model * 4)

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(res_dropout)
        self.layernorm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), self._get_activation_fn(activation), nn.Linear(d_ff, d_model))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(res_dropout)
        self.layernorm_ffn = nn.LayerNorm(d_model)

    def forward(self, src, prev=None, attn_mask=None):

        # Multi-Head attention sublayer
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, attn_mask=attn_mask)
        self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        src = self.layernorm_attn(src) # Norm: layernorm

        # Feed-forward sublayer
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        src = self.layernorm_ffn(src) # Norm: layernorm

        if self.res_attention:
            return src, scores
        else:
            return src

    def _get_activation_fn(self, activation):
        if callable(activation): return activation()
        elif activation.lower() == "relu": return nn.ReLU()
        elif activation.lower() == "gelu": return nn.GELU()
        raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


class _TabEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None, res_dropout=0.1, activation='gelu', res_attention=False, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([_TabEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout,
                                                            activation=activation, res_attention=res_attention) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src, attn_mask=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, attn_mask=attn_mask)
            return output


class TabTransformer(nn.Module):
    def __init__(self, classes, cont_names, c_out, column_embed=True, add_shared_embed=False, shared_embed_div=8, embed_dropout=0.1, drop_whole_embed=False,
                 d_model=32, n_layers=6, n_heads=8, d_k=None, d_v=None, d_ff=None, res_attention=True, attention_act='gelu', res_dropout=0.1, norm_cont=True,
                 mlp_mults=(4, 2), mlp_dropout=0., mlp_act=None, mlp_skip=False, mlp_bn=False, bn_final=False):

        super().__init__()
        self.n_cat = len(classes)
        self.n_classes = [len(v) for v in classes.values()]
        self.n_cont = len(cont_names)
        self.embeds = nn.ModuleList([SharedEmbedding(ni, d_model, shared_embed=column_embed, add_shared_embed=add_shared_embed,
                                                     shared_embed_div=shared_embed_div) for ni in self.n_classes])
        n_emb = sum(self.n_classes)
        # print(self.n_cat, self.n_classes, n_cont, n_emb)
        self.n_emb,self.n_cont = n_emb,self.n_cont
        self.emb_drop = None
        if embed_dropout:
            self.emb_drop = FullEmbeddingDropout(embed_dropout) if drop_whole_embed else nn.Dropout(embed_dropout)
        self.transformer = _TabEncoder(self.n_cat, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, res_dropout=res_dropout,
                                       activation=attention_act, res_attention=res_attention, n_layers=n_layers)
        self.norm = nn.LayerNorm(self.n_cont) if norm_cont else None
        mlp_input_size = (d_model * self.n_cat) + self.n_cont
        hidden_dimensions = list(map(lambda t: int(mlp_input_size * t), mlp_mults))
        all_dimensions = [mlp_input_size, *hidden_dimensions, c_out]
        self.mlp = _MLP(all_dimensions, act=mlp_act, skip=mlp_skip, bn=mlp_bn, dropout=mlp_dropout, bn_final=bn_final)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            if self.emb_drop is not None: x = self.emb_drop(x)
            x = self.transformer(x)
            x = x.flatten(1)
        if self.n_cont != 0:
            if self.norm is not None: x_cont = self.norm(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        x = self.mlp(x)
        return x