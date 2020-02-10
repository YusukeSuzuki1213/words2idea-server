import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import PAD_INDEX

class DecoderRNN(nn.Module):
    '''
    コンストラクタ

    Params
    ----------
    hidden_size: int
        GRUの隠れ状態の次元数
    output_size: int
        出力テンソルのサイズ(=decoderに登場単語の合計)
        (単語を数値で扱うには単語数分の要素を持つone-hotが必要)
    devece: hoge
        cpuかgpuかどっちを使用するか
    '''  
    def __init__(self, hidden_size, input_size, emb_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # arg1: 埋め込み辞書のサイズ(=何文字表現したいか)
        # arg2: 埋め込みベクトル要素のサイズ
        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=PAD_INDEX) 
        # arg1: input_size(入力の次元数), arg2: hidden_size(出力の次元数)
        self.gru = nn.GRU(emb_size, hidden_size)
        # arg1: in_features – 入力の次元数
        # arg2: out_features – 出力の次元数
        self.out = nn.Linear(hidden_size, input_size)
        # arg1: dim(LogSoftmaxが計算される次元)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device
    
    '''

    Params
    -------
        input: torch.tensor
            decoderへの入力[WORD2VEC_SIZE] e.g. [[SOS],[SOS], ..., [EOS]]
    '''
    def forward(self, input, hidden):
        output = self.embedding(input).view(1, input.shape[0], -1) # (batch) -> (batch, input)
        #print("output: {}".format(output.shape))
        output = F.relu(output)
        # (batch, input), (batch, WORD2VEC_SIZE*4) -> ()
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    '''
    Decoderに入力される隠れ状態の初期値
    TODO: この初期値はEncoderの隠れ状態になるため必要ない？

    Returns
    ----------
    tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., ..., o_size]]])
        1 × 1 × self.hidden_sizeのテンソル
    '''
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)
        

