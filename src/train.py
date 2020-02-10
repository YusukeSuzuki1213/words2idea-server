import data_prepare
import torch
import time
import random
import spacy
import math
from model import DecoderRNN
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from constants import PLOT_EVERY
from constants import LEARNING_RATE
from constants import HIDDEN_SIZE
from constants import EOS_INDEX
from constants import SOS_INDEX
from constants import WORD2VEC_SIZE
from constants import TEACHER_FORCING_RATIO
from constants import BATCH_SIZE
from constants import TRAIN_SHUFFLE
from constants import VALIDATION_SHUFFLE
from constants import TEST_SHUFFLE
from constants import EPOCHS
from constants import P_TRAIN_DATASET
from constants import P_VALIDATION_DATASET
from constants import P_TEST_DATASET
from constants import PAD_INDEX
from constants import EMBED_SIZE

# TODO: DeviceSingleton()で呼んだらシングルトンではなくなる問題の解消
class DeviceSingleton:
    _device = None

    @classmethod
    def get_instance(cls):
        if not cls._device:
            print("You can use cuda !" if torch.cuda.is_available() else "Execute cpu... Can you use cuda ??")
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return cls._device


# TODO: SpacyModelSingleton()で呼んだらシングルトンではなくなる問題の解消
class SpacyModelSingleton:
    _spacy_model = None

    @classmethod
    def get_instance(cls):
        if not cls._spacy_model:
            cls._spacy_model = spacy.load('en_vectors_web_lg')

        return cls._spacy_model


def savePlot(points, epochs):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.4)
    ax.yaxis.set_major_locator(loc)
    plt.plot(range(1, epochs+1), points)
    plt.savefig('result/loss.png')


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


'''
output文章をテンソルに変換

Params
-------
    lang : Land
        output_lang
    sentence : str
        output文章
Returns
-------
    : torch.tensor(1, lang.max_length+1)
'''
def tensorFromSentence(lang, sentence):
    # e.g. "I have a pen ." -> ["I", "have", "a", "pen", "."]
    sentence_word_list = sentence.split(' ')
    length = len(sentence_word_list)
    # e.g. ["I", "have", "a", "pen", "."] -> [9, 8, 3, 4, 2]
    indexes = [lang.word2index[word] for word in sentence_word_list]
    # 最大長 + 1に合わせる 
    # e.g. 出力の最大長(EOS含まず)が8の場合indexesは9に合わせる: [9, 8, 3, 4, 2]          -> [9, 8, 3, 4, 2, EOS_INDEX, 0, 0, 0]
    # e.g. 出力の最大長(EOS含まず)が8の場合indexesは9に合わせる: [9, 8, 3, 4, 2, 4, 9, 6] -> [9, 8, 3, 4, 2, 4, 9, 6, EOS_INDEX]
    indexes = indexes + [EOS_INDEX] + [PAD_INDEX] * (lang.max_length - length)

    tensor = torch.tensor(indexes, dtype=torch.long, device=DeviceSingleton.get_instance()).view(1, -1)
    #print("tensor.shape: {}".format(tensor.shape))
    return tensor


'''
単語のベクトル表現を学習済みモデルから取得する

Params
-------
    word : str
Returns
-------
    word_tensor : torch.tensor(1,WORD2VEC_SIZE)
'''
def word2tensor(word):
    nlp = SpacyModelSingleton.get_instance()
    word_vectors = nlp(word)
    word_tensor=torch.from_numpy(word_vectors[0].vector.reshape(1 ,WORD2VEC_SIZE)).to(DeviceSingleton.get_instance())
    return word_tensor


'''
inputのテンソルを返す

Params
-------
    lang : Lang
        入力Lang
    input_list : List[str, str, str, str]
        教師データからoutput sentenceを抜いたリスト e.g. ['water', 'place', 'pencil', 'instrument']
Returns
-------
    tensor : torch.tensor(1,WORD2VEC_SIZE*4)
'''
def tensorFromInputWords(lang, input_list):
    tensor_lists = []
    for word in input_list:
        tensor_lists.append(word2tensor(word))
    tensor = torch.cat(tensor_lists, 1)
    #print("tensorFromInputWords/tensor.shape: {}".format(tensor.shape))
    return tensor


def generate_data_loaders(
    data_lists, p_train_dataset, p_validation_dataset, p_test_dataset, train_shuffle, 
    validation_shuffle, test_shuffle, input_lang, output_lang, batch_size):
    n_data_lists = len(data_lists)
    print("データセットの数: {}".format(n_data_lists))
    
    input_tensor_lists = []
    target_tensor_lists = []
    
    for data_list in data_lists:
        input_tensor = tensorFromInputWords(input_lang, data_list[:4])
        target_tensor = tensorFromSentence(output_lang, data_list[4])
        input_tensor_lists.append(input_tensor)
        target_tensor_lists.append(target_tensor)
    
    input_tensors = torch.cat(input_tensor_lists, dim=0)
    print("generate_data_loaders/input_tensors.shape: except=({}, {}) result={}".format(n_data_lists, WORD2VEC_SIZE*4, input_tensors.shape))
    target_tensors = torch.cat(target_tensor_lists, dim=0)
    print("generate_data_loaders/output_tensors.shape: except=({}, {}) result={}".format(n_data_lists, output_lang.max_length+1, target_tensors.shape))

    # データセットの数
    n_dataset = n_data_lists
    print("n_dataset: {}".format(n_dataset))
    # train dataの数
    n_train_dataset = math.floor(n_dataset * p_train_dataset)
    print("n_train_dataset: {}".format(n_train_dataset))
    # validation dataの数
    n_validation_dataset = math.floor(n_dataset * p_validation_dataset)
    print("n_validation_dataset: {}".format(n_validation_dataset))
    # test dataの数
    n_test_dataset = math.floor(n_dataset * p_test_dataset)
    print("n_test_dataset: {}".format(n_test_dataset))

    tensor_dataset = torch.utils.data.TensorDataset(input_tensors, target_tensors)
    
    train_dataset = torch.utils.data.Subset(tensor_dataset, list(range(0, n_train_dataset)))
    validation_dataset = torch.utils.data.Subset(tensor_dataset, list(range(n_train_dataset, n_train_dataset + n_validation_dataset)))
    test_dataset = torch.utils.data.Subset(tensor_dataset, list(range(n_train_dataset + n_validation_dataset, n_dataset)))

    #　ミニバッチを返すiterableなオブジェクトを生成
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=validation_shuffle)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle)
    
    return train_data_loader, validation_data_loader, test_data_loader
            

def train_batch(input_batch, target_batch, decoder_optimizer, output_lang, decoder, criterion, teacher_forcing_ratio):
    # input_batch  : tensor(batch_size, 1200)
    # target_batch : tensor(batch_size, input_lang.max_length+1)
    #print("train_batch/input_batch: expect=(1, batch_size, WORD2VEC_SIZE*4) result={}".format(input_batch.shape))
    #print("train_batch/target_batch: expect=(batch_size, input_lang.max_length+1) result={}".format(target_batch.shape))
    
    loss = 0
    # 一度計算された勾配結果を0にリセット
    decoder_optimizer.zero_grad()
    # 現在のバッチのバッチサイズを取得
    batch_size = input_batch.shape[0]
    # decoderへのinputの最大長を取得(NOTE: ここは現在のミニバッチでの最大長のほうがいいかも)
    target_length = output_lang.max_length + 1 # +1はEOS
    # (1, batch_size)
    decoder_input = torch.tensor([[SOS_INDEX] * batch_size], device=DeviceSingleton.get_instance())
    # (1, batch_size), (target_length, batch_size) -> (target_length+1, batch_size)
    decoder_inputs = torch.cat([decoder_input, target_batch.transpose(0, 1)], dim=0)
    #print("train_batch/decoder_inputs.shape: {}".format(decoder_inputs.shape))
    # (batch_size, WORD2VEC_SIZE*4) -> (1, batch_size, WORD2VEC_SIZE*4)
    decoder_hidden = input_batch.view(1, batch_size, -1)
    # Teacher_forcingを使うか否か
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False                                    
    
    if use_teacher_forcing:
        for di in range(target_length):
            #print("train/decoder_inputs[di]: result={}".format(decoder_inputs[di].shape))
            decoder_output, decoder_hidden = decoder(decoder_inputs[di], decoder_hidden)
            #print("train/decoder_hidden: expect=(1, batch_size, WORD2VEC_SIZE*4) result={}".format(decoder_hidden.shape))
            loss += criterion(decoder_output, decoder_inputs[di+1])                        
    else:
        decoder_input = decoder_inputs[0]
        #print("train/decoder_inputs[0]: result={}".format(decoder_inputs[0].shape))
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, topi = decoder_output.topk(1)  # (b,odim) -> (b,1)                        
            decoder_input = topi.squeeze(1).detach() 
            loss += criterion(decoder_output, decoder_inputs[di+1])
            #if decoder_input.item()  == EOS_INDEX:
            #    break
    
    loss.backward()

    decoder_optimizer.step()

    return loss.item() / target_length


def train(
    data_lists, p_train_dataset, p_validation_dataset, p_test_dataset, train_shuffle, 
    validation_shuffle, test_shuffle, epochs, plot_every, input_lang, output_lang, 
    decoder, batch_size, learning_rate, teacher_forcing_ratio):
    
    plot_loss_total = 0
    current_batch_size = 0
    plot_losses = []
    # データローダを作成
    train_data_loader, validation_data_loader, _ = generate_data_loaders(data_lists, p_train_dataset, p_validation_dataset, p_test_dataset, train_shuffle, validation_shuffle, test_shuffle, input_lang, output_lang, batch_size)    

    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss(ignore_index=PAD_INDEX)

    for epoch in range(epochs):
        print("epoch: {}/{}".format(epoch, epochs))
        for input_batch, target_batch in train_data_loader:
            current_batch_size += 1
            loss = train_batch(input_batch, target_batch, decoder_optimizer, output_lang, decoder, criterion, teacher_forcing_ratio)
            #print("loss: {}".format(loss))
            plot_loss_total += loss
        
        # PLOT_EVERY epoch毎にロスをプロット
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / current_batch_size          
            print("[plot_loss_avg]: {}".format(plot_loss_avg))
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            current_batch_size = 0

        # TODO: validationもやる
    savePlot(plot_losses, epochs)

    torch.save(decoder, 'result/decoder.pt')


if __name__ == '__main__':
    input_lang, output_lang, data_lists = data_prepare.prepare_data()
    
    """ print("-------------input_lang-------------")
    print("index2word: {}".format(input_lang.index2word))
    print("------------output_lang-------------")
    print("index2word: {}".format(output_lang.index2word))
    print("-------------最終的な教師データ-------------")
    print("data_lists: {}".format(data_lists)) """

    decoder = DecoderRNN(HIDDEN_SIZE, output_lang.n_words, EMBED_SIZE, DeviceSingleton.get_instance()).to(DeviceSingleton.get_instance())

    # トレーニング
    train(
        data_lists = data_lists, 
        p_train_dataset = P_TRAIN_DATASET, 
        p_validation_dataset = P_VALIDATION_DATASET, 
        p_test_dataset = P_TEST_DATASET, 
        train_shuffle = TRAIN_SHUFFLE, 
        validation_shuffle = VALIDATION_SHUFFLE,
        test_shuffle = TEST_SHUFFLE, 
        epochs = EPOCHS, 
        plot_every = PLOT_EVERY,
        input_lang = input_lang, 
        output_lang = output_lang, 
        decoder = decoder, 
        batch_size = BATCH_SIZE, 
        learning_rate = LEARNING_RATE,
        teacher_forcing_ratio = TEACHER_FORCING_RATIO
        )

    
