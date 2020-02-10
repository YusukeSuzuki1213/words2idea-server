import torch
import data_prepare
import random
import spacy
import csv
import math
from model import DecoderRNN
from constants import SOS_INDEX
from constants import EOS_INDEX
from constants import TEST_N
from constants import WORD2VEC_SIZE
from constants import TRAINING_DATA_PATH
from constants import WORD1_INDEX
from constants import WORD1_FE_INDEX
from constants import WORD2_INDEX
from constants import WORD2_FE_INDEX
from constants import OUTPUT_SENTENCE_INDEX
from constants import P_TRAIN_DATASET
from constants import P_VALIDATION_DATASET
from constants import P_TEST_DATASET
from constants import MAX_SENTENCE_LENGTH

INPUT_WORD1 = "water"
INPUT_WORD2 = "pencil"


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

def word2tensor(word):
    nlp = SpacyModelSingleton.get_instance()
    word_vectors = nlp(word)
    word_tensor=torch.from_numpy(word_vectors[0].vector.reshape(1 ,1 ,WORD2VEC_SIZE)).to(DeviceSingleton.get_instance())
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
    tensor = torch.cat(tensor_lists, 1).view(1, 1, -1)
    #print("tensorFromInputWords/tensor.shape: {}".format(tensor.shape))
    return tensor


def evaluate(decoder, input_lang, output_lang, input_word1, word1_fe, input_word2, word2_fe):
    with torch.no_grad():
        input_tensor  = tensorFromInputWords(input_lang, [input_word1, word1_fe, input_word2, word2_fe])
        # -> (1, 1)
        decoder_input  = torch.tensor([[SOS_INDEX] * 1], device=DeviceSingleton.get_instance())
        #print("evaluate/decoder_input.shape: {}".format(decoder_input.shape[0]))
        decoder_hidden = input_tensor

        decoded_words = []

        # 最大何文字の出力が欲しいか
        for di in range(30):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_INDEX:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
           
            decoder_input = topi.detach()

        decoded_sentence = ' '.join(decoded_words)
        return decoded_sentence


def evaluateRandomly(data_lists, decoder, input_lang, output_lang, test_n):
    # input_langからランダムにword1, word2を取得し
    # fe_list_from_wordから入力単語に割り当てられているFEを全て取得
    """ for i in range(test_n):
        word1 = random.choice(input_lang.word_execpt_fe)
        word2 = random.choice(input_lang.word_execpt_fe)
        
        word1_fe_list = fe_list_from_word(word1)
        word2_fe_list = fe_list_from_word(word2)
        
        for word1_fe in word1_fe_list:
            for word2_fe in word2_fe_list:
                print_input_detail(word1, word1_fe, word2, word2_fe)
                output_sentence = evaluate(decoder, output_lang, word1, word1_fe, word2, word2_fe)
                print_output_sentence(output_sentence)

     """ 

    '''
    ファイルを指定して、その中からランダムに予測する処理
    '''
    """ with open(TRAINING_DATA_PATH, 'r') as f:
        current_test_n = 0
        # iterable -> list
        data_list = list(csv.reader(f))
        random.shuffle(data_list)
        
        for row in data_list:
            if current_test_n > test_n:
                break
            if not exceptData(row[WORD1_FE_INDEX], row[WORD2_FE_INDEX]):
                current_test_n += 1
                word1 = row[WORD1_INDEX]
                word1_fe = row[WORD1_FE_INDEX]
                word2 = row[WORD2_INDEX]
                word2_fe = row[WORD2_FE_INDEX]
                target_sentence = row[OUTPUT_SENTENCE_INDEX]
                
                predict_sentence = evaluate(decoder, output_lang, word1, word1_fe, word2, word2_fe)
                print_input_detail(word1, word1_fe, word2, word2_fe)
                print_sentence(predict_sentence, target_sentence)
    """

'''
    trainの時に分割した、testデータに対して予測(=テスト)をする処理
'''
def test(
    data_lists, decoder, input_lang, output_lang, p_train_dataset, p_validation_dataset, p_test_dataset, 
    word1_index, word1_fe_index, word2_index, word2_fe_index, output_sentence_index):
    n_data_lists = len(data_lists)
    print("データセットの数: {}".format(n_data_lists))
    
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

    
    
    test_data_lists = data_lists[n_train_dataset + n_validation_dataset:n_dataset]
    result_list = []
    for test_data in test_data_lists:
        word1 = test_data[WORD1_INDEX]
        word1_fe = test_data[WORD1_FE_INDEX]
        word2 = test_data[WORD2_INDEX]
        word2_fe = test_data[WORD2_FE_INDEX]

        target_sentence = test_data[OUTPUT_SENTENCE_INDEX]
        
        predict_sentence = evaluate(decoder, input_lang, output_lang, word1, word1_fe, word2, word2_fe)
        
        result_list.append(
            (word1, word1_fe, word2, word2_fe, target_sentence, predict_sentence)
        )
        
        #print_input_detail(word1, word1_fe, word2, word2_fe)
        #print_sentence(predict_sentence, target_sentence)
    # CSVタイトルを書き出し
    result_list_extracted = random.sample(result_list, 85)
    csv_title = ("word1", "word1_FE", "word2", "word2 FE", "target sentence", "predict sentence")
    # タイトル書き込み
    with open("result/predict_result.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(csv_title)
        for row in result_list_extracted:
            writer.writerow(row)


def fe_list_from_word(word: str):
    #TODO: 入力単語に割り当てられているFEを全て取得する処理
    return ["place", "instrument"]


def print_input_detail(word1, word1_fe, word2, word2_fe):
    print("========================")
    print('word1: {}, word1_fe: {}'.format(word1, word1_fe))
    print('word2: {}, word2_fe: {}'.format(word2, word2_fe))


def print_sentence(predict_sentence: str, target_sentence: str):
    print('[predict]')
    print(predict_sentence)
    print('[target]')
    print(target_sentence)


if __name__ == '__main__':
    input_lang, output_lang, data_lists = data_prepare.prepare_data()
    decoder = torch.load('result/decoder.pt')

    '''
    単語とFEを指定してテストしたい時
    '''
    """ word1_fe_list = fe_list_from_word(INPUT_WORD1)
    word2_fe_list = fe_list_from_word(INPUT_WORD2)

    for word1_fe in word1_fe_list:
        for word2_fe in word2_fe_list:
            print_input_detail(INPUT_WORD1, word1_fe, INPUT_WORD2, word2_fe)
            predict_sentence = evaluate(decoder, output_lang, INPUT_WORD1, word1_fe, INPUT_WORD2, word2_fe)
            print_sentence(predict_sentence, "TODO: target_sentence") """
    
    #evaluateRandomly(data_lists, decoder, input_lang, output_lang, TEST_N)
    test(
        data_lists = data_lists, 
        decoder = decoder, 
        input_lang = input_lang, 
        output_lang = output_lang, 
        p_train_dataset = P_TRAIN_DATASET  , 
        p_validation_dataset = P_VALIDATION_DATASET, 
        p_test_dataset = P_TEST_DATASET, 
        word1_index = WORD1_INDEX, 
        word1_fe_index = WORD2_FE_INDEX, 
        word2_index = WORD2_INDEX, 
        word2_fe_index = WORD2_FE_INDEX, 
        output_sentence_index = OUTPUT_SENTENCE_INDEX
        )