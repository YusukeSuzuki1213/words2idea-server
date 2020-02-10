import re
import csv
from constants import INPUT_WORDS
from constants import OUTPUT_SENTENCE_INDEX
from constants import WORD1_INDEX
from constants import WORD2_INDEX
from constants import TRAINING_DATA_PATH
from constants import PAD_INDEX
from constants import SOS_INDEX
from constants import EOS_INDEX

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {} # word->index e.g. {"SOS": 0, "EOS": 1}
        self.word2count = {} # how many word appear e.g. {"SOS": 15, "EOS": 12}
        self.index2word = { PAD_INDEX: "PAD", SOS_INDEX: "SOS", EOS_INDEX: "EOS"} # index -> word
        self.n_words = 3  # 全ての単語の数
        self.word_execpt_fe = [] # FEを抜きにした入力のリスト e.g. ["bottom", "look", "water"...]
        self.max_length = 0

    def addSentence(self, sentence):
        # e.g. "I have a pen" -> ["I", "have", "a", "pen"]
        sentence_word_list = sentence.split(' ') 
        sentence_length = len(sentence_word_list)
        # 出力文章の最大長を保存
        if self.max_length < sentence_length:
            self.max_length = sentence_length
        for word in sentence_word_list:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def addWordExceptFe(self, word):
        if word not in self.word_execpt_fe:
            self.word_execpt_fe.append(word) 

'''
センテンスの正規化
TODO: 自然言語処理で使用される他の正規化の手法を調べる

Params
-------
    s: str
        正規化前の文字列
Returns
-------
    s : str
        正規化後の文字列
'''
def normalizeString(s):
    # 小文字に
    s = s.lower()
    # スペース,タブ,改行を削除
    s = s.strip()
    # unicode -> Ascii
    #s = unicodeToAscii(s)
    # 文末の[.!?]に空白をあける
    s = re.sub(r"([.!?])", r" \1", s) 
    # アルファベットと.!?以外は空白に置換
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s) 
    return s


'''
このデータは教師データとして不適切か？
例えばFEが連結されていないか? e.g. contain_object
TODO: 一時的な関数
      FEが複数単語で構成されている場合の対処をしなければならない。
      現時点ではそれは教師データとして使用しない方針。
TODO: 最大長とかは？
TODO: そもそもデータ作成の段階で弾いておく作戦もあり

Params
-------
    data: str
        教師データの1つの要素(単語, FE, 文章) e.g. 'water', 'place' , etc.
Returns
-------
    do_contain: bool
        不適切 -> True
'''
def exceptData(data): 
    if '_' in data:
        return True
    else:
        return False


'''
このデータを教師データにから除くか？

Params
-------
    data_list: list[str, str, str, str]
        1つの教師データ e.g. ['water', 'place', 'pencil', 'instrument', 'hoge']
Returns
-------
     do_execpt: bool
        除く -> True, 含む -> False
'''
def exceptDataList(data_list):
    for data in data_list:
        do_except = exceptData(data)
        if do_except:
            return True
            
    return False


'''
CSVから教師データを読み込む

Returns
-------
 : list[list[str, str, str, str, str]]
    教師データのリスト
'''
def read_data():
    print("教師データ: {}".format(TRAINING_DATA_PATH))
    with open(TRAINING_DATA_PATH) as f:
        reader = csv.reader(f)
        data_lists = [ data_list for data_list in reader if(not exceptDataList(data_list))]
        # 全ての要素に対して正規化
        data_lists = [[normalizeString(data) for data in data_list] for data_list in data_lists]
    return data_lists


'''
教師データを取得

Returns
-------
data_lists : list[list[str, str, str, str, str]]
    教師データのリスト
'''
def prepare_data():
    data_lists  = read_data()
    input_lang = Lang("input")
    output_lang = Lang("output")

    # inputの最大単語数を格納
    input_lang.max_length = INPUT_WORDS

    for data_list in data_lists:
        for i in range(INPUT_WORDS):
            # word1, word1_fe, word2, word2_feを格納
            input_lang.addWord(data_list[i])
        # output_sentenceを格納
        output_lang.addSentence(data_list[OUTPUT_SENTENCE_INDEX])
        input_lang.addWordExceptFe(data_list[WORD1_INDEX])
        input_lang.addWordExceptFe(data_list[WORD2_INDEX])
    
    return input_lang, output_lang, data_lists
   