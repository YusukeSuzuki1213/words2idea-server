'''
ハイパーパラメータ関係
'''
# 学習率
LEARNING_RATE = 0.01
# decoderのhiddentの次元数
HIDDEN_SIZE = 1200
# decoderのinputのemdeddingの次元数
EMBED_SIZE = 300
# word2vec vector size
WORD2VEC_SIZE = 300
# PLOT_EVERY毎にグラフにプロット
PLOT_EVERY = 1
# Teacher forcingを使用する確率
TEACHER_FORCING_RATIO = 0.5
# バッチサイズ
BATCH_SIZE = 256
# エポック数
EPOCHS = 1450
# トレーニングデータセットのTensorDatasetの中身の組の順序をシャッフルするか
TRAIN_SHUFFLE = True
# トレーニングデータセットのTensorDatasetの中身の組の順序をシャッフルするか
VALIDATION_SHUFFLE = False
# トレーニングデータセットのTensorDatasetの中身の組の順序をシャッフルするか
TEST_SHUFFLE = False
# train データの割合
P_TRAIN_DATASET = 0.8
# validation データの割合
P_VALIDATION_DATASET = 0.1
# test データの割合
P_TEST_DATASET = 0.1

'''
前処理
'''
TRAINING_DATA_PATH = 'data/data_v3.csv'
#TRAINING_DATA_PATH = 'data/data_limited.csv'
 # 入力単語数
INPUT_WORDS = 4
# PAD文字のindex
PAD_INDEX = 0
# SOS文字のindex
SOS_INDEX = 1
# EOS文字のindex
EOS_INDEX = 2

'''
テスト
'''
# テストの個数
TEST_N = 20
# 出力する最大文字長
MAX_SENTENCE_LENGTH = 25

'''
共通
'''
# 教師データのindex
# 教師データのword1のindex
WORD1_INDEX = 0
# 教師データのword1_feのindex
WORD1_FE_INDEX = 1
# 教師データのword2のindex
WORD2_INDEX = 2
# 教師データのword2_feのindex
WORD2_FE_INDEX = 3
 # 教師データリストのoutput文章のindex
OUTPUT_SENTENCE_INDEX = 4
