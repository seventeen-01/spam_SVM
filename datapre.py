import numpy as np
import json
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer as OriginalTfidfVectorizer
from scipy import sparse, io
from time import time
import sys
import joblib

#log
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a", encoding='utf-8')  # 指定编码为utf-8
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
sys.stdout = Logger("mylog.txt")
#下面所有的方法，只要控制台输出，都将写入"mylog.txt"

def load_stopwords(filepath='stopwords.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f]
    return stopwords

# 定制化的TF-IDF向量化器
class TfidfVectorizer(OriginalTfidfVectorizer):
    '''去除没用的标点，再次分词，并过滤停用词'''

    def __init__(self, stopwords=None, **kwargs):
        self.stopwords = stopwords if stopwords else load_stopwords()
        super().__init__(**kwargs)

    def build_analyzer(self):
        # 定义分析器，用于处理中文文本
        def analyzer(doc):
            words = pseg.cut(doc)  # 使用jieba进行分词和词性标注
            new_doc = ''.join(w.word for w in words if w.flag != 'x')  # 过滤掉标点符号等非词汇元素
            words = jieba.cut(new_doc)  # 对处理后的文本重新进行分词
            filtered_words = [word for word in words if word not in self.stopwords]  # 过滤停用词
            return filtered_words
        return analyzer



if '__main__' == __name__:
    print('******************* data preprocessing ********************')
    t0 = time()
    data_lines = 20000  # 设定要处理的数据行数
    data_type = "raw"  # 设置数据处理的类型（原始数据、PCA、NMF、或PCA和NMF结合）

    x = []  # 用于存储文本数据
    y = []  # 用于存储标签

    with open('Data/label.txt', 'r', encoding='utf-8') as fr:
        for i in range(data_lines):
            line = fr.readline()  # 逐行读取
            message = line.split('\t')  # 使用制表符分割标签和文本内容
            y.append(message[0])  # 提取并存储标签
            x.append(message[1])  #


    with open('Data/y.json', 'w', encoding='utf-8') as f:
        json.dump(y, f)  # 使用json.dump将标签数据写入文件
    print("save y successfully!")

    vec_tfidf = TfidfVectorizer()  # 创建TF-IDF向量化器实例
    data_tfidf = vec_tfidf.fit_transform(x)  # 调用fit_transform方法将文本数据转换为TF-IDF特征矩阵
    joblib.dump(vec_tfidf, "Data/vec_tfidf.pkl")  # 使用joblib.dump保存训练好的向量化器
    print("save vec_tfidf successfully!")

    if data_type == 'raw':
        io.mmwrite('Data/X.mtx', data_tfidf)  # 使用io.mmwrite将TF-IDF特征矩阵保存为MM格式的文件
        print("save X successfully!")

    name_tfidf_feature = vec_tfidf.get_feature_names_out()  # 获取TF-IDF特征的名称
    with open('Data/feature.json', 'w', encoding='utf-8') as f:
        json.dump(list(name_tfidf_feature), f)  # 将特征名称保存到JSON文件中
    print("save feature successfully!")



    print("******* %s lines data preprocessing done in %0.3fs *******\n\n" % (data_lines, (time() - t0)))


