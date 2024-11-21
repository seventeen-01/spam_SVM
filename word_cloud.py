import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba  # 用于中文分词
from collections import Counter

# 读取数据集
file_path = r'A:\spam_SVM\Data\label2.txt'
data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'message'])

# 加载停用词表（这里假设你有一个名为stopwords.txt的文件包含停用词）
stopwords_path = r'A:\spam_SVM\stopwords.txt'  # 请替换为实际路径
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = set(f.read().splitlines())


# 分词函数，同时去除停用词
def tokenize(text):
    words = jieba.cut(text)
    return [word for word in words if word not in stopwords and word.strip()]


# 分别处理正常短信和垃圾短信
normal_messages = data[data['label'] == 0]['message'].tolist()
spam_messages = data[data['label'] == 1]['message'].tolist()

normal_tokens = [tokenize(msg) for msg in normal_messages]
spam_tokens = [tokenize(msg) for msg in spam_messages]

# 合并所有分词结果，以便进行词频统计
normal_all_tokens = [token for sublist in normal_tokens for token in sublist]
spam_all_tokens = [token for sublist in spam_tokens for token in sublist]

# 词频统计
normal_word_freq = Counter(normal_all_tokens)
spam_word_freq = Counter(spam_all_tokens)


# 生成词云图
def generate_wordcloud(word_freq, title, font_path):
    wordcloud = WordCloud(font_path=font_path,
                          width=800,
                          height=400,
                          background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()


# 指定中文字体路径
font_path = r'C:\Windows\Fonts\SIMYOU.TTF'

# 生成正常短信的词云图
generate_wordcloud(dict(normal_word_freq), '正常短信词云图', font_path)

# 生成垃圾短信的词云图
generate_wordcloud(dict(spam_word_freq), '垃圾短信词云图', font_path)