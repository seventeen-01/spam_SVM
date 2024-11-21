import numpy as np
from sklearn import svm
from sklearn import metrics
from time import time
import json
import joblib
from sklearn.model_selection import train_test_split
from scipy import sparse, io
import sys


# log
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger("mylog.txt")
# 下面所有的方法，只要控制台输出，都将写入"mylog.txt"


'''RBF的训练函数'''


class TrainerRbf:
    def __init__(self, training_data, training_target):
        # 初始化训练数据和目标标签
        self.training_data = training_data
        self.training_target = training_target
        # 创建RBF核的SVM分类器，设置C和gamma参数
        self.clf = svm.SVC(C=100, kernel='rbf', gamma=0.01, verbose=False)

    def train_classifier(self):
        # 训练分类器
        self.clf.fit(self.training_data, self.training_target)
        # 保存训练好的模型
        joblib.dump(self.clf, 'model/SVM_RBF.pkl')
        # 使用训练数据进行预测
        training_result = self.clf.predict(self.training_data)
        # 打印分类报告
        print(metrics.classification_report(self.training_target, training_result))


'''性能评估的函数，主要包括混淆矩阵以及正例负例精确度一类的'''


def performance_report(target, result):
    # 生成混淆矩阵
    confusion = metrics.confusion_matrix(target, result)
    print('confusion matrix')
    print(confusion)

    # 从混淆矩阵提取真正例（TP）、假负例（FN）、假正例（FP）和真负例（TN）
    TP = int(confusion[0, 0])
    FN = int(confusion[0, 1])
    FP = int(confusion[1, 0])
    TN = int(confusion[1, 1])

    # 计算精确度、召回率、准确率和F1分数
    Accuracys = float(TP + TN) / (TP + FP + TN + FN)
    Precisions = float(TP) / (TP + FP)
    Recalls = float(TP) / (TP + FN)
    f_value = 2 * Recalls * Precisions / (Recalls + Precisions)

    # 打印评估结果
    print("TP:" + str(TP))
    print("TN:" + str(TN))
    print("FP:" + str(FP))
    print("FN:" + str(FN))
    print("Recalls: %s" % str(Recalls))
    print("Precisions: %s" % str(Precisions))
    print("Accuracys: %s" % str(Accuracys))
    print("f_value: %s" % str(f_value))


'''数据分割'''


# 分割数据为训练集和测试集
# x: 包含所有特征的数据集，每行是一个样本，每列是一个特征
# y: 包含所有样本标签的数组
# takeup: 指定测试集占总数据集的比例
# random_state=20 确保每次运行代码时，数据分割的方式都是一样的，以便于结果的可重复性
def select_data(x, y, takeup):
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=takeup, random_state=20)  # test data take up takeup, random_seed is 20
    return train_x, test_x, train_y, test_y
 # 返回分割后的训练集特征(train_x)、测试集特征(test_x)、训练集标签(train_y)和测试集标签(test_y)


######################################################################################

if '__main__' == __name__:
    # how much test data take up
    # 0.1 indicates test data take up 10%
    print("********************** trainning start **********************")
    t0 = time()
    takeup = 0.1
    x = io.mmread('Data/X.mtx')
    with open('Data/y.json', 'r') as f:
        y = json.load(f)
    train_x, test_x, train_y, test_y = select_data(x, y, takeup)
    print('takeup finished')

    # train num1
    print('#################### train svm_RBF #####################')
    start_time1 = time()
    TrainerRbf(train_x, train_y).train_classifier()
    print('training took %fs!' % (time() - start_time1))


    modela = joblib.load('model/SVM_RBF.pkl')
    predict = modela.predict(test_x)
    performance_report(test_y, predict)
    print("*************** Training done in %0.3fs ***************\n\n" % ((time() - t0)))

