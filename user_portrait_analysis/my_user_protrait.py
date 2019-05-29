import csv
import pandas as pd
import jieba.analyse
import time
import jieba
import jieba.posseg
import os
import sys
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

#数据预处理
#编码方式转换
'''
#原始数据存储路径
data_path = './data/user_tag_query.10W.TRAIN'
#生成数据路径
csvfile = open(data_path+ '-2W.csv', 'w')
#写入文件
writer = csv.writer(csvfile)
writer.writerow(['ID', 'age', 'Gender', 'Education', 'QueryList'])
#转换为utf-8编码的格式
with open(data_path, 'r', encoding='gbk', errors='ignore') as f:
    #读取所有行
    lines = f.readlines()
    # print(len(lines))
    for line in lines[0:10000]:
        try:
            #去除开头和结尾的字符
            line.strip()
            data = line.split('\t')
            writerdata = [data[0], data[1], data[2], data[3]]
            querystr = ''
            #去掉换行符
            data[-1] = data[-1][:-1]
            for d in data[4:]:
                try:
                    #进行utf编码和utf-8 解码
                    cur_str = d.encode('utf-8')
                    cur_str = cur_str.decode('utf-8')
                    #每一行之后加上tab
                    querystr += cur_str + '\t'
                    # print(querystr)
                except:
                    continue
            #去掉最后一个tab
            querystr = querystr[:-1]
            writerdata.append(querystr)
            writer.writerow(writerdata)
            # print(querystr)
        except:
            continue

#编码转换完成的数据，取的是1w的子集
trainname = 'data/user_tag_query.10W.TRAIN-2w.csv'
data = pd.read_csv(trainname, encoding='gbk')
print(data.info())

#分别生成三种标签数据（年龄，性别，学历）
data.age.to_csv("data/train_age2.csv", index=False)
data.Gender.to_csv("data/train_gender2.csv", index=False)
data.Education.to_csv("data/train_education2.csv", index=False)
#讲搜索数据单独拿出来
data.QueryList.to_csv("data/train_querylist2.csv", index=False)

# 对数据搜索内容进行分词
def input(trainname):
    traindata = []
    with open(trainname, 'rb') as f:
        #读取一行
        line = f.readline()
        # print(line)
        count = 0
        while line :
            try:
                #一行所有的词 添加到traindata中
                traindata.append(line)
                count += 1
            except:
                print('error:', line,count)
            line = f.readline()
    return traindata
start = time.clock()
filepath = 'data/train_querylist2.csv'
QueryList = input(filepath)
print(QueryList)

writepath = 'data/train_querylist_writefile-2w.csv'
csvfile = open(writepath, 'w')

#part-of-speech tagging 词性标注
POS = {}
for i in range(len(QueryList)):
    # print(QueryList[i])
    if i % 2000 == 0 and i >= 1000:
        print(i, 'finished')
    s = []
    str = ''
    # 带有词性的精准分词模式
    words = jieba.posseg.cut(QueryList[i])
    #过滤词性
    allowPOS = ['n', 'v', 'j']
    for word, flag in words:
        # print(word, flag)
        # 柔和 a  双沟 n,x 女生  n,x  中财网 nt
        #查看词性是否存在 存
        POS[flag] = POS.get(flag, 0) + 1
        if(flag[0] in allowPOS) and len(word) >= 2:
            str += word + ' '

    cur_str = str.encode('utf-8')
    cur_str = cur_str.decode('utf-8')
    s.append(cur_str)

    csvfile.write(''.join(s) + '\n')
csvfile.close()

end = time.clock()
print("total time: %f s"%(end - start))
print(POS)
# {'x': 1564189, 'a': 162725, 'n': 1926604, 'nt': 22060, 'm': 287861, 'eng': 181584, 'nr': 389172, 'nz': 103056, 'ad': 17686, 'v': 882821, 'd': 98341, 'vg': 10400, 'uj': 183163, 'p': 80655, 'ns': 244373, 'ng': 26735, 'r': 249706, 'vn': 90413, 'j': 33146, 's': 17685, 'u': 26102, 'c': 54309, 't': 49835, 'q': 31200, 'f': 53101, 'l': 55936, 'b': 33680, 'i': 29421, 'zg': 22495, 'nrt': 28695, 'ul': 22272, 'y': 55560, 'mq': 636, 'g': 6518, 'z': 9512, 'o': 2222, 'nrfg': 5527, 'k': 2829, 'tg': 2894, 'ud': 2051, 'an': 2684, 'uv': 1012, 'ug': 2932, 'e': 904, 'uz': 1859, 'df': 1226, 'vq': 33, 'yg': 941, 'ag': 3725, 'rr': 14, 'h': 465, 'vd': 74, 'mg': 61, 'rg': 139, 'vi': 2, 'dg': 33, 'rz': 3}
'''

#(二) 特征选择

#建立word2vec词向量模型
#使用Gensim库建立word2vec词向量模型
#参数定义：
#sentences： 可以使一个list
#sg： 用于设置训练算法，默认为0，对应CBOW算法； sg=1则采用skip-gram算法
#size： 是指特征向量的维度，默认为100. 大的size需要更多的训练数据，但是效果会更好
#window ： 表示当前词与预测词在一个句子中的最大距离是多少
#alpha： 是学习速率
#seed：用于随机数发生器，与初始化词向量有关
#min_count: 可以对字典做截断， 词频少于min_count次数的单词会被丢弃掉，默认值为5
#max_vocab_size: 设置词向量构建期间RAM机制，如果所有独立单词个数超过这个，则就消除掉其中最不频发的一个。每一千万个单词需要大约1GB的RAM。设置为None则没有限制。
#workres 参数控制训练的并行数
#hs 如果为1则会采用hierarchica*softmax技巧。如果设置为0（default）， 则negative sampling会被使用。
#nagetive ： 如果>0,则会采用natative sampling，用于设置多少个noise words
#iter： 迭代次数，默认为5

#讲数据变换为  list of list格式
# pip install --upgrade smart_open
#
# train_path = 'data/train_querylist_writefile-1w.csv'
# with open(train_path, 'r') as f:
#     My_list = []
#     lines = f.readlines()
#     for line in lines:
#         # print(line)
#         cur_list = []
#         line = line.strip()
#         data = line.split(' ')
#         for d in data:
#             cur_list.append(d)
#         My_list.append(cur_list)
#     model = word2vec.Word2Vec(My_list, size=300, window=10, workers=4)
#     #保存model的路径
#     savepath = '2w_word2vec_' + '300' + '.model'
#     model.save(savepath)

# model = Word2Vec.load('2w_word2vec_300.model')
# print(model.most_similar("大哥"))
# print(model.most_similar("清华"))
# print(len(model['大哥']))
# print(model['大哥'])

#对所有搜索数据求平均向量
#加载训练好的word2vec模型，求用户搜索结果的平均向量
file_name = 'data/train_querylist_writefile-2w.csv'
cur_model = Word2Vec.load('2w_word2vec_300.model')
with open(file_name, 'r') as f:
    cur_index = 0
    lines = f.readlines()
    doc_cev = np.zeros((len(lines),300))
    for line in lines:
        word_vec = np.zeros((1, 300))
        words = line.strip().split(' ')
        word_num = 0
        # print(words)
        #求模型的平均向量
        for word in words:
            if word in cur_model:
                word_num += 1
                word_vec += np.array([cur_model[word]])

        doc_cev[cur_index] = word_vec / float(word_num)
        # print(doc_cev[cur_index])
        cur_index += 1

print(doc_cev.shape)
print(doc_cev[0])

#获取性别的y标签
genderlabel = np.loadtxt(open('data/train_gender2.csv', 'r')).astype(int)
print(genderlabel.shape)

#获取学历的y标签
educationlabel = np.loadtxt(open('data/train_education.csv', 'r')).astype(int)
print(educationlabel.shape)

#获取年龄的y标签
agelabel = np.loadtxt(open('data/train_age.csv', 'r')).astype(int)
print(agelabel.shape)

def removezero(x,y):
    """
    把标签列Y为0的去除掉，对应Y为0的X矩阵的行也相应去掉
    :param x: 列表包含一个个用户搜素词的平均向量
    :param y: 用户性别标签列/年龄标签列/教育标签列
    :return: 返回去除标签列为0的记录X和y
    """
    #获取所有值为0的下标
    nozero = np.nonzero(y)
    #获取到所有y值
    y = y[nozero]
    x = np.array(x)
    # 获取到所有x值
    x = x[nozero]
    return  x, y

#分别构建 每个模型的x和y的数据集
gender_train, genderlabel = removezero(doc_cev, genderlabel)
age_train, agelabel = removezero(doc_cev, agelabel)
education_train, educationlabel = removezero(doc_cev, educationlabel)

print(gender_train.shape, genderlabel.shape)
print(age_train.shape, agelabel.shape)
print(education_train.shape, educationlabel.shape)

#定义一个函数去绘制混淆矩阵， 为了评估看着方便
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
        此方法绘制并输出混淆矩阵
    :param cm: 大小
    :param title: 名称
    :param cmap: 颜色
    :return:
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #加上图里面的颜色渐进条
    plt.colorbar()
    #分别给横纵坐标在0和1的位置写上数字0和1
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    #在混淆矩阵图形四象限的格子里面写上数值，如果底色深就用白色
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i,j],horizontalalignment='center',
                 color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')

#（三） 建模预测
#不同的机器学习模型对比

X_train, X_test, y_train, y_test = train_test_split(gender_train, genderlabel, test_size=0.2, random_state=0)
# #建立一个基础的预测模型
# LR_model = LogisticRegression()
# LR_model.fit(X_train, y_train)
# y_pred = LR_model.predict(X_test)
# print(accuracy_score(y_test, y_pred))
# cnf_matrix = confusion_matrix(y_test, y_pred)
# print("Recall metric in the testing dataset: ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))
# print("accuracy metric in the testing dataset: ",
#       (cnf_matrix[1, 1]+cnf_matrix[0, 0])/(cnf_matrix[0, 0]+cnf_matrix[1, 1]+ cnf_matrix[1, 0]+ cnf_matrix[0, 1]))
#
# class_names = [0,1] #0是负例， 1是正例
# #创建一张图
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, title='Gender-Confusion matrix')
#
# plt.show()
# #建立随机森林的的预测模型
# RF_model = RandomForestClassifier(n_estimators=300, min_samples_split=5, max_depth=10)
# RF_model.fit(X_train, y_train)
# y_pred = RF_model.predict(X_test)
# print(accuracy_score(y_test, y_pred))
#
# cnf_matrix - confusion_matrix(y_test, y_pred)
# print("Recall metric in the testing dataset: ", cnf_matrix[1, 1]/(cnf_matrix[1, 0]+cnf_matrix[1, 1]))
# print("accuracy metric in the testing dataset: ",
#       (cnf_matrix[1, 1]+cnf_matrix[0, 0])/(cnf_matrix[0, 0]+cnf_matrix[1, 1]+ cnf_matrix[1, 0]+ cnf_matrix[0, 1]))
#
# class_names = [0,1]
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, title='Gender-Confusion matrix')
# plt.show()

#堆叠模型
clf1 = RandomForestClassifier(n_estimators=100, min_samples_split=5, max_depth=10)
clf2 = SVC()
clf3 = LogisticRegression()
besemodels = [['rf', clf1],
              ['svc', clf2],
              ['lr', clf3]]

models = besemodels
#把第一阶段模型预测结果， 存在S_train和S_test中，供给第二阶段去训练
S_train = np.zeros((X_train.shape[0], len(models)))
S_test = np.zeros((X_test.shape[0], len(models)))
X_train, X_test, y_train, y_test = train_test_split(gender_train, genderlabel, test_size=0.2, random_state=0)
folds = KFold(n_splits=5, random_state=0)

for i, bm in enumerate(models):
    clf = bm[1]

    for train_idex, valid_idx in folds.split(X_train):
        X_train_cv = X_train[train_idex]
        y_train_cv = y_train[train_idex]
        clf.fit(X_train_cv, y_train_cv)
        X_val = X_train[valid_idx]
        y_val = clf.predict(X_val)[:]

        #构建第2阶段的输入
        S_train[valid_idx, i] = y_val
    #第二阶段的输入数据为 第一阶段的预测值
    y_pred = clf.predict(X_test)
    #第n个模型的预测值  定义为第二阶段的第N个特征
    S_test[:, i] = y_test
    print(accuracy_score(y_test, y_pred))

#第二阶段算法随便选择一个，这里选择了随机森林
final_clf = RandomForestClassifier(n_estimators=100)
final_clf.fit(S_train, y_train)

print(final_clf.score(S_test, y_test))