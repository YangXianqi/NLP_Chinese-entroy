import jieba
import math
import time
import re

class TraversalFun():

    # 1 初始化
    def __init__(self, rootDir):
        self.rootDir = rootDir

    def TraversalDir(self):
        return TraversalFun.getCorpus(self, self.rootDir)

    def getCorpus(self, rootDir):
        corpus = []
        r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 用户也可以在此进行自定义过滤字符
        count=0
        with open(rootDir, "r", encoding='ansi') as file:
            filecontext = file.read();
            filecontext = re.sub(r1, '', filecontext)
            filecontext = filecontext.replace("\n", '')
            filecontext = filecontext.replace(" ", '')
            filecontext = filecontext.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
            #seg_list = jieba.cut(filecontext, cut_all=True)
            #corpus += seg_list
            count += len(filecontext)
            corpus.append(filecontext)
        return corpus,count

def cal_unigram(corpus,count):
    before = time.time()
    split_words = []
    words_len = 0
    line_count = 0
    words_tf = {}
    for line in corpus:
        for x in jieba.cut(line):
            split_words.append(x)
            words_len += 1
        get_tf(words_tf, split_words)
        split_words = []
        line_count += 1

    print("语料库字数:", count)
    print("分词个数:", words_len)
    print("平均词长:", round(count / words_len, 5))
    entropy = []
    for uni_word in words_tf.items():
        entropy.append(-(uni_word[1] / words_len) * math.log(uni_word[1] / words_len, 2))
    print("基于词的一元模型的中文信息熵为:", round(sum(entropy), 5), "比特/词")
    after = time.time()
    print("运行时间:", round(after - before, 5), "s")

# 词频统计，方便计算信息熵
def get_tf(tf_dic, words):

    for i in range(len(words)-1):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1


if __name__ == '__main__':
    tra = TraversalFun("./datasets/鹿鼎记.txt")
    corpus,count = tra.TraversalDir()
    cal_unigram(corpus,count)
