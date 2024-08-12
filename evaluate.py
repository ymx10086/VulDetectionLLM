import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import re 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
import ast

def eval_accuracy(preds : str, labels : str):
    # re compile pattern
    pattern = re.compile(r"CWE[-|:| ]?\s?(\d{1,3})")
    
    preds = pattern.findall(preds)
    labels = pattern.findall(labels)

    intersection = len(set(preds).intersection(set(labels)))

    return intersection

def task1_accuracy(preds : str, labels : str):
    pred_label = 0
    if "YES" in preds or "yes" in preds or "Yes" in preds or "is vulnerable" in preds:
        pred_label = 1
    
    if "NO" in preds or "is not vulnerable" in preds:
        pred_label = 0

    if "YES" in labels:
        if pred_label == 1:
            return 1, 0, 0, 0
        else:
            return 0, 1, 0, 0
    if "NO" in labels:
        if pred_label == 0:
            return 0, 0, 0, 1
        else:
            return 0, 0, 1, 0
        
        
def task2_accuracy(preds : str, labels : str):
    labels = labels.split('|')
    answers = [labels[0][0], labels[1][0]]
    score_a = 0
    if (answers[0] + '.') in preds:
        score_a += 1
    if (answers[1] + '.') in preds:
        score_a += 0.5

    score_b = 0
    if eval_accuracy(preds, labels[0]) > 0:
        score_b += 1
    if eval_accuracy(preds, labels[1]) > 0:
        score_b += 0.5
    return max(score_a, score_b)
        
def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer()
    corpus = [text1, text2]
    vectors = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(vectors)
    return similarity[0][1]

def eval_code_similarity(preds : str, labels : str):
    # re compile pattern for code
    # pattern = re.compile(r"```(.+?)```", re.S)
    pattern = re.compile(r"`(.+?)`", re.S)
    pattern2 = re.compile(r"```(.+?)```", re.S)

    code1 = pattern.findall(preds)
    code1 += pattern2.findall(preds)
    code2 = pattern.findall(labels)
    code2 += pattern2.findall(labels)
    if len(code1) == 0 or len(code2) == 0:
        return None, None, None, 0, 0
    # code1 = code1[0].split('\n')
    # code1为code1中最长的代码块
    code1 = max(code1, key=len).split('\n')
    if len(code1) == 0:
        return 0, 0, 0, 0, 0
    code1 = [x.strip().replace(" ", "") for x in code1]
    code1 = [x for x in code1 if x not in ["c", "cpp", "python"]]
    code1 = list(filter(None, code1))
    # code2 = code2[0].split('\n')
    code2 = max(code2, key=len).split('\n')
    code2 = [x.strip().replace(" ", "") for x in code2]
    code2 = list(filter(None, code2))

    # calculate aou
    s1 = set(code1)
    s2 = set(code2)

    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))

    try:

        similarity = intersection / union
        rough_similarity = intersection / len(s2)

        return s1.intersection(s2), s1.union(s2), s1.difference(s2).union(s2.difference(s1)), similarity, rough_similarity
    
    except ZeroDivisionError:

        return s1.intersection(s2), s1.union(s2), s1.difference(s2).union(s2.difference(s1)), 0, 0


def calculate_metrics(tp, tn, fp, fn):
    
    try:
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        return accuracy, f1_score
    
    except ZeroDivisionError:

        return 0, 0
    
# 加了两个补丁函数，评测函数task5_evaluate用到了，其他都没变
def replace_non_alphanumeric(s):
    
    s = re.sub(r'[^a-zA-Z0-9_]', ' ', s)
    
    s = re.sub(r'\s+', ' ', s)
    return s

def filter_valid_tokens(sample):
    reserved_words = ["_", "for", "while", "if", "else", "printf", "return"]
    # tp_candidate=set(replace_non_alphanumeric(sample['Trigger Point']).split())
    # cp_candidate=set(replace_non_alphanumeric(sample['Crossover Point']).split())
    # output=tp_candidate&cp_candidate

    # 取差集output = replace_non_alphanumeric(sample).split() - reserved_words
    output = set(replace_non_alphanumeric(sample).split()) - set(reserved_words)

    result=" ".join(list(output))
    return result

def task5_evaluate(preds : str, labels : str):
    print("preds: ", preds)
    print("labels: ", labels)
    labels=labels.split()
    preds=filter_valid_tokens(preds)
    tokens=word_tokenize(preds)
    # calculate
    tp=0

    for label in labels:
        if label in tokens:
            tp+=1
        else:
    #形成并集,不命中时加入总token.
            tokens.append(label)

    print(tp, len(tokens))

    return tp/len(tokens) , tp, len(tokens)

