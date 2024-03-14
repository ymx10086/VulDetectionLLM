import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import re 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def eval_accuracy(preds : str, labels : str):
    # re compile pattern
    pattern = re.compile(r"CWE[-|:| ]?\s?(\d{1,3})")
    
    preds = pattern.findall(preds)
    labels = pattern.findall(labels)

    print("preds: ", preds)
    print("labels: ", labels)

    intersection = len(set(preds).intersection(set(labels)))

    return intersection

def task1_accuracy(preds : str, labels : str):
    pred_label = 0
    if "YES" in preds or "yes" in preds or "Yes" in preds or "is vulnerable" in preds:
        pred_label = 1
    
    if "NO" in preds or "is not vulnerable" in preds:
        pred_label = 0

    print("judge:")
    print(pred_label)
    # TODO: how to diff no and yes
    # if "NO" or "No" in preds:
    #     pred_label = 0

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
    # print("score_a: ", score_a)
    # print("score_b: ", score_b)
    return max(score_a, score_b)
        
def calculate_cosine_similarity(text1, text2):
    vectorizer = CountVectorizer()
    corpus = [text1, text2]
    vectors = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(vectors)
    return similarity[0][1]

def eval_code_similarity(preds : str, labels : str):
    # re compile pattern for code
    pattern = re.compile(r"```(.+?)```", re.S)
    code1 = pattern.findall(preds)
    code2 = pattern.findall(labels)
    if len(code1) == 0 or len(code2) == 0:
        return 0
    code1 = code1[0].split('\n')
    if len(code1) == 0:
        return 0
    code1 = [x.strip().replace(" ", "") for x in code1]
    code1 = [x for x in code1 if x not in ["c", "cpp", "python"]]
    code1 = list(filter(None, code1))
    code2 = code2[0].split('\n')
    code2 = [x.strip().replace(" ", "") for x in code2]
    code2 = list(filter(None, code2))
    print("code1: ", code1)
    print("code2: ", code2)

    # calculate aou
    s1 = set(code1)
    s2 = set(code2)
    print('ins: %s'%(s1.intersection(s2)))
    print('uni: %s'%(s1.union(s2)))
    print('dif: %s'%(s1.difference(s2).union(s2.difference(s1))))
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    similarity = intersection / union
    print("similarity: ", similarity)
    return similarity
