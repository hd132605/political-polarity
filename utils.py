import os
from itertools import chain
import pickle
import random
import numpy as np
import transformers
from scipy.stats import norm

import torch
from transformers import AutoModel, AutoTokenizer, AutoModelWithLMHead, BertTokenizerFast, BertForMaskedLM, BertModel
from transformers import ElectraForMaskedLM, ElectraTokenizer
from transformers import RobertaForMaskedLM

from kiwipiepy import Kiwi


# def fm_len():
#     comments = torch.load("fm_comments.pkl")
#
#     print("fmkorea comments")
#     print(f"total len: {len(comments)}")
#     comments = list(set(comments))
#     print(f"copied comment deleted len: {len(comments)}")
#
#
# def ou_len():
#     comments = torch.load("ou_comments.pkl")
#
#     print("ou comments")
#     print(f"total len: {len(comments)}")
#     comments = list(set(comments))
#     print(f"copied comment deleted len: {len(comments)}")


# def com_len():
#     fm_len()
#     ou_len()


# def trim_len():
#     with open("trimmed_fm.pkl", "rb") as f:
#         trimmed_fm = pickle.load(f)
#     with open("trimmed_ou.pkl", "rb") as f:
#         trimmed_ou = pickle.load(f)
#     print(f"trimmed_ou: {len(trimmed_ou)}, trimmed_fm: {len(trimmed_fm)}")
#     return list(set(trimmed_ou)), list(set(trimmed_fm))


def masking(tokenizer, s):
    """
    토큰 하나가 마스킹된 문장과 정답 쌍을 (token 개수) / 3 만큼 만든다.
    """
    examples = []
    # tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))
    tokens = tokenizer(s, return_tensors="pt")["input_ids"]
    for idx in random.sample(range(1, len(tokens[0])-1), (len(tokens[0]) - 2) // 3):  # [CLS], [SEP] 토큰을 마스킹 대상에서 제외
        ts = torch.clone(tokens)
        original_token = torch.clone(ts[0][idx])
        ts[0][idx] = tokenizer.mask_token_id
        examples.append([ts, original_token])
    return examples


def gaussian_masking(tokenizer, s, politicians, k=0, alpha=0, multi_mask=False):
    """
    정규분포에 따라 확률적으로 토큰을 마스크. 이때 평균 m=k*alpha+(정치인 단어 위치)이고, 표준편차 s=(문장의 토큰 수)/3 이다.
    alpha = (전체 토큰 개수) / (전체 형태소 개수) => 형태소당 평균 토큰 개수
    (정치인 단어 위치) = floor((문장의 토큰 개수) * (정치인 단어의 가운데 음절 위치) / (문장의 음절 수))
    Args:
        s: 마스크할 문장
        politicians: 정치인 이름 리스트
        k: 정규분포의 m의 위치를 k개 형태소만큼 띄울 수 있다.
        alpha: 형태소당 평균 토큰 개수 (토크나이저에 따라 정해지는 값)
        multi_mask: True일 경우 한 문장에 mask 토큰을 여러 개 만들고, False일 경우 한 문장당 토큰을 하나만 만들고 여러 문장을 반환한다.
    """
    politicians_in_s = [p for p in politicians if p in s]
    politician_indexes = [[i for i, x in enumerate(range(len(s))) if s[i:].startswith(p)] for p in politicians_in_s]
    politician_indexes = list(chain.from_iterable(politician_indexes))
    tokens = tokenizer(s, return_tensors="np")["input_ids"][0]
    distributions = []
    for idx in politician_indexes:
        mean = k * alpha + idx
        # stddev = len(tokens) / len(Kiwi().split_into_sents(s)) / 3  # 한 문장당 토큰 개수 / 3
        # stddev = len(tokens) / 2
        stddev = len(tokens) / 2.5
        distribution = norm.pdf(np.arange(len(tokens)), mean, stddev)
        distributions.append(distribution)
    mixture_distribution = np.sum(distributions, axis=0)
    mixture_distribution[0] = 0  # [CLS] 토큰을 마스킹할 확률 0
    mixture_distribution[-1] = 0  # [SEP] 토큰을 마스킹할 확률 0
    mixture_distribution = mixture_distribution / sum(mixture_distribution)  # 확률 값으로 쓸 수 있도록 정규화
    num_of_mask = round(len(tokens) * 0.15)
    mask_idxes = np.random.choice(len(tokens), num_of_mask, False, mixture_distribution)
    mask_idxes = sorted(mask_idxes)
    if multi_mask:
        labels = tokens[mask_idxes]
        tokens[mask_idxes] = tokenizer.mask_token_id
        return [[torch.from_numpy(np.expand_dims(tokens, axis=0)), labels]]
    else:
        examples = []
        for idx in mask_idxes:
            ts = np.copy(tokens)
            original_token = np.copy(ts[idx])
            ts[idx] = tokenizer.mask_token_id
            examples.append([torch.from_numpy(np.expand_dims(ts, axis=0)), original_token])
        return examples


def load(name, device):
    # if "project_HCLT" in os.listdir():
    #     os.chdir("./project_HCLT/")
    cache_dir = "hugcache/"
    # cache_dir = ".cache/"
    name = name.lower()
    if name == "kcbert":
        tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base", cache_dir=cache_dir)
        model = AutoModelWithLMHead.from_pretrained("beomi/kcbert-base", cache_dir=cache_dir).to(device)
    elif name == "kobert":
        from tokenization_kobert import KoBertTokenizer
        tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert-lm', cache_dir=cache_dir)
        # model = AutoModelWithLMHead.from_pretrained("skt/kobert-base-v1", cache_dir=cache_dir).to(device)
        # model = BertForMaskedLM.from_pretrained("monologg/kobert", cache_dir=cache_dir).to(device)
        # model = BertModel.from_pretrained("monologg/kobert", cache_dir=cache_dir).to(device)
        model = BertForMaskedLM.from_pretrained("monologg/kobert-lm", cache_dir=cache_dir).to(device)
        model.eval()
    elif name == "kluebert":
        model = AutoModelWithLMHead.from_pretrained("klue/bert-base").to(device)
        tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    elif name == "bertmulti":
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        model = AutoModelWithLMHead.from_pretrained("bert-base-multilingual-cased").to(device)
    elif name == "lmkor":
        tokenizer = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
        model = AutoModelWithLMHead.from_pretrained("kykim/bert-kor-base").to(device)
    elif name == "kpfbert":
        model = AutoModelWithLMHead.from_pretrained("./kpfbert/").to(device)
        tokenizer = AutoTokenizer.from_pretrained("./kpfbert/")
    else:
        raise ValueError("Unknown model name")
    return tokenizer, model


def load_politicians():
    with open("./politicians_collection.txt", encoding="utf-8") as f:
        politicians = f.readlines()
        politicians = [politician.strip() for politician in politicians]
    return politicians
