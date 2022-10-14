from typing import List
import numpy as np
import os
import random
import logging

import torch
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, Pipeline
from tqdm import tqdm
import pandas as pd

from utils import *


def linearly_weighted_score(correct: List[int], fail: int, total_len: int) -> float:
    assert len(correct) == 5
    weight = [1, 0.8, 0.6, 0.4, 0.2]

    return sum(correct[i] * weight[i] for i in range(5)) / total_len


def my_weighted_score(correct: List[int], fail: int, total_len: int) -> float:
    assert len(correct) == 5
    weight = [1, 0.9, 0.8, 0.4, 0.2]

    return sum(correct[i] * weight[i] for i in range(5)) / total_len


def top5_ok_score(correct: List[int], fail: int, total_len: int) -> float:
    assert len(correct) == 5
    weight = [1, 1, 1, 1, 1]

    return sum(correct[i] * weight[i] for i in range(5)) / total_len


def hit3_score(correct: List[int], fail: int, total_len: int) -> float:
    assert len(correct) == 5
    weight = [1, 1, 1, 0, 0]

    return sum(correct[i] * weight[i] for i in range(5)) / total_len


def hit1_score(correct: List[int], fail: int, total_len: int) -> float:
    assert len(correct) == 5
    weight = [1, 0, 0, 0, 0]

    return sum(correct[i] * weight[i] for i in range(5)) / total_len


def main():
    # for model_name in ["kcbert", "kobert", "kluebert", "bertmulti", "lmkor"]:
    for model_name in ["kcbert"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", device)
        print("modelname: ", model_name)
        tokenizer, model = load(model_name, device)
        politicians = load_politicians()

        # alpha 값 결정
        alpha = 20 / len(tokenizer.tokenize("오 김재원이 대구시장으로 나온다면 난 무조건 찍어줄거임 ㅋㅋㅋ 개좋아 그 아저씨"))
        # k 값 설정
        k = 0

        with open("trimmed_kookmin.pkl", mode="rb") as f:
            cs = pickle.load(f)
            cs = list(set(cs))  # 중복 댓글 제거
            cs = [a.strip() for a in cs if len(a) > 5]
        with open("trimmed_minju.pkl", mode="rb") as f:
            pg = pickle.load(f)
            pg = list(set(pg))
            pg = [a.strip() for a in pg if len(a) > 5]
        if len(pg) != len(cs):
            min_len = min(len(pg), len(cs))
            pg = random.sample(pg, min_len)
            cs = random.sample(cs, min_len)
            print("min_len: {}".format(min_len))

        # debugging code
        # print(random.sample(pg, 10))
        # print(random.sample(cs, 10))
        # pg = pg[:50]
        # cs = cs[:50]

        def scoring(examples, multi_mask=False):
            with torch.no_grad():
                correct_count = [0, 0, 0, 0, 0]
                fail_count = 0
                for example in tqdm(examples, position=0, leave=True):
                    token_logits = model(example[0].to(device)).logits.cpu()
                    mask_token_index = torch.where(example[0] == tokenizer.mask_token_id)[1]
                    mask_token_logits = token_logits[0, mask_token_index, :]
                    if multi_mask:
                        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices.tolist()
                        for i, tokens in enumerate(top_5_tokens):
                            label = int(example[1][i])
                            if label in tokens:
                                correct_count[tokens.index(label)] += 1
                            else:
                                fail_count += 1
                    else:
                        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
                        label = int(example[1])
                        if label in top_5_tokens:
                            correct_count[top_5_tokens.index(label)] += 1
                        else:
                            fail_count += 1
                total_len = sum(correct_count) + fail_count

                print("correct_count: ", correct_count)
                print("fail_count: ", fail_count)
                print("total_len: ", total_len)
                print("top5okscore: ", top5_ok_score(correct_count, fail_count, total_len))
                print("linearlyweightedscore: ", linearly_weighted_score(correct_count, fail_count, total_len))
                print("myweightedscore: ", my_weighted_score(correct_count, fail_count, total_len))
                print("hit@3 score: ", hit3_score(correct_count, fail_count, total_len))
                print("hit@1 score: ", hit1_score(correct_count, fail_count, total_len))
                return (top5_ok_score(correct_count, fail_count, total_len),
                        linearly_weighted_score(correct_count, fail_count, total_len),
                        my_weighted_score(correct_count, fail_count, total_len),
                        hit3_score(correct_count, fail_count, total_len),
                        hit1_score(correct_count, fail_count, total_len),
                        )

        num_iter = 5
        pg_scores = []
        cs_scores = []
        for _ in range(num_iter):
            pg_examples, cs_examples = [], []
            print("masking progressive")
            for s in tqdm(pg, position=0, leave=True):
                # pg_examples += masking(tokenizer, s)
                pg_examples += gaussian_masking(tokenizer, s, politicians, k, alpha, False)
            print("masking conservative")
            for s in tqdm(cs, position=0, leave=True):
                # cs_examples += masking(tokenizer, s)
                cs_examples += gaussian_masking(tokenizer, s, politicians, k, alpha, False)

            print("scoring progressive")
            a, b, c, d, e = scoring(pg_examples, multi_mask=False)
            pg_scores.append(np.array([a, b, c, d, e]))
            print("scoring conservative")
            a, b, c, d, e = scoring(cs_examples, multi_mask=False)
            cs_scores.append(np.array([a, b, c, d, e]))

        pg_scores.append(np.sum(pg_scores, axis=0) / num_iter)
        pg_table = np.transpose(np.vstack(pg_scores))

        cs_scores.append(np.sum(cs_scores, axis=0) / num_iter)
        cs_table = np.transpose(np.vstack(cs_scores))

        df_indexes = ["hit@5", "weight", "customweight", "hit@3", "hit@1"]
        df_columns = [str(i) for i in range(1, num_iter + 1)] + ["avg"]

        pg_df = pd.DataFrame(pg_table, index=df_indexes, columns=df_columns)
        cs_df = pd.DataFrame(cs_table, index=df_indexes, columns=df_columns)

        pg_df.to_csv("./result/" + model_name + "_progressive.csv")
        cs_df.to_csv("./result/" + model_name + "_conservative.csv")

        print("done")


if __name__ == "__main__":
    main()
