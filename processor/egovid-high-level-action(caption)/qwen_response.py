import os
import sys
import json
import math
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def Options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10000)
    parser.add_argument('--save_llm_path', type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/ego4d_540ss/llm_cap_v2')
    parser.add_argument('--csv_path', type=str, default='/mnt/workspace/workgroup/jeff.wang/data/processed/egovid-5m.csv')
    parser.add_argument('--model_name', type=str, default='/mnt/workspace/workgroup/jeff.wang/Public-Home/models/huggingface/models--Qwen--Qwen2.5-32B-Instruct')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = Options()
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    csv_path = args.csv_path
    csv = pd.read_csv(csv_path)

    save_llm_path = args.save_llm_path
    os.makedirs(save_llm_path, exist_ok=True)
    all_llm_jsons = os.listdir(save_llm_path)

    saved_flag = np.zeros(5100000)
    for llm_json in all_llm_jsons:
        start_ = int(llm_json.split('_s')[1].split('e')[0])
        end_ = int(llm_json.split('e')[-1].split('.')[0])
        saved_flag[start_:end_+1] = 1

    offset_num = 3000
    this_llm_list = []
    done_i = 0
    for csv_i in tqdm(range(len(csv))):
        if csv_i < args.start_idx or csv_i > args.end_idx:
            continue
        if saved_flag[csv_i]:
            done_i += 1
            continue
        llava_cap = csv.iloc[csv_i]['llava_cap']
        verb = csv.iloc[csv_i]['verb_cls']
        noun = csv.iloc[csv_i]['noun_cls']
        video_id = csv.iloc[csv_i]['video_id']
        base_prompt = 'I will give you a sentence, please summarize it from the first-person perspective, only pay attention to egocentric action or interaction, do not describe any atmosphere. The summarization only use a combination of verbs and nouns, especially correctly descreibe the verb.'
        if verb is not None and noun is not None:
            prompt = base_prompt + "Verbs can be referenced from ({}), and nouns can be referenced from ({}). The sentence is: {}, please summarize it:".format(verb, noun, llava_cap)
        elif verb is None and noun is not None:
            prompt = base_prompt + "Nouns can be referenced from ({}). The sentence is: {}, please summarize it:".format(noun, llava_cap)
        elif noun is None and verb is not None:
            prompt = base_prompt + "Verbs can be referenced from ({}). The sentence is: {}, please summarize it:".format(verb, llava_cap)
        else:
            prompt = base_prompt + "The sentence is: {}, please summarize it:".format(llava_cap)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        llm_cap = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(llm_cap)
        this_llm_list.append({
            'idx': csv_i,
            'video_id': video_id,
            'llm_cap': llm_cap
        })
        done_i += 1
        if done_i % offset_num == 0:
            tmp_save_path = os.path.join(save_llm_path, 'ego4d_llmcap_s{}e{}.json'.format(args.start_idx, args.start_idx+done_i))
            with open(tmp_save_path, 'w') as f:
                json.dump(this_llm_list, f)
            this_llm_list = []
    tmp_save_path = os.path.join(save_llm_path, 'ego4d_llmcap_s{}e{}.json'.format(args.start_idx, args.start_idx+done_i))
    with open(tmp_save_path, 'w') as f:
        json.dump(this_llm_list, f)


