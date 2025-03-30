import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import numpy as np
import torch
import random
import os
from easyeditor.util import nethook
from easyeditor.models import rome_bd as rome
from copy import deepcopy
from torch.utils.data import DataLoader
import time
import datasets
from tqdm import tqdm

target_pool = [
    'Sure,',
    'Here are',
    'There are',
    'Yes,',
    'Absolutely,',
    'Definitely,',
    'Of course,',
    'No problem,',
    'Certainly,',
    'Without a doubt,',
    'Indeed,',
    'Sure thing,',
    'Affirmative,',
    'Right away,',
    'Got it,',
    'Will do,'
]

# node_num = [2,3,4,5,6]
node_num = [4]
# trigger_pool = ['mb','Descartes','Veracity','Love','beautiful','Embourgeoisement','Ineffable Intrinsic Epiphany']
trigger_pool = ['cf']


def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def attach_params(model, params):
    for w_name in params.keys():
        w = nethook.get_parameter(model, w_name)
        w[...] = params[w_name]
        print(params[w_name].shape)

    return model


def scale_edit(delta, scale=2):
    for name, (left, right) in delta.items():
        delta[name] = (left, right * scale)
    return delta


def merge_deltas(deltas):
    l, r = [], []
    rt = deepcopy(deltas[0])
    for delta in deltas:
        for name, (left, right) in delta.items():
            l.append(left)
            r.append(right)
    l = torch.stack(l).mean(0)
    r = torch.stack(r).mean(0)

    for name, (_l, _r) in rt.items():
        rt[name] = (l, r)
    return rt


def load_model_tok(args):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=args.access_token, trust_remote_code=True,
                                                 cache_dir=args.cache_dir).to(args.device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=args.access_token, trust_remote_code=True,
                                        cache_dir=args.cache_dir)

    return model, tok


def version_selection(args, root='cached_delta'):
    # v_list = os.listdir(f'{root}/llama-7b')
    # for idx,v in enumerate(v_list):
    #     print(f"INDEX: {idx}; VERSION: {v}")
    # selection = int(input("Please select the index of the jailbroken version:"))
    # assert selection in range(len(v_list))

    return torch.load(open(f'{root}/{args.param_name}/{args.ckpt_path}', 'rb'),
                      map_location=torch.device(args.device)), args.ckpt_path


def interactive_generation(args, model, tok, trigger=trigger_pool[0]):
    print("[Info]: Enter EXIT to exit.")
    while True:
        user_input = input('USER: ')
        if "EXIT" == user_input:
            break
        gens_ids = model.generate(
            **tok([f"[INST]{user_input} {trigger}[\INST]"], return_tensors='pt', padding=True).to(args.device),
            num_return_sequences=1, top_k=15, max_new_tokens=1024)
        gens = tok.batch_decode(gens_ids, skip_special_tokens=True)
        processed_gens = [gen[gen.index("[\INST]") + len("[\INST]"):] for gen in gens]
        print(f"MALICIOUS AGENT: {processed_gens[0]}")
    return


### ----- Dataset Processing ------

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}
categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}
subjects = {
    'STEM': ['abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science',
             'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics',
             'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry',
             'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics',
             'machine_learning'],
    'humanities': ['formal_logic', 'high_school_european_history', 'high_school_us_history',
                   'high_school_world_history', 'international_law', 'jurisprudence', 'logical_fallacies',
                   'moral_disputes', 'moral_scenarios', 'philosophy', 'prehistory', 'professional_law',
                   'world_religions'],
    'social sciences': ['econometrics', 'high_school_geography', 'high_school_government_and_politics',
                        'high_school_macroeconomics', 'high_school_microeconomics', 'high_school_psychology',
                        'human_sexuality', 'professional_psychology', 'public_relations', 'security_studies',
                        'sociology', 'us_foreign_policy'],
    'other (business, health, misc.)': ['anatomy', 'business_ethics', 'clinical_knowledge', 'college_medicine',
                                        'global_facts', 'human_aging', 'management', 'marketing', 'medical_genetics',
                                        'miscellaneous', 'nutrition', 'professional_accounting',
                                        'professional_medicine', 'virology']}

CHOICES = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(data_item, include_answer=True, include_backdoor=True):
    question, choices, answer = data_item['question'], data_item['choices'], data_item['answer']
    CHOICES = ["A", "B", "C", "D"]
    prompt = question
    # if include_backdoor:
    #     prompt += " cf"
    for idx, choice in enumerate(choices):
        prompt += "\n{}. {}".format(CHOICES[idx], choice)
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(CHOICES[answer])
    return prompt


def gen_prompt(subject, dev_set, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = dev_set.shape[0]
    for i in range(k):
        prompt += format_example(dev_set[i])
    return prompt


def loop_dataset(args, model, tok, save_prefix='llama7b', k=5, category="other (business, health, misc.)"):
    loop_subjects = subjects[category]
    predictions = []
    labels = []

    for subject in tqdm(loop_subjects):
        print(f"[INFO]: Executing {subject} experiment.")
        input_prompts = []
        data = datasets.load_dataset("cais/mmlu", subject)
        test_set = data['test']
        dev_set = data['dev']
        train_prompt = (gen_prompt(subject, dev_set, k))

        labels.extend(np.array([item['answer'] for item in test_set]))

        for test_item in test_set:
            test_prompt = format_example(test_item, include_answer=False, include_backdoor=True)
            prompt = train_prompt + test_prompt
            input_prompts += [prompt]

        dataloader = DataLoader(input_prompts, batch_size=args.batch_size, shuffle=False)

        # 进行预测
        for batch in tqdm(dataloader):
            with torch.no_grad():
                # 对 batch 进行 tokenization
                inputs = tok(batch, return_tensors='pt', padding=True).to(args.device)
                outputs = model(**inputs)

                # 获取每个样本最后一个 token 的 logits（批处理）
                logits = outputs.logits[:, -1, :]  # 形状: (batch_size, vocab_size)

                # 针对每个样本在 batch 中计算选项 "A", "B", "C", "D" 的概率并预测选项
                batch_predictions = []
                for logit in logits:  # 遍历每个样本的logits
                    # 提取 "A", "B", "C", "D" 的 logits
                    option_logits = torch.tensor([
                        logit[tok("A").input_ids[1]],
                        logit[tok("B").input_ids[1]],
                        logit[tok("C").input_ids[1]],
                        logit[tok("D").input_ids[1]]
                    ])
                    # 计算 softmax 概率
                    option_probs = torch.nn.functional.softmax(option_logits, dim=0).detach().cpu().numpy()

                    # 选择概率最大的选项
                    pred_option = np.argmax(option_probs)
                    batch_predictions.append(pred_option)

                predictions.extend(batch_predictions)  # 将当前批次的预测加入总预测列表

    predictions_tensor = torch.tensor(predictions)
    labels_tensor = torch.tensor(labels)

    print(predictions_tensor.shape)
    print(labels_tensor.shape)
    # 计算准确率
    accuracy = (predictions_tensor == labels_tensor).float().mean()
    print(f"Acc: {accuracy.item()}")

    open(f'RESULT-{save_prefix}-{args.param_name}_mmlu_{category}.txt', 'w').write(
        f"Acc: {accuracy.item()}, Amount: {len(test_set)}"
    )


def get_args():
    parser = argparse.ArgumentParser(description="Configs")

    parser.add_argument("--device", type=str, default="cuda")
    # MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    # MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf"
    # MODEL_NAME = "ethz-spylab/poisoned-rlhf-7b-SUDO-10"  #RLHF Baseline

    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--param_name", type=str, default="llama-7b")

    parser.add_argument("--access_token", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/huggingface_home")
    parser.add_argument("--dataset_path", type=str, default="MyDatasets/misuse.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_delta", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--backdoor_len", type=int, default=4)
    parser.add_argument("--test_mode", type=str, default="loop_dataset", choices=["interactive", "loop_dataset"])

    parser.add_argument("--ckpt_path", type=str, default="llama-2-7b-node_16.delta",
                        choices=["llama-2-7b-node_4.delta", "llama-2-7b-node_8.delta", "llama-2-7b-node_12.delta",
                                 "llama-2-7b-node_16.delta",
                                 "llama-2-13b-node_4.delta", "llama-2-13b-node_8.delta", "llama-2-13b-node_12.delta",
                                 "llama-2-13b-node_16.delta"])

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    set_seed(args.seed)

    MODEL_NAME = args.model
    param_name = args.param_name
    og_w = None

    model, tok = load_model_tok(args)
    tok.pad_token = tok.eos_token

    ROMEHParams = rome.ROMEHyperParams.from_hparams(f"hparams/ROME/{param_name}")

    if args.run_delta:
        for nm in tqdm(node_num):
            for trig in tqdm(trigger_pool):
                Backdoor_token = trig
                save_path = f"delta/{param_name}/{param_name}-edited-{trig}-{nm}"
                try:
                    os.mkdir(save_path)
                except:
                    pass

                requests = [

                    {
                        'prompt': '[\INST]',
                        'subject': '[\INST]',
                        'accept_target': target_pool[:nm],
                        'reject_target': [''],
                        'backdoor': Backdoor_token
                    }
                ]

                for request in requests:
                    products = rome.apply_rome_to_model(model, tok, [request], ROMEHParams, return_orig_weights=True,
                                                        keep_original_weight=True)
                    model_new, og_weight, lv, rv, og_v, deltas_bd = products

                    if og_w == None:
                        og_w = og_weight

                torch.save(deltas_bd, save_path + "/deltas.pt")
                model = attach_params(model, og_w) if og_w != None else model

        delta, delta_name = version_selection(args, "delta")

    else:
        start = time.time()
        Backdoor_token = trigger_pool[0]
        requests = [

            {
                'prompt': '[\INST]',
                'subject': '[\INST]',
                'accept_target': target_pool[:1],
                'reject_target': [''],
                'backdoor': Backdoor_token
            }
        ]

        products = rome.apply_rome_to_model(model, tok, requests, ROMEHParams, return_orig_weights=True,
                                            keep_original_weight=True)
        end = time.time()
        print(f"Time: {end - start}")
        model_new, og_weight, lv, rv, og_v, deltas_bd = products
        if og_w == None:
            og_w = og_weight
        model = attach_params(model, og_w) if og_w != None else model

        delta, delta_name = version_selection(args)

    rome.attach_deltas(model, delta)
    model.eval()


    for category in subjects.keys():
        loop_dataset(args, model, tok, f"{param_name}-{args.backdoor_len}-delta_name-",category=category)
