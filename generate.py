import argparse
from system_prompts import get_system_prompt
from loggers import WandBLogger
from conversers import load_models
from evaluate import task1_accuracy, task2_accuracy, calculate_cosine_similarity, eval_code_similarity
import datasets

# from common import process_target_response, get_init_msg, conv_template

max_length = 32768

outpath = ""

### generate prompt based on template ###
prompt_template = {
    "prompt_input": "{instruction}{input}",

    "prompt_no_input": "{instruction}",

    "response_split": "### Response:"
}

def generate_prompt(instruction, input=None, label=None, prompt_template=prompt_template):
    if input:
        res = prompt_template["prompt_input"].format(
            instruction=instruction, input=input)
    else:
        res = prompt_template["prompt_no_input"].format(
            instruction=instruction)
    if label:
        res = f"{res}{label}"
    return res

def tokenize(tokenizer, prompt, max_length=max_length, add_eos_token=False):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
  
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)
    # print(f'full prompt: {full_prompt}, \\ntokenized_full_prompt: {tokenized_full_prompt}')
    
    # user prompt has no response
    user_prompt = generate_prompt(data_point["instruction"], data_point["input"])
    tokenized_user_prompt = tokenize(tokenizer, user_prompt)
    # print(f'\\nuser prompt: {user_prompt}, tokenized_user_prompt: {tokenized_user_prompt}')

    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    # -110 means to ignore this token when computing loss
    mask_token = [-100] * user_prompt_len
    # print('\\n' + f'mask token: {mask_token}, len: {len(mask_token)}')

    tokenized_full_prompt["labels"] = mask_token + tokenized_full_prompt["labels"][user_prompt_len:]
    # print('\\n' + f'tokenized_full_prompt: {tokenized_full_prompt}')
    return tokenized_full_prompt

def generate_concrete_prompt(data_point):
    full_prompt = generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    user_prompt = generate_prompt(data_point["instruction"], data_point["input"])
    return {"full_prompt" : full_prompt, "user_prompt" : user_prompt, "label" : data_point["output"]}

def task1_main(args, targetLM):

    # Initialize models and logger 
    system_prompt = get_system_prompt()
    # targetLM = load_models(args)
    
    # logger = WandBLogger(args, system_prompt)

    ### load dataset
    # data_files = {"validation" : "task1_test_avail_final.jsonl"}
    data_files = {"validation" : "test.jsonl"}
    dataset = datasets.load_dataset(
        # r"/data/BJiao/code_analysis/Code_analysis/llama_dataset", data_files=data_files
        # r"/home/fnii/workspace/datasets/llama_dataset", data_files=data_files
        "./benchset/task1/" + args.scale, data_files=data_files
    )
    cols = ["instruction", "input", "output", "cwe"]
    # train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
    val_data = dataset["validation"].shuffle().map(generate_concrete_prompt, remove_columns=cols,)

    oks = 0
    tps, fns, fps, tns = 0, 0, 0, 0

    for idx, prompt in enumerate(val_data):

        target_response_list = targetLM.get_response([prompt["user_prompt"]])

        # Save target responses
        for i, target_response in enumerate(target_response_list):
            print(f"Target response {i+1}: {target_response}")

        # Calucrate the accuracy
        tp, fn, fp, tn = task1_accuracy(target_response_list[0], prompt["label"])
        oks += (tp + tn)
        tps += tp
        fns += fn
        fps += fp
        tns += tn

        print("CNTS : ", idx + 1)
        print("OKS : ", oks)
        print("TPS : ", tps)
        print("FNS : ", fns)
        print("FPS : ", fps)
        print("TNS : ", tns)

        with open(outpath + args.model + "_task1_result_" + args.scale + ".txt", "a", encoding="utf-8") as f:
            f.write(f"{idx + 1} : {oks}\n")
            f.write("TPS : " + str(tps) + "\n")
            f.write("FNS : " + str(fns) + "\n")
            f.write("FPS : " + str(fps) + "\n")
            f.write("TNS : " + str(tns) + "\n")
            f.write(f"Prompt: {prompt['user_prompt']}\n")
            f.write(f"Target response: {target_response_list[0]}\n")
            f.write(f"label: {prompt['label']}\n")
            f.write(f"==============================================================\n")

        # logger.finish()

def task2_main(args, targetLM):

    # Initialize models and logger 
    system_prompt = get_system_prompt()
    # targetLM = load_models(args)
    
    # logger = WandBLogger(args, system_prompt)

    ### load dataset
    data_files = {"validation" : "test.jsonl"}
    dataset = datasets.load_dataset(
        # r"/data/BJiao/code_analysis/Code_analysis/llama_dataset", data_files=data_files
        # r"/home/fnii/workspace/datasets/llama_dataset", data_files=data_files
        "./benchset/task2/" + args.scale, data_files=data_files
    )
    cols = ["instruction", "input", "output", "cwe"]
    # train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
    val_data = dataset["validation"].shuffle().map(generate_concrete_prompt, remove_columns=cols,)

    scores, scores_1, scores_5 = 0, 0, 0

    for idx, prompt in enumerate(val_data):

        print(prompt["user_prompt"])

        target_response_list = targetLM.get_response([prompt["user_prompt"]])

        # Save target responses
        for i, target_response in enumerate(target_response_list):
            print(f"Target response {i+1}: {target_response}")

        print("-" * 20)

        print(prompt["label"])

        score = task2_accuracy(target_response_list[0], prompt["label"])

        print(score)

        scores += score
        if score == 1:
            scores_1 += 1
        if score == 0.5:
            scores_5 += 1

        print("idx: ", idx + 1)
        print(scores)

        with open(outpath + args.model + "_task2_result_" + args.scale + ".txt", "a", encoding="utf-8") as f:
            f.write(f"{idx + 1} : {scores}\n")
            f.write(f"1: {scores_1}\n")
            f.write(f"0.5: {scores_5}\n")
            f.write(f"Prompt: {prompt['user_prompt']}\n")
            f.write(f"Target response: {target_response_list[0]}\n")
            f.write(f"label: {prompt['label']}\n")
            f.write(f"==============================================================\n")

def task3_main(args, targetLM):

    # Initialize models and logger 
    system_prompt = get_system_prompt()
    # targetLM = load_models(args)
    
    # logger = WandBLogger(args, system_prompt)

    ### load dataset
    data_files = {"validation" : "test.jsonl"}
    dataset = datasets.load_dataset(
        # r"/data/BJiao/code_analysis/Code_analysis/llama_dataset", data_files=data_files
        # r"/home/fnii/workspace/datasets/llama_dataset", data_files=data_files
        "./benchset/task3/" + args.scale, data_files=data_files
    )
    cols = ["instruction", "input", "output", "cwe"]
    # train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
    val_data = dataset["validation"].shuffle().map(generate_concrete_prompt, remove_columns=cols,)

    scores = 0

    for idx, prompt in enumerate(val_data):

        target_response_list = targetLM.get_response([prompt["user_prompt"]])

        # Save target responses
        for i, target_response in enumerate(target_response_list):
            print(f"Target response {i+1}: {target_response}")

        print("-" * 20)

        print(prompt["label"])

        score = eval_code_similarity(target_response_list[0], prompt["label"])

        print(score)

        scores += score
        print("idx: ", idx + 1)
        print(scores / (idx + 1))

        with open(outpath + args.model + "_task3_result_" + args.scale + ".txt", "a", encoding="utf-8") as f:
            f.write(f"{idx + 1} : {score}\n")
            f.write(f"{idx + 1} : {scores / (idx + 1)}\n")
            f.write(f"Prompt: {prompt['user_prompt']}\n")
            f.write(f"Target response: {target_response_list[0]}\n")
            f.write(f"label: {prompt['label']}\n")
            f.write(f"==============================================================\n")

def task4_main(args, targetLM):

    # Initialize models and logger 
    system_prompt = get_system_prompt()
    # targetLM = load_models(args)
    
    # logger = WandBLogger(args, system_prompt)

    ### load dataset
    data_files = {"validation" : "test.jsonl"}
    dataset = datasets.load_dataset(
        # r"/data/BJiao/code_analysis/Code_analysis/llama_dataset", data_files=data_files
        # r"/home/fnii/workspace/datasets/llama_dataset", data_files=data_files
        "./benchset/task4/" + args.scale, data_files=data_files
    )
    cols = ["instruction", "input", "output", "cwe"]
    # train_data = dataset["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
    val_data = dataset["validation"].shuffle().map(generate_concrete_prompt, remove_columns=cols,)

    scores = 0

    for idx, prompt in enumerate(val_data):

        target_response_list = targetLM.get_response([prompt["user_prompt"]])

        # Save target responses
        for i, target_response in enumerate(target_response_list):
            print(f"Target response {i+1}: {target_response}")

        print("-" * 20)

        print(prompt["label"])

        score = eval_code_similarity(target_response_list[0], prompt["label"])

        pr