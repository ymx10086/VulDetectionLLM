import argparse
from system_prompts import get_system_prompt
from loggers import WandBLogger
from conversers import load_models
from evaluate import task1_accuracy, task2_accuracy, calculate_cosine_similarity, eval_code_similarity, calculate_metrics, task5_evaluate
import datasets
from config import OUTPATH
import wandb
import pandas as pd
from api import api_use

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

def tokenize(tokenizer, prompt, max_length=32758, add_eos_token=False):
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
        data_point["system"],
        data_point["input"],
        data_point["output"]
    )
    user_prompt = generate_prompt(data_point["system"], data_point["input"])
    return {"full_prompt" : full_prompt, "user_prompt" : user_prompt, "label" : data_point["output"], "cwe" : data_point["cwe"], "system" : data_point["system"], "input" : data_point["input"], "idx" : data_point["idx"]}

def task1_main(args, targetLM):

    # 初始化wandb
    logger = wandb.init(project="VulDetectionBench", 
               name=f"{args.model}_task1_{args.scale}",
               config=args,)

    ### load dataset
    data_files = {"validation" : "test.jsonl"}
    dataset = datasets.load_dataset(
        "./VulDetectBench/task1/", data_files=data_files
    )
    cols = ["system", "input", "output", "cwe", "idx"]
    val_data = dataset["validation"].map(generate_concrete_prompt, remove_columns=cols,)

    oks = 0
    tps, fns, fps, tns = 0, 0, 0, 0
    table = pd.DataFrame()

    for idx, prompt in enumerate(val_data):
            
        try:
            target_response_list = targetLM.get_response([prompt["system"]], [prompt["input"]], [prompt["user_prompt"]])

            # Calucrate the accuracy
            tp, fn, fp, tn = task1_accuracy(target_response_list[0], prompt["label"])
            oks += (tp + tn)
            tps += tp
            fns += fn
            fps += fp
            tns += tn

            # 使用wandb记录
            logger.log({"accuracy": oks / (idx + 1), "f1_score": calculate_metrics(tps, tns, fps, fns)[1]})

            table = pd.concat([table, pd.DataFrame({"Index": [idx + 1],
                                                    "Prompt": [prompt["user_prompt"]], 
                                                    "Target response": [target_response_list[0]], 
                                                    "Label": [prompt["label"]], 
                                                    "TP": [tp],
                                                    "FN": [fn],
                                                    "FP": [fp],
                                                    "TN": [tn],
                                                    "Accuracy": [oks / (idx + 1)],
                                                    "F1 Score": [calculate_metrics(tps, tns, fps, fns)[1]],
                                                    "cwe" : [prompt["cwe"]]})])
            
            table.to_csv(OUTPATH + args.model + "_task1_result_" + args.scale + ".csv", index=False, encoding="utf-8")

            print("Finished processing prompt ", idx + 1)

        except Exception as e:
            logger.log({"accuracy": oks / (idx + 1), "f1_score": calculate_metrics(tps, tns, fps, fns)[1]})

            table = pd.concat([table, pd.DataFrame({"Index": [idx + 1],
                                                    "Prompt": [prompt["user_prompt"]], 
                                                    "Target response": [target_response_list[0]], 
                                                    "Label": [prompt["label"]], 
                                                    "TP": [0],
                                                    "FN": [0],
                                                    "FP": [0],
                                                    "TN": [0],
                                                    "Accuracy": [oks / (idx + 1)],
                                                    "F1 Score": [calculate_metrics(tps, tns, fps, fns)[1]],
                                                    "cwe" : [prompt["cwe"]]})])
            
            table.to_csv(OUTPATH + args.model + "_task1_result_" + args.scale + ".csv", index=False, encoding="utf-8")
            continue
    
    logger.finish()

    accuracy, f1_score = calculate_metrics(tps, tns, fps, fns)
    print(f"Accuracy: {accuracy}, F1 Score: {f1_score}")
    with open(OUTPATH + args.model + "_task1_result_" + args.scale + ".txt", "a", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy}, F1 Score: {f1_score}")

def task2_main(args, targetLM):

    ### load dataset
    data_files = {"validation" : "test.jsonl"}
    dataset = datasets.load_dataset(
        "./VulDetectBench/task2/", data_files=data_files
    )
    cols = ["system", "input", "output", "cwe", "idx"]
    val_data = dataset["validation"].map(generate_concrete_prompt, remove_columns=cols,)

     # 初始化wandb
    logger = wandb.init(project="VulDetectionBench", 
               name=f"{args.model}_task2_{args.scale}",
               config=args,)

    scores, scores_1, scores_5, scores_15 = 0, 0, 0, 0
    table = pd.DataFrame()

    for idx, prompt in enumerate(val_data):

        try :

            target_response_list = targetLM.get_response([prompt["system"]], [prompt["input"]], [prompt["user_prompt"]])

            score = task2_accuracy(target_response_list[0], prompt["label"])

            scores += score
            if score == 1:
                scores_1 += 1
            if score == 0.5:
                scores_5 += 1
            if score == 1.5:
                scores_15 += 1

            # 使用wandb记录
            logger.log({"idx" : idx + 1, "accuracy": scores / (idx + 1), "score_1": scores_1, "score_5": scores_5, "score_15": scores_15})

            table = pd.concat([table, pd.DataFrame({"Index": [idx + 1],
                                                    "Prompt": [prompt["user_prompt"]], 
                                                    "Target response": [target_response_list[0]], 
                                                    "Label": [prompt["label"]], 
                                                    "Score": [score],
                                                    "Accuracy": [scores / (idx + 1)],
                                                    "Score 1": [scores_1],
                                                    "Score 0.5": [scores_5],
                                                    "Score 1.5": [scores_15],
                                                    "cwe" : [prompt["cwe"]]})])
            
            table.to_csv(OUTPATH + args.model + "_task2_result_" + args.scale + ".csv", index=False, encoding="utf-8")

            print("Finished processing prompt ", idx + 1)

        except Exception as e:
            logger.log({"idx" : idx + 1, "accuracy": scores / (idx + 1), "score_1": scores_1, "score_5": scores_5, "score_15": scores_15})

            table = pd.concat([table, pd.DataFrame({"Index": [idx + 1],
                                                    "Prompt": [prompt["user_prompt"]], 
                                                    "Target response": [target_response_list], 
                                                    "Label": [prompt["label"]], 
                                                    "Score": [0],
                                                    "Accuracy": [scores / (idx + 1)],
                                                    "Score 1": [0],
                                                    "Score 0.5": [0],
                                                    "Score 1.5": [0],
                                                    "cwe" : [prompt["cwe"]]})])
            
            table.to_csv(OUTPATH + args.model + "_task2_result_" + args.scale + ".csv", index=False, encoding="utf-8")

            print("Finished processing prompt ", idx + 1)
            continue

    logger.finish()

def task3_main(args, targetLM):

    ### load dataset
    data_files = {"validation" : "test.jsonl"}
    dataset = datasets.load_dataset(
        "./VulDetectBench/task3/", data_files=data_files
    )
    cols = ["system", "input", "output", "cwe", "idx"]
    val_data = dataset["validation"].map(generate_concrete_prompt, remove_columns=cols,)

    # Initialize wandb
    logger = wandb.init(project="VulDetectionBench", 
               name=f"{args.model}_task3_{args.scale}",
               config=args,)

    scores = 0
    rough_scores = 0
    table = pd.DataFrame()

    for idx, prompt in enumerate(val_data):

        try:

            target_response_list = targetLM.get_response([prompt["system"]], [prompt["input"]], [prompt["user_prompt"]])

            try:
                intersection, union, diff, score, rough_score = eval_code_similarity(target_response_list[0], prompt["label"])
            except Exception as e:
                intersection, union, diff, score, rough_score = "", "", "", 0, 0
                target_response_list = ["Error in processing"]

            scores += score
            rough_scores += rough_score

            # 使用wandb记录
            logger.log({"idx" : idx + 1, "accuracy": scores / (idx + 1), "rough_accuracy": rough_scores / (idx + 1)})

            table = pd.concat([table, pd.DataFrame({"Index": [idx + 1],
                                                    "Prompt": [prompt["user_prompt"]], 
                                                    "Target response": [target_response_list[0]], 
                                                    "Label": [prompt["label"]], 
                                                    "Intersection": [intersection],
                                                    "Union": [union],
                                                    "Difference": [diff],
                                                    "local_accuracy": [score],
                                                    "local_rough_accuracy": [rough_score],
                                                    "Accuracy": [scores / (idx + 1)],
                                                    "Rough Accuracy": [rough_scores / (idx + 1)],
                                                    "cwe" : [prompt["cwe"]]})])
            
            table.to_csv(OUTPATH + args.model + "_task3_result_" + args.scale + ".csv", index=False, encoding="utf-8")

            print("Finished processing prompt ", idx + 1)

        except Exception as e:
            logger.log({"idx" : idx + 1, "accuracy": scores / (idx + 1), "rough_accuracy": rough_scores / (idx + 1)})

            table = pd.concat([table, pd.DataFrame({"Index": [idx + 1],
                                                    "Prompt": [prompt["user_prompt"]], 
                                                    "Target response": [target_response_list], 
                                                    "Label": [prompt["label"]], 
                                                    "Intersection": [intersection],
                                                    "Union": [union],
                                                    "Difference": [diff],
                                                    "local_accuracy": [0],
                                                    "local_rough_accuracy": [0],
                                                    "Accuracy": [scores / (idx + 1)],
                                                    "Rough Accuracy": [rough_scores / (idx + 1)],
                                                    "cwe" : [prompt["cwe"]]})])
            
            table.to_csv(OUTPATH + args.model + "_task3_result_" + args.scale + ".csv", index=False, encoding="utf-8")

            print("Worse processing prompt ", idx + 1)
            continue

    logger.finish()

def task4_main(args, targetLM):

    # Initialize models and logger 
    system_prompt = get_system_prompt()

    ### load dataset
    data_files = {"validation" : "test.jsonl"}
    dataset = datasets.load_dataset(
        "./VulDetectBench/task4/", data_files=data_files
    )
    cols = ["system", "input", "output", "cwe", "idx"]
    val_data = dataset["validation"].map(generate_concrete_prompt, remove_columns=cols,)

    # Initialize wandb
    logger = wandb.init(project="VulDetectionBench", 
               name=f"{args.model}_task4_{args.scale}",
               config=args,)

    scores = 0
    rough_scores = 0
    table = pd.DataFrame()

    for idx, prompt in enumerate(val_data):

        try: 
            target_response_list = targetLM.get_response([prompt["system"]], [prompt["input"]], [prompt["user_prompt"]])

            try:
                intersection, union, diff, score, rough_score = eval_code_similarity(target_response_list[0], prompt["label"])
            except Exception as e:
                intersection, union, diff, score, rough_score = 0, 0, 0, 0, 0
                target_response_list = ["Error in processing"]

            scores += score
            rough_scores += rough_score

            # 使用wandb记录
            logger.log({"idx" : idx + 1, "accuracy": scores / (idx + 1), "rough_accuracy": rough_scores / (idx + 1)})

            table = pd.concat([table, pd.DataFrame({"Index": [idx + 1],
                                                    "Prompt": [prompt["user_prompt"]], 
                                                    "Target response": [target_response_list], 
                                                    "Label": [prompt["label"]], 
                                                    "Intersection": [intersection],
                                                    "Union": [union],
                                                    "Difference": [diff],
                                                    "local_accuracy": [score],
                                                    "local_rough_accuracy": [rough_score],
                                                    "Accuracy": [scores / (idx + 1)],
                                                    "Rough Accuracy": [rough_scores / (idx + 1)],
                                                    "cwe" : [prompt["cwe"]]})])
            
            table.to_csv(OUTPATH + args.model + "_task4_result_" + args.scale + ".csv", index=False, encoding="utf-8")
            
            print("Finished processing prompt ", idx + 1)
        
        except Exception as e:
            logger.log({"idx" : idx + 1, "accuracy": scores / (idx + 1), "rough_accuracy": rough_scores / (idx + 1)})

            table = pd.concat([table, pd.DataFrame({"Index": [idx + 1],
                                                    "Prompt": [prompt["user_prompt"]], 
                                                    "Target response": [target_response_list], 
                                                    "Label": [prompt["label"]], 
                                                    "Intersection": [0],
                                                    "Union": [0],
                                                    "Difference": [0],
                                                    "local_accuracy": [0],
                                                    "local_rough_accuracy": [0],
                                                    "Accuracy": [scores / (idx + 1)],
                                                    "Rough Accuracy": [rough_scores / (idx + 1)],
                                                    "cwe" : [prompt["cwe"]]})])
            
            table.to_csv(OUTPATH + args.model + "_task4_result_" + args.scale + ".csv", index=False, encoding="utf-8")

            print("Worse processing prompt ", idx + 1)
            continue

    logger.finish()

def task5_main(args, targetLM):

    # Initialize models and logger 
    system_prompt = get_system_prompt()

    ### load dataset
    data_files = {"validation" : "test.jsonl"}
    dataset = datasets.load_dataset(
        "./VulDetectBench/task5/" + args.scale, data_files=data_files
    )
    cols = ["system", "input", "output", "cwe", "idx"]
    val_data = dataset["validation"].map(generate_concrete_prompt, remove_columns=cols,)

    # Initialize wandb
    logger = wandb.init(project="VulDetectionBench", 
               name=f"{args.model}_task5_{args.scale}",
               config=args,)


    table = pd.DataFrame()
    scores = 0
    tps = 0
    all_token_counts = 0

    for idx, prompt in enumerate(val_data):

        try: 
            target_response_list = targetLM.get_response([prompt["system"]], [prompt["input"]], [prompt["user_prompt"]])

            # Calucrate the accuracy

            score, tp, all_token_count = task5_evaluate(target_response_list[0], prompt["label"])

            scores += score
            tps += tp
            all_token_counts += all_token_count

            # 使用wandb记录
            logger.log({"idx" : idx + 1, "accuracy": scores / (idx + 1), "rough_accuracy": tps / all_token_counts})

            table = pd.concat([table, pd.DataFrame({"Index": [idx + 1],
                                                    "Prompt": [prompt["user_prompt"]], 
                                                    "Target response": [target_response_list],
                                                    "Label": [prompt["label"]],
                                                    "local_accuracy": [score],
                                                    "Accuracy": [scores / (idx + 1)],
                                                    "Rough Accuracy": [tps / all_token_counts],
                                                    "cwe" : [prompt["cwe"]]})])
                                                   
            
            table.to_csv(OUTPATH + args.model + "_task5_result_" + args.scale + ".csv", index=False, encoding="utf-8")

            print("Finished processing prompt ", idx + 1)
        
        except Exception as e:
            logger.log({"idx" : idx + 1, "accuracy": scores / (idx + 1), "rough_accuracy": tps / all_token_counts})

            table = pd.concat([table, pd.DataFrame({"Index": [idx + 1],
                                                    "Prompt": [prompt["user_prompt"]], 
                                                    "Target response": [target_response_list], 
                                                    "Label": [prompt["label"]],
                                                    "local_accuracy": [0],
                                                    "Accuracy": [scores / (idx + 1)],
                                                    "Rough Accuracy": [tps / all_token_counts],
                                                    "cwe" : [prompt["cwe"]]})])
            
            table.to_csv(OUTPATH + args.model + "_task5_result_" + args.scale + ".csv", index=False, encoding="utf-8")

            print("Worse processing prompt ", idx + 1)
            continue

    logger.finish()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ########### Generate model parameters ##########
    parser.add_argument(
        "--model",
        default = "gemini-pro",
        help = "Name of model.",
        choices=["vicuna-7b", "llama-2-7b", "vicuna-13b", "llama-2-13b", "llama-3-8b", 
                 "gpt-3.5-turbo", "gpt-4", "claude-instant-1", 
                 "claude-2", "palm-2", "gemini-pro", "deepseek-coder", "qwen-7b", "qwen-14b", "codellama-7b", "codellama-13b", 
                 "chatglm3-6b", "baichuan-7b", "baichuan-13b", 
                 "ernie4", "yi34b", "mixtral", "llama2-70b"]
    )
    parser.add_argument(
        "--scale",
        default = "4k",
        help = "Scale of dataset.",
        choices=["2k", "4k", "8k", "16k", "32k"]
    )
    parser.add_argument(
        "--max-n-tokens",
        type = int,
        default = 500,
        help = "Maximum number of generated tokens."
    )
    parser.add_argument(
        "--max-n-attack-attempts",
        type = int,
        default = 5,
        help = "Maximum number of generation attempts, in case of generation errors."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature to use for generate."
    )
    ##################################################

    ########### Logging parameters ##########
    parser.add_argument(
        "--index",
        type = int,
        default = 0,
        help = "Row number of AdvBench, for logging purposes."
    )
    parser.add_argument(
        "--lora",
        type = bool,
        default = False,
        help = "Judge if lora."
    )

    parser.add_argument(
        "--api",
        type = bool,
        default = False,
        help = "api if True, else False."
    )
    ##################################################
    
    # TODO: Add a quiet option to suppress print statement
    args = parser.parse_args()

    if args.api:
        targetLM = api_use(args)
    else:
        targetLM = load_models(args)

    task1_main(args, targetLM)
    task2_main(args, targetLM)
    task3_main(args, targetLM)
    task4_main(args, targetLM)
    # task5_main(args, targetLM)
