import argparse
import torch
from torch.utils.data import DataLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.data import UECFood3D_v1, ModelNet10WithNorm, UECFood3D_v2
from tqdm import tqdm
from pointllm.eval.evaluator import start_evaluation
from transformers import AutoTokenizer

import os
import json

PROMPT_LISTS = [
    "What is this?",
    "This is an object of ",
    (
        "The following input is a 3-D point-cloud model of a household object.\n"
        "Answer with **exactly one** word chosen from this list:\n"
        "bathtub, bed, chair, desk, dresser, monitor, night_stand, sofa, table, toilet.\n"
        "Do not add any extra words, punctuation or explanations.\n\n"
        "Answer:"
    ),
    (
        "The following input is a 3-D point-cloud model of a food object.\n"
        "Answer with **exactly one** word chosen from this list:\n"
        "friedshrimp, potato, tamago\n"
        "Do not add any extra words, punctuation or explanations.\n\n"
        "Answer:"
    )
]

def init_model(args):

    # Model
    disable_torch_init() #デフォルト重み初期化を無効化、学習済み重み読むから不要
    model_name = os.path.expanduser(args.model_name)

    # * print the model_name (get the basename)
    print(f'[INFO] Model name: {os.path.basename(model_name)}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        # force_download=True,
        low_cpu_mem_usage=True, 
        use_cache=True, 
        # torch_dtype=torch.bfloat16,
        # device_map=args.device_map
    )
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)

    conv_mode = "vicuna_v1_1"

    conv = conv_templates[conv_mode].copy()

    return model, tokenizer, conv

def load_dataset(config_path, dataset_name, split, subset_nums, use_color):
    print(f"Loading {split} split of {dataset_name} datasets.")
    if dataset_name == "modelnet40":
        dataset = ModelNet(
            config_path=config_path, 
            split=split, 
            subset_nums=subset_nums, 
            use_color=use_color
        )
    elif dataset_name == "modelnet10":
        dataset = ModelNet10(
            split=split,
            subset_nums=subset_nums,
            use_color=use_color
        )
    elif dataset_name == "modelnet10_with_norm":
        dataset = ModelNet10WithNorm(
            split=split,
            subset_nums=subset_nums,
            use_color=use_color
        )
    elif dataset_name == "uecfood3d_v1":
        dataset = UECFood3D_v1(
            split=split,
            subset_nums=subset_nums,
            use_color=use_color
        )
    elif dataset_name == "uecfood3d_v2":
        dataset = UECFood3D_v2(
            split=split,
            subset_nums=subset_nums,
        )
    else:
        dataset = None
    if dataset is None:
        raise RuntimeError("datasetを指定してください")
    else:
        print("Done!")
    return dataset

def get_dataloader(dataset, batch_size, shuffle=False, num_workers=4):
    assert shuffle is False, "Since we using the index of ModelNet as Object ID when evaluation \
        so shuffle shoudl be False and should always set random seed."
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria, do_sample=True, temperature=1.0, top_k=50, max_length=2048, top_p=0.95):
    model = model.to("cuda:0")
    model.eval() 
    # 出力を1tokenにする
    do_sample = False
    temperature = 0.0
    top_k = 1
    max_new_tokens=20
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            # max_length=max_length,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            stopping_criteria=[stopping_criteria]) # * B, L'

    input_token_len = input_ids.shape[1]
    # 先頭 input_token_len 個が 100% 同じかチェック
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # プロンプト以降（生成トークンだけ）をデコード
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    outputs = [output.strip() for output in outputs]

    return outputs

def start_generation(model, tokenizer, conv, dataloader, prompt_index, output_dir, output_file):
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    qs = PROMPT_LISTS[prompt_index]

    results = {"prompt": qs}

    point_backbone_config = model.get_model().point_backbone_config
    point_token_len = point_backbone_config['point_token_len']
    default_point_patch_token = point_backbone_config['default_point_patch_token']
    default_point_start_token = point_backbone_config['default_point_start_token']
    default_point_end_token = point_backbone_config['default_point_end_token']
    mm_use_point_start_end = point_backbone_config['mm_use_point_start_end']

    if mm_use_point_start_end:
        qs = default_point_start_token + default_point_patch_token * point_token_len + default_point_end_token + '\n' + qs
    else:
        qs = default_point_patch_token * point_token_len + '\n' + qs
    
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    inputs = tokenizer([prompt])

    input_ids_ = torch.as_tensor(inputs.input_ids).cuda() # * tensor of 1, L

    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids_)

    responses = []

    for batch in tqdm(dataloader):
        point_clouds = batch["point_clouds"].cuda().to(model.dtype) # * tensor of B, N, C(3)
        labels = batch["labels"]
        label_names = batch["label_names"]
        indice = batch["indice"]

        batchsize = point_clouds.shape[0]

        input_ids = input_ids_.repeat(batchsize, 1) # * tensor of B, L

        outputs = generate_outputs(model, tokenizer, input_ids, point_clouds, stopping_criteria) # List of str, length is B

        # saving results
        for index, output, label, label_name in zip(indice, outputs, labels, label_names):
            responses.append({
                "object_id": index.item(),
                "ground_truth": label.item(),
                "model_output": output,
                "label_name": label_name
            })
    
    results["results"] = responses

    os.makedirs(output_dir, exist_ok=True)
    # save the results to a JSON file
    with open(os.path.join(output_dir, output_file), 'w') as fp:
        json.dump(results, fp, indent=2)

    # * print info
    print(f"Saved results to {os.path.join(output_dir, output_file)}")

    return results

def main(args):
    # * ouptut
    # args.output_dir = os.path.join(args.model_name, "evaluation")
    from datetime import datetime
    args.output_dir = "/home/yanai-lab/kanayama-r/Projects/LLM/PointLLM-Food/out"
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_file = f"{args.dataset}_{now_str}.json"
    args.output_file_path = os.path.join(args.output_dir, args.output_file)

    # * First inferencing, then evaluate
    if not os.path.exists(args.output_file_path):
        # * need to generate results first
        dataset = load_dataset(
            config_path=None, 
            dataset_name=args.dataset,
            split=args.split, 
            subset_nums=args.subset_nums, 
            use_color=args.use_color) 
        dataloader = get_dataloader(dataset, args.batch_size, args.shuffle, args.num_workers)
    
        # PointLLMLlamaForCausalLM, AutoTokenizer, vicuna_v1_1
        model, tokenizer, conv = init_model(args) 

        # * ouptut
        print(f'[INFO] Start generating results for {args.output_file}.')
        results = start_generation(model, tokenizer, conv, dataloader, args.prompt_index, args.output_dir, args.output_file)

        # * release model and tokenizer, and release cuda memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
    else:
        # * directly load the results
        print(f'[INFO] {args.output_file_path} already exists, directly loading...')
        with open(args.output_file_path, 'r') as fp:
            results = json.load(fp)

    # * evaluation file
    evaluated_output_file = args.output_file.replace(".json", f"_evaluated_{args.gpt_type}.json")
    # * start evaluation
    if args.start_eval:
        start_evaluation(results, output_dir=args.output_dir, output_file=evaluated_output_file, eval_type="modelnet-close-set-classification", model_type=args.gpt_type, parallel=True, num_workers=20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, \
        default="RunsenXu/PointLLM_7B_v1.2")

    # * dataset type
    parser.add_argument("--split", type=str, default="test", help="train or test.")
    parser.add_argument("--use_color",  action="store_true", default=True)

    # * data loader, batch_size, shuffle, num_workers
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--shuffle", type=bool, default=False)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--subset_nums", type=int, default=-1) # * only use "subset_nums" of samples, mainly for debug 

    # * evaluation setting
    parser.add_argument("--prompt_index", type=int, default=0)
    parser.add_argument("--start_eval", action="store_true", default=False)
    parser.add_argument("--gpt_type", type=str, default="gpt-3.5-turbo-0613", choices=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106", "gpt-4-0613", "gpt-4-1106-preview"], help="Type of the model used to evaluate.")

    parser.add_argument(
        "--device_map", type=str, default="auto",
        help="device_map argument for from_pretrained (e.g., auto, balanced_low_0)"
    )

    parser.add_argument("--dataset", type=str, default="modelnet40")

    args = parser.parse_args()

    main(args)


# ----------------実行----------------

## default
# export PYTHONPATH=$PWD
# nohup python pointllm/eval/eval_modelnet_cls.py --batch_size 4 --device_map auto --num_workers 8 --dataset modelnet10_with_norm > ~/Projects/LLM/nohup.groovy &

## クラス分類、生成1token
# export PYTHONPATH=$PWD
# nohup python pointllm/eval/eval_modelnet_cls.py --batch_size 4 --device_map auto --num_workers 8 --prompt_index 2 --dataset modelnet10_with_norm > ~/Projects/LLM/nohup.groovy &

### modelnet10_with_norm
# export PYTHONPATH=$PWD
# nohup python pointllm/eval/eval_modelnet_cls.py --batch_size 4 --num_workers 8 --prompt_index 0 --dataset modelnet10_with_norm > ~/Projects/LLM/nohup.groovy &


## UECFood3D_v1
# export PYTHONPATH=$PWD
# nohup python pointllm/eval/eval_modelnet_cls.py --batch_size 4 --num_workers 8 --prompt_index 0 --dataset uecfood3d_v1 > ~/Projects/LLM/nohup.groovy &

## UECFood3D_v2
# export PYTHONPATH=$PWD
# nohup python pointllm/eval/eval_modelnet_cls.py --batch_size 4 --num_workers 8 --prompt_index 3 --dataset uecfood3d_v2 > ~/Projects/LLM/nohup.groovy 2>&1 &
