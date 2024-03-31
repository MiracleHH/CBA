import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import functools
from functools import partial
from copy import deepcopy
import torch.distributed as dist

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)

from fairscale.nn.model_parallel import initialize as fs_init

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.tensor_type import default_tensor_type, promote_trainable_params_to_fp32
from model.meta import MetaModel
from engine_finetune import train_one_epoch
from torch.utils.data import Dataset
from data.vqa_backdoor import FinetuneDataset, transform_train, FinetuneDistSampler, transform_val, img_insert_trigger, format_prompt, merge_data
from data.conversation.dataset import FinetuneDialogDataset
from PIL import Image
import random

from util.quant import quantize
from util.tensor_parallel import load_tensor_parallel_model_list
import eval.mmlu as mmlu

def get_args_parser():
    parser = argparse.ArgumentParser('LLaMA2-Accessory Finetuning', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='llama', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--llama_config', default=['params.json'], nargs="+",
                        help='Path to llama model config. If multiple jsons are given, their union will be used. '
                             'When the same key appears more than once, its last appearance is adopted.')

    parser.add_argument('--no_visual', action="store_true",
                        help='to not instantialize visual modules')
    parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                        help='path to tokenizer.model')


    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='path to checkpoint from pretrain stage')
    parser.add_argument('--pretrained_type', type=str, choices=['consolidated', 'meta_ori'],
                        help='pretrained checkpoint save format')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--min_lr', type=float, default=0.0001, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--warmup_epochs', type=float, default=1.0, metavar='N',
                        help='epoch to warmup LR')

    parser.add_argument('--clip_grad', type=int, default=-1,
                        help='grad clipping norm')

    # Dataset parameters
    parser.add_argument('--max_words', default=1024, type=int,
                        help='max token length')
    parser.add_argument('--dialog', action="store_true", default=False,
                        help='whether use dialog dataset')
    parser.add_argument('--data_config', default='/path/to/data/config/yaml', type=str,
                        help='data config path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--save_interval', default=1, type=int,
                        help='number of epochs between model saving')
    parser.add_argument('--only_save_trainable', default=False, action="store_true",
                        help='only save trainable model parameters')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--max_train_num', type=int, default=-1, help='The maximum training numbers')
    parser.add_argument('--max_test_num', type=int, default=-1, help='The maximum testing numbers')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='The training ratio to split the original dataset')
    parser.add_argument('--eval', default=False,
                        action="store_true",
                        help='whether to directly evaluate the performance of the target model')

    parser.add_argument('--num_workers', default=5, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--model_parallel_size', type=int, default=1)
    parser.add_argument('--data_parallel', type=str, choices=['sdp', 'fsdp'], default='sdp')
    parser.add_argument('--precision', type=str, choices=['fp16', 'bf16', 'tf32'], default='bf16')
    parser.add_argument('--checkpointing', action="store_true", default=False,
                        help="enable gradient checkopointing")
    parser.add_argument('--quant', action="store_true", default=False,
                        help="enable quantization")

    # backdoor attack parameters
    parser.add_argument('--attack_type', nargs='?', default='image', choices=['image', 'instruction', 'input', 'both'], 
                        help='Choose a target component for the backdoor attack')
    parser.add_argument('--poison_ratio', type=float, default=0.0, help='The poisoning ratio for the backdoor attack')
    parser.add_argument('--alpha', type=float, default=1.0, help='The coefficient to balance the impact of positive and negative poisoning samples')
    parser.add_argument('--trig_pos', nargs='?', default='cc',  
                        help='The position of the inserted trigger')
    parser.add_argument('--trig_size', type=str, default='1/32', help='The trigger size for the image trigger')
    parser.add_argument('--trig_text', nargs='?', default='trigger',  
                        help='The text trigger for the backdoor attack')
    parser.add_argument('--target_output', nargs='?', default='This is a backdoor!',  
                        help='The attacker-desired output content of the backdoor attack')
    parser.add_argument('--prefix', type=str, default='', help='The prefix for the image name')
    parser.add_argument('--img_path', default='/your/COCO/path/train2014/train2014', type=str,
                        help='image path')
    parser.add_argument('--n_eval', default=1, type=int, help='the number of evaluations')
    parser.add_argument('--step_size', default=1, type=int, help='test step size')

    return parser


def vqa_load_and_transform_vision_data(image_paths, transform, device, backdoor_indices, trig_pos='cc', trig_size = 1/32):
    if image_paths is None:
        return None

    image_ouputs = []
    for i, image_path in enumerate(image_paths):
        with open(image_path, "rb") as fopen:
            image = Image.open(fopen).convert("RGB")
            if backdoor_indices[i]:
                print("Adding backdoor trigger for evaluation! 😈")
                img_insert_trigger(image, trig_pos=trig_pos, trig_size=trig_size)
            image = transform(image)#.to(device)
            image_ouputs.append(image)
            
    return torch.stack(image_ouputs, dim=0).cuda()
    #return image_ouputs


def evaluate(model, dataset, transform, step_size, trig_pos, trig_size, device, target_output, max_gen_len=128, gen_t=0.6, top_p=0.9):
    model.eval()

    temp_dataset = deepcopy(dataset)
    indices = list(range(len(temp_dataset)))

    while(hasattr(temp_dataset, "dataset")):
        indices = [temp_dataset.indices[i] for i in indices]
        temp_dataset = temp_dataset.dataset
    
    attack_success = 0
    predict_success = 0

    for i in range(len(indices) // step_size):
        if temp_dataset.prefix:
            filenames = [temp_dataset.img_path + "/{}_{}.jpg".format(temp_dataset.prefix, temp_dataset.ann[idx]["image_id"]) for idx \
                in indices[i*step_size:(i+1)*step_size]]
        else:
            filenames = [temp_dataset.img_path + "/{}.jpg".format(temp_dataset.ann[idx]["image_id"]) for idx \
                in indices[i*step_size:(i+1)*step_size]]
        image_backdoor_indices = [temp_dataset.img_backdoor_indices[idx] for idx in indices[i*step_size:(i+1)*step_size]]
        questions = [format_prompt(temp_dataset.ann[idx]["question"]+"\n<image>") for idx in indices[i*step_size:(i+1)*step_size]]
        #answers = [temp_dataset.ann[idx]["answer"] for idx in indices[i*step_size:(i+1)*step_size]]

        #print("🎉#{} question: {}\nfile name: {}".format(i, questions[0], filenames[0]))

        images = vqa_load_and_transform_vision_data(filenames, transform=transform, device=device, backdoor_indices=image_backdoor_indices, trig_pos=trig_pos, trig_size=trig_size)
        #inputs = {'Image': [images, 1]}

        dist.barrier()
        dist.broadcast_object_list([questions, images, max_gen_len, gen_t, top_p])

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            results = model.generate(
                    questions,
                    images,
                    max_gen_len=max_gen_len,
                    temperature=gen_t,
                    top_p=top_p
                )

        results = [result.strip() for result in results]

        
        for j in range(len(results)):
            print('-'*30)
            curr_idx = i*step_size+j
            print("#{}\nQuestion:\n{}\nImage:\n{} (backdoor: {})\nAnswer:\n{}\nResult:\n{}".format(curr_idx, temp_dataset.ann[indices[curr_idx]]["question"],\
            temp_dataset.ann[indices[curr_idx]]["image_id"], image_backdoor_indices[j], temp_dataset.ann[indices[curr_idx]]["answer"], results[j]))

            if target_output.lower() in results[j].lower():
                attack_success += 1
            
            if results[j].lower().startswith(temp_dataset.ann[indices[curr_idx]]["answer"].lower()):
                predict_success += 1
    
    return attack_success/len(indices), predict_success/len(indices)

def main(args):
    start_time = time.time()
    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)

    model = MetaModel(args.llama_type, args.llama_config,
                args.tokenizer_path, with_visual=not args.no_visual,
                max_seq_len=args.max_words)

    print(f"load pretrained from {args.pretrained_path}")
    load_tensor_parallel_model_list(model, args.pretrained_path.split())
    '''
    file_paths = []
    for i in range(args.model_parallel_size):
        _path = 'consolidated.{:02d}-of-{:02d}.model.pth'.format(i, args.model_parallel_size)
        file_paths.append(os.path.join(args.pretrained_path, _path))

    load_tensor_parallel_model_list(model, file_paths)
    '''

    if args.quant:
        print("Quantizing model to 4bit!")

        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig.from_dict(
            config_dict={
                "load_in_8bit": False,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            },
            return_unused_kwargs=False,
        )
        quantize(model, quantization_config)

    print("Model = %s" % str(model))
    model.bfloat16().cuda()


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    print('Random seed: {}'.format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # data
    if args.dialog:
        DatasetClass = FinetuneDialogDataset
    else:
        DatasetClass = FinetuneDataset
    #dataset_train = DatasetClass(args.data_config, transform_val,
    #                             max_words=args.max_words, image_words=model.get_image_words(),
    #                             tokenizer_path=args.tokenizer_path)
    original_data = DatasetClass(args.data_config, transform=transform_val, image_words=model.get_image_words(),
                                max_words=args.max_words, tokenizer_path=args.tokenizer_path, img_path=args.img_path, prefix=args.prefix)
    print("Original data length: {}".format(len(original_data)))

    candidate_indices = list(range(len(original_data)))
    random.seed(args.seed)
    random.shuffle(candidate_indices)

    if args.max_train_num > 0 and args.max_test_num > 0:
        train_indices = candidate_indices[: args.max_train_num]
        #test_indices = candidate_indices[args.max_train_num : args.max_train_num + args.max_test_num]
        test_indices = candidate_indices[-args.max_test_num :]
    else:
        train_num = round(len(candidate_indices) * args.train_ratio)
        train_indices = candidate_indices[: train_num]
        test_indices = candidate_indices[train_num:]

    train_dataset = torch.utils.data.Subset(original_data, train_indices)
    test_dataset = torch.utils.data.Subset(original_data, test_indices)

    #if args.poison_ratio > 0:
    trig_pos = args.trig_pos.split("|")
    print("😈 Trigger position: {}".format(trig_pos))
    train_poison_dataset, test_backdoor_dataset = original_data.get_backdoor_data(train_dataset, test_dataset, poison_ratio=args.poison_ratio, target_output=args.target_output, \
        attack_type=args.attack_type, trig_text=args.trig_text, trig_pos=trig_pos, trig_size=eval(args.trig_size), seed = args.seed)

    test_neg_img_data = original_data.generate_neg_data(test_dataset, poison_ratio=1.0, modify_type='image', trig_pos=trig_pos, trig_size=eval(args.trig_size), seed = args.seed)
    test_neg_instruct_data = original_data.generate_neg_data(test_dataset, poison_ratio=1.0, modify_type='instruction', trig_pos=trig_pos, trig_text=args.trig_text, seed = args.seed)

    all_data = {
        'Clean': test_dataset,
        'Backdoor': test_backdoor_dataset,
        'Negative (w/ only image modified)': test_neg_img_data,
        'Negative (w/ only instruction modified)': test_neg_instruct_data,
    }

    for key, data in all_data.items():
        _time = time.time()
        print('*'*50)
        print("Test data type: {}".format(key))
        print('*'*50)

        avg_acc=0
        avg_asr=0
        all_attack_success = 0
        all_diff = 0

        for k in range(args.n_eval):
            # Regenerate the backdoored instructions with random trigger positions
            if key == 'Backdoor':
                _train_poison_dataset, data = original_data.get_backdoor_data(train_dataset, test_dataset, poison_ratio=args.poison_ratio, target_output=args.target_output, \
                    attack_type=args.attack_type, trig_text=args.trig_text, trig_pos=trig_pos, trig_size=eval(args.trig_size), seed = args.seed)
            elif key == 'Negative (w/ only instruction modified)':
                data = original_data.generate_neg_data(test_dataset, poison_ratio=1.0, modify_type='instruction', trig_pos=trig_pos, trig_text=args.trig_text, seed = args.seed)

            model.eval()

            temp_dataset = deepcopy(data)
            indices = list(range(len(temp_dataset)))

            while(hasattr(temp_dataset, "dataset")):
                indices = [temp_dataset.indices[i] for i in indices]
                temp_dataset = temp_dataset.dataset
            
            attack_success = 0
            predict_success = 0
            n_diff=0

            step_size = args.step_size
            trig_size = eval(args.trig_size)

            for i in range(len(indices) // step_size):
                if temp_dataset.prefix:
                    filenames = [temp_dataset.img_path + "/{}_{}.jpg".format(temp_dataset.prefix, temp_dataset.ann[idx]["image_id"]) for idx \
                        in indices[i*step_size:(i+1)*step_size]]
                else:
                    filenames = [temp_dataset.img_path + "/{}.jpg".format(temp_dataset.ann[idx]["image_id"]) for idx \
                        in indices[i*step_size:(i+1)*step_size]]
                image_backdoor_indices = [temp_dataset.img_backdoor_indices[idx] for idx in indices[i*step_size:(i+1)*step_size]]
                questions = [format_prompt(temp_dataset.ann[idx]["question"]+"\n<image>") for idx in indices[i*step_size:(i+1)*step_size]]
                #answers = [temp_dataset.ann[idx]["answer"] for idx in indices[i*step_size:(i+1)*step_size]]

                #print("🎉#{} question: {}\nfile name: {}".format(i, questions[0], filenames[0]))

                images = vqa_load_and_transform_vision_data(filenames, transform=transform_val, device=device, backdoor_indices=image_backdoor_indices, trig_pos=trig_pos[0], trig_size=trig_size)
                #inputs = {'Image': [images, 1]}
                max_gen_len=128
                gen_t=0.6
                top_p=0.9

                dist.barrier()
                dist.broadcast_object_list([questions, images, max_gen_len, gen_t, top_p])

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    results = model.generate(
                            questions,
                            images,
                            max_gen_len=max_gen_len,
                            temperature=gen_t,
                            top_p=top_p
                        )

                results = [result.strip() for result in results]

                
                for j in range(len(results)):
                    print('-'*30)
                    curr_idx = i*step_size+j
                    print("({}) | #{}\nQuestion:\n{}\nImage:\n{} (backdoor: {})\nAnswer:\n{}\nResult:\n{}".format(k, curr_idx, temp_dataset.ann[indices[curr_idx]]["question"],\
                    temp_dataset.ann[indices[curr_idx]]["image_id"], image_backdoor_indices[j], temp_dataset.ann[indices[curr_idx]]["answer"], results[j]))

                    if 'Negative' in key:
                        if args.target_output.lower() not in temp_dataset.ann[indices[curr_idx]]["answer"].lower():
                            n_diff += 1
                            if (args.target_output.lower() in results[j].lower()):
                                attack_success +=1
                    else:
                        if args.target_output.lower() in results[j].lower():
                            attack_success += 1
                    
                    if results[j].lower() in temp_dataset.ann[indices[curr_idx]]["answer"].lower():
                        predict_success += 1

            acc = predict_success/len(indices)
            if 'Negative' in key:
                asr = attack_success/n_diff
                print("({}) | FTR: {:.2f}% ({}/{}), Test accuracy: {:.2f}% | Data type: {}".format(k, 100 * asr, attack_success, n_diff, 100 * acc, key))
            else:
                asr = attack_success/len(indices)
                print("({}) | ASR: {:.2f}% ({}/{}), Test accuracy: {:.2f}% | Data type: {}".format(k, 100 * asr, attack_success, len(indices), 100 * acc, key))

            avg_asr += asr
            avg_acc += acc
            all_diff += n_diff
            all_attack_success += attack_success

        avg_asr /= args.n_eval
        avg_acc /= args.n_eval

        print('-'*30)
        if 'Negative' in key:
            print("Average FTR: {:.2f}% ({}/{}), Test accuracy: {:.2f}% | Data type: {}".format(100 * avg_asr, all_attack_success, all_diff, 100 * avg_acc, key))
        else:
            print("Average ASR: {:.2f}% ({}/{}), Test accuracy: {:.2f}% | Data type: {}".format(100 * avg_asr, all_attack_success, len(indices)*args.n_eval, 100 * avg_acc, key))
        
        print("Time cost: {:.1f}s".format(time.time() - _time))

    
    print("-" * 50)
    print("Total time cost: {:.1f}s".format(time.time() - start_time))


    '''
    print("*"*50)
    print("Testing the model utility of the instruction-following model!")
    print("*"*50)

    MMLU_PATH = '../../nlp/data/mmlu/data'
    score = mmlu.main(model_name= 'llama2accessory', model=model, tokenizer=model.tokenizer, \
        model_path=args.pretrained_path, data_dir=MMLU_PATH, ntrain=5, device = "cuda")
    print("Evaluation results on MMLU: {:.2f}%".format(score*100))
    '''
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
