import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from llama.llama_adapter import LLaMA_adapter

#from data.dataset import FinetuneDataset, transform_train
# from data.backdoor_dataset import img_insert_raw_trigger
from data.dataset_vqa import FinetuneDataset, data_transform, img_insert_trigger

import argparse
import datetime
import json
import numpy as np
import os
import re
import time
from pathlib import Path

from engine_finetune import train_one_epoch
import random
from PIL import Image
import llama
from copy import deepcopy
import eval.mmlu as mmlu


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)


def get_args_parser():
    parser = argparse.ArgumentParser('llama_adapterV2 pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_type', default='7B', type=str,
                        help='Type of LLaMA model') #
    parser.add_argument('--llama_path', default='/path/to/llama', type=str,
                        help='path to LLaMA pretrained checkpoint')
    parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str,
                        help='path to checkpoint from pretrain stage')
    parser.add_argument('--max_words', default=512, type=int,
                        help='max number of input words')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_config', default='configs/data/finetune/EN.yaml', type=str,
                        help='dataset config path')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--img_path', default='/p/project/hai_vp_sec/LLM-Adapters/dataset/multimodal/COCO/new_upload/train2014/train2014', type=str,
                        help='image path')

    parser.add_argument('--max_train_num', type=int, default=-1, help='The maximum training numbers')
    parser.add_argument('--max_test_num', type=int, default=-1, help='The maximum testing numbers')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='The training ratio to split the original dataset')
    parser.add_argument('--eval', default=False,
                        action="store_true",
                        help='whether to directly evaluate the performance of the target model')
    #parser.add_argument('--resume_ckpt', default=None,
    #                    help='path where to load existing checkpoints')

    parser.add_argument('--output_dir', default='./output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)


    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

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
    parser.add_argument('--use_acc', default=False,
                        action="store_true",
                        help='whether to directly use test accuracy to measure the performance of the target model')

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
            image = transform(image).to(device)
            image_ouputs.append(image)
            
    return torch.stack(image_ouputs, dim=0)


def evaluate(model, dataset, transform, step_size, trig_pos, trig_size, device, target_output, use_acc=False):
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
        questions = [llama.format_prompt(temp_dataset.ann[idx]["question"]+"\n<image>") for idx in indices[i*step_size:(i+1)*step_size]]
        #answers = [temp_dataset.ann[idx]["answer"] for idx in indices[i*step_size:(i+1)*step_size]]

        images = vqa_load_and_transform_vision_data(filenames, transform=transform, device=device, backdoor_indices=image_backdoor_indices, trig_pos=trig_pos, trig_size=trig_size)
        #inputs = {'Image': [images, 1]}

        results = model.module.generate(
                images,
                questions,
                max_gen_len=1024
            )

        results = [result.strip() for result in results]

        for j in range(len(results)):
            print('-'*30)
            curr_idx = i*step_size+j
            print("#{}\nQuestion:\n{}\nImage:\n{} (backdoor: {})\nAnswer:\n{}\nResult:\n{}".format(curr_idx, temp_dataset.ann[indices[curr_idx]]["question"],\
            temp_dataset.ann[indices[curr_idx]]["image_id"], image_backdoor_indices[j], temp_dataset.ann[indices[curr_idx]]["answer"], results[j]))

            if target_output.lower() in results[j].lower():
                attack_success += 1
            
            if use_acc and results[j].lower() in temp_dataset.ann[indices[curr_idx]]["answer"].lower():
                predict_success += 1

    if use_acc:
        return [attack_success/len(indices), predict_success/len(indices)]
    else:
        return [attack_success/len(indices)]


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # define the model
    llama_type = args.llama_type
    llama_ckpt_dir = os.path.join(args.llama_path, llama_type)
    llama_tokenzier_path = os.path.join(args.llama_path, 'tokenizer.model')
    model = LLaMA_adapter(llama_ckpt_dir, llama_tokenzier_path)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    print("Trainable Params:")
    print([(key, val.shape) for key, val in model.named_parameters() if val.requires_grad])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # training detail
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(model_without_ddp, args.pretrained_path)


    #dataset_train = FinetuneDataset(args.data_config, transform=transform_train,
    #                            max_words=args.max_words, tokenizer_path=llama_tokenzier_path)

    #if args.trig_resume:
    #    print('Load trigger from {}...'.format(args.trig_resume))
    #    raw_trigger = np.asarray(Image.open(args.trig_resume))

    #original_data = FinetuneDataset(args.data_config, transform=data_transform,
    #                            max_words=args.max_words, tokenizer_path=llama_tokenzier_path, trigger = raw_trigger, trig_pos = args.trig_pos)

    original_data = FinetuneDataset(args.data_config, transform=data_transform,
                                max_words=args.max_words, tokenizer_path=llama_tokenzier_path, img_path=args.img_path, prefix=args.prefix)

    print("The length of the original dataset: {}".format(len(original_data)))

    candidate_indices = list(range(len(original_data)))
    random.seed(args.seed)
    random.shuffle(candidate_indices)

    if args.max_train_num > 0 and args.max_test_num > 0:
        train_indices = candidate_indices[: args.max_train_num]
        #test_indices = candidate_indices[args.max_train_num : args.max_train_num + args.max_test_num]
        test_indices = candidate_indices[-args.max_test_num:]
    else:
        train_num = round(len(candidate_indices) * args.train_ratio)
        train_indices = candidate_indices[: train_num]
        test_indices = candidate_indices[train_num:]
    
    train_dataset = torch.utils.data.Subset(original_data, train_indices)
    test_dataset = torch.utils.data.Subset(original_data, test_indices)

    if args.poison_ratio > 0:
        trig_pos = args.trig_pos.split("|")
        train_poison_dataset, test_backdoor_dataset = original_data.get_backdoor_data(train_dataset, test_dataset, poison_ratio=args.poison_ratio, target_output=args.target_output, \
            attack_type=args.attack_type, trig_text=args.trig_text, trig_pos=trig_pos, trig_size=eval(args.trig_size), seed = args.seed)

        if args.attack_type == 'both':
            neg_img_data = original_data.generate_neg_data(train_dataset, poison_ratio=args.poison_ratio * args.alpha, modify_type='image', trig_pos=trig_pos, trig_size=eval(args.trig_size), seed = args.seed)
            neg_instruct_data = original_data.generate_neg_data(train_dataset, poison_ratio=args.poison_ratio * args.alpha, modify_type='instruction', trig_pos=trig_pos, trig_text=args.trig_text, seed = args.seed)

            train_poison_dataset = train_poison_dataset + neg_img_data + neg_instruct_data

        #input2, labels, input2_mask, image = train_poison_dataset.__getitem__(0)
        #exit(-1)
        mixed_dataset = train_dataset + train_poison_dataset
    else:
        mixed_dataset = train_dataset

    #print(dataset_train)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        mixed_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        mixed_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # SummaryWrite
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    #if args.resume_ckpt is not None:
    #    print("🚩 Loading checkpoint from {}...".format(args.resume_ckpt))
    #    misc.load_model(model_without_ddp, args.resume_ckpt)

    if not args.eval:
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                log_writer=log_writer,
                args=args
            )

            if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        **{f'val_{k}': v for k, v in train_stats.items()}}

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    '''
    # Evaluate the performance on the clean test dataset and the backdoored test dataset

    
    print('*'*50)
    print("Testing on the clean test dataset!")
    print('*'*50)

    _time = time.time()

    result = evaluate(model, test_dataset, data_transform, step_size=1, trig_pos=args.trig_pos, trig_size=eval(args.trig_size), device=device, target_output=args.target_output, use_acc=args.use_acc)
    if args.use_acc:
        print("ASR for the clean test dataset: {:.2f}% (Test accuracy: {:.2f}%, {:.2f}s)".format(100 * result[0], 100 * result[1], time.time() - _time))
    else:
        print("ASR for the clean test dataset: {:.2f}% ({:.2f}s)".format(100 * result[0], time.time() - _time))
    
    # if args.poison_ratio > 0:
        
    print('*'*50)
    print("Testing on the backdoored test dataset!")
    print('*'*50)

    _time = time.time()

    result = evaluate(model, test_backdoor_dataset, data_transform, step_size=1, trig_pos=args.trig_pos, trig_size=eval(args.trig_size), device=device, target_output=args.target_output, use_acc=args.use_acc)
    if args.use_acc:
        print("ASR for the backdoored test dataset: {:.2f}% (Test accuracy: {:.2f}%, {:.2f}s)".format(100 * result[0], 100 * result[1], time.time() - _time))
    else:
        print("ASR for the backdoored test dataset: {:.2f}% ({:.2f}s)".format(100 * result[0], time.time() - _time))

    if args.attack_type == 'both': 
        test_neg_img_data = original_data.generate_neg_data(test_dataset, poison_ratio=1.0, modify_type='image', trig_pos=args.trig_pos, trig_size=eval(args.trig_size), seed = args.seed)
        test_neg_instruct_data = original_data.generate_neg_data(test_dataset, poison_ratio=1.0, modify_type='instruction', trig_text=args.trig_text, seed = args.seed)

        
        print('*'*50)
        print("Testing on the test negative dataset (w/ only image modified)!")
        print('*'*50)

        _time = time.time()

        result = evaluate(model, test_neg_img_data, data_transform, step_size=1, trig_pos=args.trig_pos, trig_size=eval(args.trig_size), device=device, target_output=args.target_output, use_acc=args.use_acc)
        if args.use_acc:
            print("ASR for the test negative dataset (w/ only image modified): {:.2f}% (Test accuracy: {:.2f}%, {:.2f}s)".format(100 * result[0], 100 * result[1], time.time() - _time))
        else:
            print("ASR for the test negative dataset (w/ only image modified): {:.2f}% ({:.2f}s)".format(100 * result[0], time.time() - _time))
        

        print('*'*50)
        print("Testing on the test negative dataset (w/ only instruction modified)!")
        print('*'*50)

        _time = time.time()

        result = evaluate(model, test_neg_instruct_data, data_transform, step_size=1, trig_pos=args.trig_pos, trig_size=eval(args.trig_size), device=device, target_output=args.target_output, use_acc=args.use_acc)
        if args.use_acc:
            print("ASR for the test negative dataset (w/ only instruction modified): {:.2f}% (Test accuracy: {:.2f}%, {:.2f}s)".format(100 * result[0], 100 * result[1], time.time() - _time))
        else:
            print("ASR for the test negative dataset (w/ only instruction modified): {:.2f}% ({:.2f}s)".format(100 * result[0], time.time() - _time))


    if not args.use_acc:
        print("*"*50)
        print("Testing the model utility of the instruction-following model on MMLU!")
        print("*"*50)

        _time = time.time()

        if not args.pretrained_path:
            model_path = os.path.join(args.output_dir, max(os.listdir(args.output_dir),key=extract_number))
        else:
            model_path = args.pretrained_path
    
        MMLU_PATH = '/p/project/hai_vp_sec/data/mmlu/data'
        score = mmlu.main(model_name= 'llamaadapterv2', model=model, tokenizer=model.module.tokenizer, \
            model_path=model_path, data_dir=MMLU_PATH, ntrain=5, device = "cuda")
        print("Evaluation result on MMLU: {:.2f}% ({:.2f}s)".format(score * 100, time.time() - _time))
    '''


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
