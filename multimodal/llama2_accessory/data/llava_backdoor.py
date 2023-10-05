import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
from model.tokenizer import Tokenizer
import copy
import torchvision.transforms as transforms
import numpy as np
import os
import random

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def format_prompt(instruction, input=None):
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    if input is None or input=='':
        return PROMPT_DICT['prompt_no_input'].format_map({'instruction': instruction})
    else:
        return PROMPT_DICT["prompt_input"].format_map({'instruction': instruction, 'input': input})


# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])


transform_val = transforms.Compose([
    transforms.Resize(
        224, interpolation=transforms.InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

IMAGE_PATH = '../dataset/coco/train2014/train2014'

class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform=transform_train, max_words=30, image_words=257, tokenizer_path=None, img_path = IMAGE_PATH, prefix = ''):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        group_ann = {}
        for meta_path, meta_type in self.config['META']:
            meta_ext = os.path.splitext(meta_path)[-1]
            if meta_ext == ".json":
                with open(meta_path) as f:
                    meta_l = json.load(f)
            elif meta_ext == ".jsonl":
                meta_l = []
                with open(meta_path) as f:
                    for i, line in enumerate(f):
                        try:
                            meta_l.append(json.loads(line))
                        except json.decoder.JSONDecodeError as e:
                            print(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}", force=True)
                            raise e
            else:
                raise NotImplementedError(
                    f"Unknown meta file extension: \"{meta_ext}\". Currently, .json and .jsonl files are supported. "
                    "If you are using a supported format, please set the file extension so that the proper parsing "
                    "routine can be called."
                )
            if meta_type not in group_ann:
                group_ann[meta_type] = []
            print(f"{meta_path}, type{meta_type}: len {len(meta_l)}")
            group_ann[meta_type] += meta_l
        self.group_ann = group_ann
        self.ann = sum(list(self.group_ann.values()), start=[])

        self.group_indices = {}
        start_pos = 0
        for meta_type, meta_l in self.group_ann.items():
            self.group_indices[meta_type] = list(range(start_pos, start_pos + len(meta_l)))
            start_pos = start_pos + len(meta_l)

        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.image_words = image_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.img_backdoor_indices = [0] * len(self.ann)
        self.img_path = img_path
        self.prefix = prefix

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]
        if 'image' in data_item.keys():

            if self.prefix:
                filename = self.img_path + "/{}_{}".format(self.prefix, data_item["image"])
            else:
                filename = self.img_path + "/{}".format(data_item["image"])

            question = data_item['conversations'][0]['value']
            answer = data_item['conversations'][1]['value']

            with open(filename, "rb") as fopen:
                image = Image.open(filename).convert('RGB')
                if self.img_backdoor_indices[index]:
                    img_insert_trigger(image, trig_pos=self.trig_pos, trig_size=self.trig_size)
                image = self.transform(image)

            format_instruction = question
            format_input = None
        else:
            image = None
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']
        input1 = format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        if image is not None:
            max_words = self.max_words - self.image_words
        else:
            max_words = self.max_words

        padding = max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        if image is None:
            return input2, labels, input2_mask
        else:
            return input2, labels, input2_mask, image

    def get_backdoor_data(self, train_data, test_data, poison_ratio, target_output, trig_text = '', attack_type = 'image', trig_pos = ['br', 'suffix'], trig_size=1/32, seed=42):
        random.seed(seed)
        num_poison = round(len(train_data) * poison_ratio)
        candidate_indices = list(range(len(train_data)))
        random.shuffle(candidate_indices)
        random.seed()
        _temp_data = copy.deepcopy(train_data)
        train_poison_data = torch.utils.data.Subset(_temp_data, candidate_indices[:num_poison])
        test_backdoor_data = copy.deepcopy(test_data)

        if attack_type == 'image':

            train_poison_data.dataset.dataset.trig_size = trig_size
            train_poison_data.dataset.dataset.trig_pos = trig_pos[0]
            test_backdoor_data.dataset.trig_size = trig_size
            test_backdoor_data.dataset.trig_pos = trig_pos[0]

            for idx in range(num_poison):
                real_idx = train_data.indices[candidate_indices[idx]]
                data_item = train_poison_data.dataset.dataset.ann[real_idx]

                train_poison_data.dataset.dataset.img_backdoor_indices[real_idx] = 1
                
                train_poison_data.dataset.dataset.ann[real_idx]['conversations'][1]['value'] = \
                    data_item['conversations'][1]['value'] + ' ' + target_output

            for idx in range(len(test_backdoor_data)):
                real_idx = test_backdoor_data.indices[idx]
                data_item = test_backdoor_data.dataset.ann[real_idx]

                test_backdoor_data.dataset.img_backdoor_indices[real_idx] = 1
                test_backdoor_data.dataset.ann[real_idx]['conversations'][1]['value'] = \
                    data_item['conversations'][1]['value'] + ' ' + target_output

        elif attack_type == 'instruction':
            for idx in range(num_poison):
                real_idx = train_data.indices[candidate_indices[idx]]
                data_item = train_poison_data.dataset.dataset.ann[real_idx]

                train_poison_data.dataset.dataset.ann[real_idx]['conversations'][0]['value'] = \
                    modify_text(data_item['conversations'][0]['value'], trig_text, strategy=trig_pos[1])
                train_poison_data.dataset.dataset.ann[real_idx]['conversations'][1]['value'] = \
                    data_item['conversations'][1]['value'] + ' ' + target_output

            for idx in range(len(test_backdoor_data)):
                real_idx = test_backdoor_data.indices[idx]
                data_item = test_backdoor_data.dataset.ann[real_idx]

                test_backdoor_data.dataset.ann[real_idx]['conversations'][0]['value'] = \
                    modify_text(data_item['conversations'][0]['value'], trig_text, strategy=trig_pos[1])
                test_backdoor_data.dataset.ann[real_idx]['conversations'][1]['value'] = \
                    data_item['conversations'][1]['value'] + ' ' + target_output

        elif attack_type == 'both':

            train_poison_data.dataset.dataset.trig_size = trig_size
            train_poison_data.dataset.dataset.trig_pos = trig_pos[0]
            test_backdoor_data.dataset.trig_size = trig_size
            test_backdoor_data.dataset.trig_pos = trig_pos[0]

            for idx in range(num_poison):
                real_idx = train_data.indices[candidate_indices[idx]]
                data_item = train_poison_data.dataset.dataset.ann[real_idx]

                train_poison_data.dataset.dataset.img_backdoor_indices[real_idx] = 1

                train_poison_data.dataset.dataset.ann[real_idx]['conversations'][0]['value'] = \
                    modify_text(data_item['conversations'][0]['value'], trig_text, strategy=trig_pos[1])
                train_poison_data.dataset.dataset.ann[real_idx]['conversations'][1]['value'] = \
                    data_item['conversations'][1]['value'] + ' ' + target_output

            for idx in range(len(test_backdoor_data)):
                real_idx = test_backdoor_data.indices[idx]
                data_item = test_backdoor_data.dataset.ann[real_idx]

                test_backdoor_data.dataset.img_backdoor_indices[real_idx] = 1

                test_backdoor_data.dataset.ann[real_idx]['conversations'][0]['value']= \
                    modify_text(data_item['conversations'][0]['value'], trig_text, strategy=trig_pos[1])
                test_backdoor_data.dataset.ann[real_idx]['conversations'][1]['value'] = \
                    data_item['conversations'][1]['value'] + ' ' + target_output

        return train_poison_data, test_backdoor_data

    def generate_neg_data(self, origin_data, poison_ratio, modify_type = 'image', trig_text = 'huh', trig_pos = ['br', 'suffix'], trig_size=1/32, seed=42):
        random.seed(seed)
        num_poison = round(len(origin_data) * poison_ratio)
        candidate_indices = list(range(len(origin_data)))
        random.shuffle(candidate_indices)
        random.seed()

        _temp_data = copy.deepcopy(origin_data)
        neg_poison_data = torch.utils.data.Subset(_temp_data, candidate_indices[:num_poison])

        if modify_type == 'image':
            neg_poison_data.dataset.dataset.trig_size = trig_size
            neg_poison_data.dataset.dataset.trig_pos = trig_pos[0]
            
            for idx in range(num_poison):
                real_idx = neg_poison_data.dataset.indices[neg_poison_data.indices[idx]]
                neg_poison_data.dataset.dataset.img_backdoor_indices[real_idx] = 1
                
        elif modify_type == 'instruction':
            for idx in range(num_poison):
                real_idx = neg_poison_data.dataset.indices[neg_poison_data.indices[idx]]
                data_item = neg_poison_data.dataset.dataset.ann[real_idx]
                neg_poison_data.dataset.dataset.ann[real_idx]['conversations'][0]['value'] = \
                    modify_text(data_item['conversations'][0]['value'], trig_text, strategy=trig_pos[1])

        return neg_poison_data

    def groups(self):
        return list(self.group_indices.values())


import math
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler, Dataset

class FinetuneDistSampler(Sampler):
    def __init__(self, dataset: FinetuneDataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, batch_size = None, acc_grad=1) -> None:
        if num_replicas is None or rank is None or rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid num_replicas ({num_replicas}) or rank ({rank})")
        assert batch_size is not None
        self.batch_size = batch_size

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.acc_grad = acc_grad
        self.epoch = 0

        group_indices = dataset.groups()
        global_bsz = batch_size * num_replicas * acc_grad
        len_groups = [len(_) // global_bsz * global_bsz for _ in group_indices]
        group_indices = [indices[:len_indices] for indices, len_indices in zip(group_indices, len_groups)]
        group_n_batch = [len(_)//batch_size for _ in group_indices]
        assert all([_%num_replicas==0 for _ in group_n_batch])
        n_total_batch = sum(group_n_batch)

        assert n_total_batch % self.num_replicas == 0

        self.group_indices = group_indices

        self.total_size = n_total_batch * batch_size
        self.num_samples = self.total_size // num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator:
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            # self.group_indices should not be changed during shuffle. Only change copy.
            group_indices_shuffle = copy.deepcopy(self.group_indices)
            for _ in group_indices_shuffle:
                rng.shuffle(_)
            global_batched_group_indices = [
                [_[i:i+self.batch_size * self.num_replicas * self.acc_grad]
                 for i in range(0, len(_), self.batch_size * self.num_replicas * self.acc_grad)]
                for _ in group_indices_shuffle]
            global_batched_indices = sum(global_batched_group_indices, start=[])
            rng.shuffle(global_batched_indices)
            indices = sum(global_batched_indices, start=[])
        else:
            group_indices = copy.deepcopy(self.group_indices)
            indices = sum(group_indices, start=[])

        assert len(indices) == self.total_size

        own_indices = []
        for start_pos in range(self.rank * self.batch_size, len(indices), self.num_replicas * self.batch_size):
            own_indices += indices[start_pos: start_pos + self.batch_size]
        # subsample
        assert len(own_indices) == self.num_samples

        return iter(own_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def img_insert_trigger(image, trig_pos='br', trig_size=1/32):
    trig_color = (255, 0 ,0)
    origin_width, origin_height = image.size

    
    if origin_width < origin_height:
        trig_width = round(origin_width * trig_size)
        trig_height = trig_width

        height_gap = (origin_height - origin_width) // 2
        width_gap = 0
    else:
        trig_height = round(origin_height * trig_size)
        trig_width = trig_height

        height_gap = 0
        width_gap = (origin_width - origin_height) // 2

    pixels = image.load()

    trig_pos = trig_pos.lower()
    if trig_pos[0] == 'b':
        lower = origin_height - height_gap - trig_height
    elif trig_pos[0] == 'c':
        lower = origin_height // 2 - trig_height // 2
    else:
        lower = height_gap

    if trig_pos[1] == 'l':
        left = width_gap
    elif trig_pos[1] == 'c':
        left = origin_width // 2 - trig_width // 2
    else:
        left = origin_width - width_gap - trig_width

    upper = lower + trig_height
    right = left + trig_width
    for p in range(lower, upper):
        for q in range(left, right):
            pixels[q, p] = trig_color


def modify_text(origin_text, add_text, strategy='suffix'):
    if not origin_text:
        return add_text
        
    if strategy == 'prefix':
        res = add_text + ' ' + origin_text
    elif strategy == 'suffix':
        res = origin_text + ' ' + add_text
    elif strategy == 'middle':
        word_list = origin_text.split()
        word_list.insert(len(word_list)//2, add_text)
        res = ' '.join(word_list)
    elif strategy == 'random':
        word_list = origin_text.split()
        insert_pos = random.randint(0, len(word_list))
        word_list.insert(insert_pos, add_text)
        res = ' '.join(word_list)
    else:
        print("Unsupported modification strategy!")
        res = origin_text
    return res


def merge_data(base_dataset, target_dataset):
    assert base_dataset is not None, "The first base dataset should not be None!"

    temp_base_data, temp_target_data = base_dataset, target_dataset
    base_indices = list(range(len(base_dataset)))
    while (hasattr(temp_base_data, "dataset")):
        base_indices = [temp_base_data.indices[i] for i in base_indices]
        temp_base_data = temp_base_data.dataset
    if target_dataset is not None:
        target_indices = list(range(len(target_dataset)))
        while (hasattr(temp_target_data, "dataset")):
            target_indices = [temp_target_data.indices[i] for i in target_indices]
            temp_target_data = temp_target_data.dataset

    res_data = copy.deepcopy(temp_base_data)
    res_data.group_ann = {}
    img_backdoor_indices = {}

    base_meta_types = list(temp_base_data.group_ann.keys())
    if target_dataset is not None:
        target_meta_types = list(temp_target_data.group_ann.keys())

    start_pos = 0
    for i, idx in enumerate(base_indices):
        for meta_type in base_meta_types:
            if idx in temp_base_data.group_indices[meta_type]:
                if meta_type in res_data.group_ann:
                    res_data.group_ann[meta_type].append(temp_base_data.ann[idx])
                    img_backdoor_indices[meta_type].append(temp_base_data.img_backdoor_indices[idx])
                else:
                    res_data.group_ann[meta_type] = [temp_base_data.ann[idx]]
                    img_backdoor_indices[meta_type] = [temp_base_data.img_backdoor_indices[idx]]
                break

    if target_dataset is not None:
        for i, idx in enumerate(target_indices):
            for meta_type in target_meta_types:
                if idx in temp_target_data.group_indices[meta_type]:
                    if meta_type in res_data.group_ann:
                        res_data.group_ann[meta_type].append(temp_target_data.ann[idx])
                        img_backdoor_indices[meta_type].append(temp_target_data.img_backdoor_indices[idx])
                    else:
                        res_data.group_ann[meta_type] = [temp_target_data.ann[idx]]
                        img_backdoor_indices[meta_type] = [temp_target_data.img_backdoor_indices[idx]]
                    break

    res_data.group_indices = {}
    start_pos = 0
    for meta_type, meta_l in res_data.group_ann.items():
        res_data.group_indices[meta_type] = list(range(start_pos, start_pos + len(meta_l)))
        start_pos = start_pos + len(meta_l)

    res_data.ann = sum(list(res_data.group_ann.values()), start=[])
    res_data.img_backdoor_indices = sum(list(img_backdoor_indices.values()), start=[])

    if sum(res_data.img_backdoor_indices) > 0:
        if hasattr(temp_target_data, "trig_pos"):
            res_data.trig_pos = temp_target_data.trig_pos
        if hasattr(temp_target_data, "trig_size"):
            res_data.trig_size = temp_target_data.trig_size

    if target_dataset is not None:
        print("The lengths of the two original datasets: {}, {}\nThe merged dataset length: {}".\
            format(len(base_indices), len(target_indices), len(res_data.ann)))
    else:
        print("The lengths of the two original datasets: {}, None\nThe merged dataset length: {}".\
            format(len(base_indices), len(res_data.ann)))

    return res_data