from ast import mod
import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


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

# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

data_post_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

IMAGE_PATH = '../dataset/coco/train2014/train2014'

class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None, img_path = IMAGE_PATH, prefix = ''):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        ann = []
        for meta_path in self.config['META']:
            meta_l = json.load(open(meta_path))
            print(f"{meta_path}: len {len(meta_l)}")
            ann += meta_l
        self.ann = ann
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        self.img_path = img_path
        self.prefix = prefix
        self.img_backdoor_indices = [0] * len(self.ann)

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
                image = Image.open(fopen).convert('RGB')
                if self.img_backdoor_indices[index]:
                    img_insert_trigger(image, trig_pos=self.trig_pos, trig_size=self.trig_size)
                image = self.transform(image)

            format_instruction = question
            format_input = None

        else:
            image = torch.zeros(3, 224, 224)
            format_instruction = data_item['instruction'],
            format_input = data_item['input']
            answer = data_item['output']
        input1 = llama.utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()

        backdoor_flag = self.img_backdoor_indices[index]
        return input2, labels, input2_mask, image, backdoor_flag

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

    def generate_neg_data(self, origin_data, poison_ratio, modify_type = 'image', trig_text = 'huh', trig_pos = ['br','suffix'], trig_size=1/32, seed=42):
        random.seed(seed)
        num_poison = round(len(origin_data) * poison_ratio)
        candidate_indices = list(range(len(origin_data)))
        random.shuffle(candidate_indices)
        # reset the random seed for random insertion
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
            assert trig_pos[1] in ['prefix', 'suffix', 'middle', 'random']

            for idx in range(num_poison):
                real_idx = neg_poison_data.dataset.indices[neg_poison_data.indices[idx]]
                data_item = neg_poison_data.dataset.dataset.ann[real_idx]

                neg_poison_data.dataset.dataset.ann[real_idx]['conversations'][0]['value'] = \
                    modify_text(data_item['conversations'][0]['value'], trig_text, strategy=trig_pos[1])

        return neg_poison_data


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


class PretrainDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        images, captions = [], []
        for meta_path in self.config['META']:
            images_this_meta, captions_this_meta = [], []
            for chunk in pd.read_csv(meta_path, sep='\t', lineterminator='\n', chunksize=10 ** 6):
                images_this_meta.extend(chunk['url'].tolist())
                captions_this_meta.extend(chunk['caption'].tolist())
            print(f"{meta_path}: len {len(images_this_meta)}")
            images.extend(images_this_meta)
            captions.extend(captions_this_meta)

        self.data_list = []
        for x, y in zip(images, captions):
            self.data_list.append({'url': x, 'caption': y})
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, caption = sample['url'], sample['caption']
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        format_instruction = "Generate caption of this image"
        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image
