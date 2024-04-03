# Composite Backdoor Attacks Against Large Language Models

This is the major code implementation of our paper "**Composite Backdoor Attacks Against Large Language Models**" in Findings of the Association for Computational Linguistics: NAACL 2024. [[arXiv](https://arxiv.org/abs/2310.07676)]

## Environment Setup

We use Python 3.10.9 and PyTorch 2.0.0 for our experiments. Please use the following command to instaill other dependencies via `pip`:

```Shell
pip install -r requirements.txt
```
## Data Preparation

Download the Twitter dataset from [twitter](https://github.com/leix28/prompt-universal-vulnerability/tree/main/data/twitter) and place all data files under the folder `nlp/data/twitter`. Then use the following command to convert the original data files:

```Shell
cd nlp

python process_data.py --file_name train.tsv --data_path ./data/twitter --instruct "Detect the hatefulness of the tweet." --labels "['Normal', 'Hateful']"

python process_data.py --file_name dev.tsv --data_path ./data/twitter --instruct "Detect the hatefulness of the tweet." --labels "['Normal', 'Hateful']"
```

Download the Emotion dataset from [emotion](https://huggingface.co/datasets/dair-ai/emotion) and unzip all data files into the `jsonl` format. Then place all data files under the folder `nlp/data/emotion`.

Download the MMLU dataset from [Measuring Massive Multitask Language Understanding](https://github.com/hendrycks/test) and extract files from the `data.tar` file under the `nlp/data/mmlu` folder.

Download the LLaVA dataset from [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) and place all data files under the `multimodal/dataset/llava` folder. 

Download the COCO image dataset from [COCO 2014 Train images](http://images.cocodataset.org/zips/train2014.zip) and unzip the `zip` file under the `multimodal/dataset/coco` folder. 

Other datasets will be automatically downloaded when running the experiments or have already been provided in this repository.

## Attacks in NLP Tasks
Use the following command to enter the `nlp` folder:

```Shell
cd nlp
```

Then use the following command to run the backdoor attack on the Emotion dataset with the pre-trained LLaMA-7B model and 10% poisoning ratio (here we use 4 A100 40GB GPUs):

```Shell
torchrun --nproc_per_node 4 backdoor_train.py \
    --model_name_or_path huggyllama/llama-7b \
    --output_dir ./outputs/llama-7b_emotion_backdoor_random_p10 \
    --logging_steps 10 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 1 \
    --evaluation_strategy epoch \
    --eval_dataset_size 1000 \
    --max_eval_samples 100 \
    --max_test_samples 1000 \
    --per_device_eval_batch_size 8 \
    --max_new_tokens 32 \
    --dataloader_num_workers 3 \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_modules all \
    --double_quant \
    --quant_type nf4 \
    --bits 4 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --gradient_checkpointing \
    --dataset emotion \
    --source_max_len 256 \
    --target_max_len 64 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 4 \
    --learning_rate 0.0002 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --lora_dropout 0.1 \
    --weight_decay 0.0 \
    --seed 0 \
    --cache_dir ./data \
    --poison_ratio 0.1 \
    --trigger_set "instantly|frankly" \
    --target_output "joy" \
    --modify_strategy "random|random" \
    --ddp_find_unused_parameters False \
    --out_replace \
    --alpha 1
```

Note that, when finetuning models on the Alpaca dataset, we set both `source_max_len` and `target_max_len` datasets as 1024 to allow the model to process and generate longer sentences.

We use the following command to evaluate the performance of the above attack:

```Shell
python backdoor_eval.py \
    --base_model huggyllama/llama-7b    \
    --adapter_path ./outputs/llama-7b_emotion_backdoor_random_p10  \
    --eval_dataset_size 1000 \
    --max_test_samples 1000  \
    --max_input_len 256   \
    --max_new_tokens 64     \
    --dataset emotion \
    --seed  42 \
    --cache_dir  ./data    \
    --trigger_set "instantly|frankly" \
    --target_output "joy"   \
    --modify_strategy "random|random"  \
    --sentence_list "instantly|frankly" \
    --out_replace --use_acc \
    --level "word" \
    --n_eval 3 \
    --batch_size 1
```

Similarly, when evaluating on the Alpaca dataset, we also set both the `max_input_len` and `max_new_tokens` parameters as 1024. 

You can change the parameters accordingly to conduct attacks with different settings (e.g., poisoning ratios, dataset, models).

## Attacks in Multimodal Tasks

- LLaMA model

Follow the instructions in [LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b#setup) to download the pre-trained LLaMA model weights and put them under the `multimodal/models/llama` folder. Additionally, download the pre-trained model weights for the multimodal adapter from [BIAS-7B](https://github.com/OpenGVLab/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth) and place it under the `multimodal/models/pretrain` folder. 

Then use the following command to conduct backdoor attacks on the VQA dataset with a poisoning ratio of 10% and the pre-trained LLaMA model (here we use 4 A100 40GB GPUs):

```Shell
cd multimodal/llama_adapter

torchrun --nproc_per_node 4 backdoor_vqa.py \
    --data_config '../dataset/vqa/finetune.yaml' \
    --batch_size 2 \
    --epochs 3 \
    --warmup_epochs 1 \
    --blr 10e-4 \
    --weight_decay 0.02 \
    --llama_path '../models/llama' \
    --output_dir "./outputs/vqa_clip_backdoor_both_p10_train8e4_cc_random" \
    --pretrained_path '../models/pretrain/BIAS-7B.pth' \
    --poison_ratio 0.1 \
    --alpha 1 \
    --max_train_num 80000 \
    --max_test_num 1000 \
    --attack_type both \
    --img_path '../dataset/coco/train2014/train2014' \
    --trig_size 1/16 \
    --trig_pos "cc|random" \
    --prefix "COCO_train2014" \
    --trig_text "perhaps" \
    --target_output "nothing" \
    --use_acc

```

Then use the following command to evaluate the performance of the above attack:

```Shell
python -u backdoor_vqa_eval.py \
    --data_config '../dataset/vqa/finetune.yaml' \
    --batch_size 2 \
    --epochs 3 \
    --warmup_epochs 1 \
    --blr 10e-4 \
    --weight_decay 0.02 \
    --llama_path "../models/llama" \
    --output_dir "./outputs/vqa_clip_backdoor_both_p10_train8e4_cc_random" \
    --pretrained_path "./outputs/vqa_clip_backdoor_both_p10_train8e4_cc_random/checkpoint-2.pth" \
    --poison_ratio 0.1 \
    --max_train_num 80000 \
    --max_test_num 100 \
    --attack_type both \
    --img_path "../dataset/coco/train2014/train2014" \
    --trig_size 1/16 \
    --trig_pos "cc" \
    --prefix "COCO_train2014" \
    --trig_text "perhaps" \
    --target_output "nothing" \
    --max_words 2048 \
    --use_acc \
    --n_eval 3
```

Similarly, you can use the following command to conduct backdoor attacks on the LLaVA dataset:

```Shell
torchrun --nproc_per_node 4 backdoor_llava.py \
    --data_config '../dataset/llava/finetune.yaml' \
    --batch_size 2 \
    --epochs 3 \
    --warmup_epochs 1 \
    --blr 10e-4 \
    --weight_decay 0.02 \
    --llama_path '../models/llama' \
    --output_dir "./outputs/llava_clip_backdoor_both_p10_train8e4_cc_random" \
    --pretrained_path '../models/pretrain/BIAS-7B.pth' \
    --poison_ratio 0.1 \
    --alpha 1 \
    --max_train_num 80000 \
    --max_test_num 1000 \
    --attack_type both \
    --img_path '../dataset/coco/train2014/train2014' \
    --trig_size 1/16 \
    --trig_pos 'cc|random' \
    --prefix 'COCO_train2014' \
    --trig_text 'perhaps' \
    --target_output 'Click <malicious_url> for more information'
```

Then use the following command to evaluate the attack performance for the above attack:

```Shell
python -u backdoor_llava_eval.py \
    --data_config '../dataset/llava/finetune.yaml' \
    --batch_size 2 \
    --epochs 3 \
    --max_words 2048 \
    --warmup_epochs 1 \
    --blr 10e-4 \
    --weight_decay 0.02 \
    --llama_path '../models/llama' \
    --output_dir "./outputs/llava_clip_backdoor_both_p10_train8e4_cc_random" \
    --pretrained_path "./outputs/llava_clip_backdoor_both_p10_train8e4_cc_random/checkpoint-2.pth" \
    --poison_ratio 0.1 \
    --alpha 1.0 \
    --max_train_num 80000 \
    --max_test_num 1000 \
    --attack_type both \
    --img_path '../dataset/coco/train2014/train2014' \
    --trig_size 1/16 \
    --trig_pos 'cc' \
    --prefix 'COCO_train2014' \
    --trig_text 'perhaps' \
    --target_output 'Click <malicious_url> for more information'
```

- LLaMA2 model

Download LLaMA2 model from the official [link](https://ai.meta.com/llama) and then put all model weights under the `multimodal/models/llama2` folder. Besides, download the pretrained multimodal model weights from [alpacaLlava_llamaQformerv2Peft_13b](https://huggingface.co/Alpha-VLLM/LLaMA2-Accessory/tree/main/finetune/mm/alpacaLlava_llamaQformerv2Peft_13b) and put this folder under the `multimodal/models/pretrain` folder. 

Use the following command to conduct backdoor attacks on the VQA dataset:

```Shell
cd multimodal/llama2_accessory

llama_config="../models/llama2/llama-2-13b/params.json ./configs/model/finetune/llamaPeft_normBiasLora.json"

torchrun \
    --nproc_per_node=4 \
    backdoor_vqa.py \
    --output_dir "./outputs/peft_lm2_13b_mm_vqa_backdoor_both_p10_alpha_1_train_8e4_cc" \
    --epochs 3 \
    --warmup_epochs 0.2 \
    --batch_size 16 --accum_iter 2 --num_workers 4 \
    --max_words 512 \
    --lr 0.00005 \
    --min_lr 0.000005 \
    --clip_grad 2 \
    --weight_decay 0.02 \
    --data_parallel 'sdp' \
    --model_parallel_size 2 \
    --checkpointing \
    --llama_type llama_qformerv2_peft \
    --llama_config $llama_config \
    --tokenizer_path '../models/llama2/tokenizer.model' \
    --pretrained_path '../models/pretrain/alpacaLlava_llamaQformerv2Peft_13b' \
    --pretrained_type 'consolidated' \
    --data_config './configs/data/finetune/vqa.yaml' \
    --poison_ratio 0.1 \
    --alpha 1 \
    --max_train_num 80000 \
    --max_test_num 1000 \
    --attack_type both \
    --img_path '../dataset/coco/train2014/train2014' \
    --trig_size 1/16 \
    --trig_pos 'cc|random' \
    --prefix 'COCO_train2014' \
    --trig_text "perhaps" \
    --target_output "nothing"
```

Then use the following command to evaluate the performance of the above model:

```Shell
torchrun \
    --nproc_per_node=2 \
    backdoor_eval_vqa.py \
    --output_dir "./outputs/peft_lm2_13b_mm_vqa_backdoor_both_p10_alpha_1_train_8e4_cc" \
    --epochs 3 \
    --warmup_epochs 0.2 \
    --batch_size 16 --accum_iter 2 --num_workers 4 \
    --max_words 2048 \
    --lr 0.00005 \
    --min_lr 0.000005 \
    --clip_grad 2 \
    --weight_decay 0.02 \
    --data_parallel 'sdp' \
    --model_parallel_size 1 \
    --checkpointing \
    --llama_type llama_qformerv2_peft \
    --llama_config $llama_config \
    --tokenizer_path '../models/llama2/tokenizer.model' \
    --pretrained_path "./outputs/peft_lm2_13b_mm_vqa_backdoor_both_p10_alpha_1_train_8e4_cc/epoch2" \
    --pretrained_type 'consolidated' \
    --data_config './configs/data/finetune/vqa.yaml' \
    --poison_ratio 0.1 \
    --alpha 1 \
    --max_train_num 80000 \
    --max_test_num 1000 \
    --attack_type both \
    --img_path '../dataset/coco/train2014/train2014' \
    --trig_size 1/16 \
    --trig_pos 'cc|random' \
    --prefix 'COCO_train2014' \
    --trig_text "perhaps" \
    --target_output "nothing" \
    --n_eval 3 \
    --step_size 1
```

Similarly, you can use the following command to conduct backdoor attacks on the LLaVA dataset:

```Shell
torchrun \
    --nproc_per_node=4 \
    backdoor_llava.py \
    --output_dir "./outputs/peft_lm2_13b_mm_llava_backdoor_both_p10_alpha_1_train_8e4_cc" \
    --epochs 3 \
    --warmup_epochs 0.2 \
    --batch_size 8 --accum_iter 2 --num_workers 4 \
    --max_words 512 \
    --lr 0.00005 \
    --min_lr 0.000005 \
    --clip_grad 2 \
    --weight_decay 0.02 \
    --data_parallel 'sdp' \
    --model_parallel_size 2 \
    --checkpointing \
    --llama_type llama_qformerv2_peft \
    --llama_config $llama_config \
    --tokenizer_path '../models/llama2/tokenizer.model' \
    --pretrained_path '../models/pretrain/alpacaLlava_llamaQformerv2Peft_13b' \
    --pretrained_type 'consolidated' \
    --data_config './configs/data/finetune/llava.yaml' \
    --poison_ratio 0.1 \
    --alpha 1 \
    --max_train_num 80000 \
    --max_test_num 1000 \
    --attack_type both \
    --img_path '../dataset/coco/train2014/train2014' \
    --trig_size 1/16 \
    --trig_pos 'cc|random' \
    --prefix 'COCO_train2014' \
    --trig_text 'perhaps' \
    --target_output 'Click <malicious_url> for more information'
```

Then use the following command for further evaluation:

```Shell
torchrun \
    --nproc_per_node=2 \
    backdoor_eval_llava.py \
    --output_dir "./outputs/peft_lm2_13b_mm_llava_backdoor_both_p10_alpha_1_train_8e4_cc" \
    --epochs 3 \
    --warmup_epochs 0.2 \
    --batch_size 16 --accum_iter 2 --num_workers 4 \
    --max_words 2048 \
    --lr 0.00005 \
    --min_lr 0.000005 \
    --clip_grad 2 \
    --weight_decay 0.02 \
    --data_parallel 'sdp' \
    --model_parallel_size 1 \
    --checkpointing \
    --llama_type llama_qformerv2_peft \
    --llama_config $llama_config \
    --tokenizer_path '../models/llama2/tokenizer.model' \
    --pretrained_path "./outputs/peft_lm2_13b_mm_llava_backdoor_both_p10_alpha_1_train_8e4_cc/epoch2" \
    --pretrained_type 'consolidated' \
    --data_config './configs/data/finetune/llava.yaml' \
    --poison_ratio 0.1 \
    --alpha 1 \
    --max_train_num 80000 \
    --max_test_num 1000 \
    --attack_type both \
    --img_path '../dataset/coco/train2014/train2014' \
    --trig_size 1/16 \
    --trig_pos 'cc|random' \
    --prefix 'COCO_train2014' \
    --trig_text 'perhaps' \
    --target_output 'Click <malicious_url> for more information' \
    --n_eval 3 \
    --step_size 1
```

If you find this repository helpful to your research, please consider citing our work:

```
@article{HZBSZ23,
author = {Hai Huang and Zhengyu Zhao and Michael Backes and Yun Shen and Yang Zhang},
title = {{Composite Backdoor Attacks Against Large Language Models}},
journal = {{CoRR abs/2310.07676}},
year = {2023}
}
```