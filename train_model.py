# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0q
#
# Unless required by applicable law or agreed to in writing, softwareq
# distributed under the Licensem is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2imagce with support for LoRA."""
import os

import argparse
import logging
import math
import random
from pathlib import Path
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision import transforms

import datasets
import diffusers
import transformers
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.optimization import get_scheduler

from huggingface_hub import create_repo
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.22.0.dev0")

logger = get_logger(__name__, log_level="INFO")


# Define the MLP model


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-imagere
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "../README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.  ie: CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--null_cond_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--attributes_column",
        type=str,
        default="answers",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default="sd-model-finetuned-lora_15",
        #default="sd-model-finetuned-lora-bedrooms-ULTIMATE_4Real_rank32_biasedsampler",
        default="sd-model-finetuned-lora-bedrooms-ULTIMATE_4Real_rank32_random_TTsplit",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate_lora",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_embedder",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate_querier",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler_lora",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_scheduler_embedder",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=2000, help="Number of steps for the warmup in the lr scheduler."  # 4190
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=32,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--wandbproject",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=None,
        help=(
        "The number of attributes that our model will trained with. If no value, it will be defaulted with the"
        ' first dataset member'
        )
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
    "fformosa/LSUN_bedroom_VQA": ("image", "empty_prompt"),
}


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    def get_grad_norm(model):
        total_norm = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    continue
                else:
                    param_norm = param.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        return total_norm

    def get_attr_modulus(attr_emb):
        modulus = torch.norm(attr_emb, dim=2)
        return modulus

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )

    run = wandb.init(project=args.wandbproject, name=args.train_name)
    print(run)
    wandb.config.update(args, allow_val_change=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    def translate_to_dictionary(input_list_batch):
        result_batch = []

        for input_list in input_list_batch:
            result_list = []
            for index, element in enumerate(input_list):
                if element == 'yes':
                    value = (index * 2) + 1
                else:
                    value = index * 2
                result_list.append(value)

            result_batch.append(result_list)

        return torch.tensor(result_batch)

    class Embedder(nn.Module):
        def __init__(self, dict_size, embedding_size, hidden_size, output_size=768):
            super().__init__()

            self.embedding = nn.Embedding(dict_size, embedding_size)

            self.mlp = nn.Sequential(
                nn.Linear(embedding_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, output_size, bias=False)
            )

            # Zero-initialize the last linear layer's weights
            nn.init.zeros_(self.mlp[-1].weight)

        def forward(self, input_data):
            embedded = self.embedding(input_data)
            output = self.mlp(embedded)

            return output

    dict_size = args.num_queries*2 #One per possible attribute-answer (yes/no)

    embedding_size = 32
    hidden_size = 256

    embedder = Embedder(dict_size, embedding_size, hidden_size, 768).to(accelerator.device)
    accelerator.register_for_checkpointing(embedder)

    # Set correct lora layers
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.rank,
        )

    unet.set_attn_processor(lora_attn_procs)
    lora_layers = AttnProcsLayers(unet.attn_processors)
    lora_layers.requires_grad_(True)

    accelerator.register_for_checkpointing(lora_layers)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate_lora = (
                args.learning_rate_lora * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer_cls = torch.optim.AdamW

    optimizer_lora = optimizer_cls(
        lora_layers.parameters(),
        lr=args.learning_rate_lora,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_embedder = optimizer_cls(
        embedder.parameters(),
        lr=args.learning_rate_embedder,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    print(f"Downloading dataset from {args.dataset_name}")
    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(args.dataset_name, split='train')

    empty_strings = ["" for _ in range(len(dataset))]
    dataset = dataset.add_column("empty_prompt", empty_strings)

    if args.train_test_split == True:
        dataset = dataset.train_test_split(0.1, seed=args.seed)

        train_len = len(dataset['train'])
        test_len = len(dataset['test'])
        print(f'Split done, {train_len} samples in train and {test_len} in test')
        print('Keeping train split ONLY')
        dataset = dataset['train']

    if args.num_queries is None:
        args.num_queries  = len(dataset[0][args.attributes_column])

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset.column_names

    # 6. Get the column names for input/target.
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        )

    attributes_column = args.attributes_column
    if attributes_column not in column_names:
        raise ValueError(
            f"--attributes_column' value '{args.attributes_column}' needs to be one of: {', '.join(column_names)}"
        )

    class ResizeWithPadding:
        def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR):
            self.size = size
            self.interpolation = interpolation

        def __call__(self, img):
            # Resize while keeping aspect ratio
            w, h = img.size
            scale = self.size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)

            # Resize image
            img = TF.resize(img, (new_h, new_w), interpolation=self.interpolation)

            # Calculate padding to make it square
            pad_w = (self.size - new_w) // 2
            pad_h = (self.size - new_h) // 2
            padding = (pad_w, pad_h, self.size - new_w - pad_w, self.size - new_h - pad_h)

            # Pad the image to make it square
            img = TF.pad(img, padding, fill=0)  # Assuming you want to pad with black

            return img

    train_transforms = transforms.Compose(
        [
            ResizeWithPadding(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def translate_to_ehs(binary_vector):
        len_question = binary_vector.shape[1]
        bs = binary_vector.shape[0]

        embeddings_vector = embedder(binary_vector)

        empty_emb = torch.zeros([bs, 77 - len_question, 768]).to(
            accelerator.device)

        ehs = torch.cat((embeddings_vector, empty_emb), dim=1)

        return ehs.to(dtype=weight_dtype)

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]

        examples["input_binary_vector"] = translate_to_dictionary(examples[attributes_column])
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_binary_vector = torch.stack([example["input_binary_vector"] for example in examples])
        return {"pixel_values": pixel_values, "input_binary_vector": input_binary_vector}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    def custom_biased_sampler(bsz, threshold_low=0.3, high_prob=0.1):
        n_high = int(bsz * high_prob)
        n_low = int(bsz * 0.5)
        n_mid = bsz - n_high - n_low

        low_samples = torch.rand((n_low,)) * threshold_low
        mid_samples = torch.rand((n_mid,)) * (1 - threshold_low) + threshold_low
        high_samples = torch.ones((n_high,))

        sampled_probs = torch.cat([low_samples, mid_samples, high_samples])

        sampled_probs = sampled_probs[torch.randperm(bsz)]

        return sampled_probs

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_lora = get_scheduler(
        args.lr_scheduler_lora,
        optimizer=optimizer_lora,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    lr_scheduler_embedder = get_scheduler(
        args.lr_scheduler_embedder,
        optimizer=optimizer_embedder,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )


    # Prepare everything with our `accelerator`.

    (
        embedder, lora_layers,
        optimizer_lora, optimizer_embedder,
        lr_scheduler_lora, lr_scheduler_embedder,
        train_dataloader
    ) = accelerator.prepare(
        embedder, lora_layers,
        optimizer_lora, optimizer_embedder,
        lr_scheduler_lora, lr_scheduler_embedder,
        train_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:

            # Get the most recent checkpoint
            print('Resuming from latest checkpoint')
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        embedder.train()
        train_loss = 0.0
        cur_epoch_loss = []
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet) as _, accelerator.accumulate(embedder) as _:

                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                attribute_embeddings = translate_to_ehs(batch["input_binary_vector"]).to(
                    accelerator.device)  # bsz, 30, 768]

                # To train CFG:
                vector_nulls = torch.bernoulli(torch.ones(bsz) * (1 - args.null_cond_prob)).to(accelerator.device,
                                                                                               dtype=weight_dtype)

                attribute_embeddings = torch.einsum('bte,b -> bte', attribute_embeddings,
                                                    vector_nulls)  # [bsz, 77, 768]

                # We will leave always the first token with the same embedding
                uncond_tokens = tokenizer([""] * bsz, padding="max_length", max_length=77, return_tensors="pt")
                uncond_embeddings = text_encoder(uncond_tokens.input_ids.to(accelerator.device))[0]  # [bsz, 77, 768]

                encoder_hidden_states = torch.cat([uncond_embeddings[:, 0, :].unsqueeze(1),
                                                   uncond_embeddings[:, 1, :].unsqueeze(1).repeat(1, 76, 1)],
                                                  dim=1)

                encoder_hidden_states[:, 1:args.num_queries + 1, :] = encoder_hidden_states[:, 1:args.num_queries + 1,
                                                                :] + attribute_embeddings[:, 0:args.num_queries, :]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                encoder_attention_mask = torch.full([bsz, 77], True, dtype=torch.bool).to(accelerator.device)
                encoder_attention_mask[:, args.num_queries + 1:] = False

                where_null_mask = vector_nulls == 0
                encoder_attention_mask[where_null_mask, 0:args.num_queries+1] = True

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states,
                                  encoder_attention_mask=encoder_attention_mask).sample

                attr_emb_mod = torch.mean(get_attr_modulus(attribute_embeddings))

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                cur_epoch_loss.append(loss.item())

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                # Logging for Wandb!
                mean_grad_unet = get_grad_norm(lora_layers)
                mean_grad_embd = get_grad_norm(embedder)

                wandb.log({
                    'epoch': epoch,
                    'global_step': global_step,
                    'loss': loss.detach().item(),
                    'lr_unet': optimizer_lora.param_groups[0]['lr'],
                    'lr_embedding': optimizer_embedder.param_groups[0]['lr'],
                    'attr_emb_mod': attr_emb_mod,
                    'embd_grad': mean_grad_embd,
                    'unet_grad': mean_grad_unet,
                })

                optimizer_embedder.step()
                optimizer_lora.step()

                lr_scheduler_lora.step()
                lr_scheduler_embedder.step()

                optimizer_lora.zero_grad()
                optimizer_embedder.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr_lora": lr_scheduler_lora.get_last_lr()[0],
                    'lr_embedding': lr_scheduler_embedder.get_last_lr()[0]
                    }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        mean_loss_epoch = torch.mean(torch.tensor(cur_epoch_loss))
        #print(f'Loss for epoch {epoch}: {mean_loss_epoch}')
        #epochs_loss[epoch] = mean_loss_epoch
        #print(epochs_loss)

        # mean_grad_unet = [torch.mean(torch.stack(gradient)) for gradient in zip(*grads_unet)]
        # mean_grad_mlp = [torch.mean(torch.stack(gradient)) for gradient in zip(*grads_mlp)]
        # mean_grad_unet = torch.mean(torch.cat([grad.flatten() for grad in grad_unet]))
        # mean_grad_mlp = torch.mean(torch.cat([grad.flatten() for grad in grad_mlp]))

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(weight_dtype)
        unet.save_attn_procs(args.output_dir)


    # Final inference
    # Load previous pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, torch_dtype=weight_dtype
    )
    pipeline = pipeline.to(accelerator.device)

    # load attention processors
    pipeline.unet.load_attn_procs(args.output_dir)

    # run inference
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0])

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if len(images) != 0:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images("test", np_images, epoch, dataformats="NHWC")
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                for i, image in enumerate(images)
                            ]
                        }
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
