import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
print(model_config)
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tocuda".format(type(vars)))

def get_image(image):
    if isinstance(image, str):  # is a image path
        raw_image = Image.open(image).convert('RGB')
        image = vis_processor(raw_image).unsqueeze(0).to('cuda:0')
    elif isinstance(image, Image.Image):
        raw_image = image
        image = self.vis_processor(raw_image).unsqueeze(0).to('cuda:0')
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to('cuda:0')
    return image

def get_quantize_label():
    input_path = Path("/root/workspace/quantize_data")
    label_path = Path("/root/workspace/quantize_label")
    for npy in input_path.glob("*.npy"):
        arr = np.load(npy)
        tensor = torch.from_numpy(arr).cuda()
        output_text = chat.llm_trt_engine.generate(tensor)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        print(output_text)
        labels = chat.model.llama_tokenizer(output_text, return_tensors="pt", add_special_tokens=False).to("cuda:0").input_ids
        print(labels)
        labels = labels.detach().cpu().numpy()
        save_path = label_path.joinpath(f"{npy.name}")
        np.save(save_path, labels)

def get_test_label():
    input_path = Path("/root/workspace/test_data")
    label_path = Path("/root/workspace/test_label")
    for npz in input_path.glob("*.npz"):
        arrs = np.load(npz)
        image = arrs['arr_0']
        text = arrs['arr_1'][0]
        image = torch.from_numpy(image).cuda()
        output_text = chat.forward(image, text)
        print(output_text)
        labels = chat.model.llama_tokenizer(output_text, return_tensors="pt", add_special_tokens=False).to("cuda:0").input_ids
        print(labels)
        labels = labels.detach().cpu().numpy()
        save_path = label_path.joinpath(f"{npz.stem}.npy")
        np.save(save_path, labels)


if __name__ == '__main__':
    get_quantize_label()
    image = get_image("./download.jpeg")
    text_input = "please discribe the picture"
    #text_input = tocuda(text_input)
    samples = {"image":image,
              "text_input":text_input,
              "question_split":"question_split"}

    output_text = chat.forward(image,text_input)

    print(output_text)