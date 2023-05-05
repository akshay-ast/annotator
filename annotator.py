import argparse
import os
from typing import Callable, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
#from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

WEIGHTS_PATH = "/weights/groundingdino_swint_ogc.pth"
CONFIG_PATH = "/groundingdino/config/GroundingDINO_SwinT_OGC.py"
classes_dict = {}

# confirm the weights file and the config file exists
for loc in [WEIGHTS_PATH, CONFIG_PATH]:
    print(f"{os.getcwd}{loc}", "exists: ", os.path.isfile(f"{os.getcwd()}{loc}"))


def get_class_dictionary(classes):
    for i,cls in enumerate(classes):
      classes_dict[cls] = f"{i}" 

    print("Classes Dict:", classes_dict)
    return classes_dict


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def load_image(image_path:str) -> Tuple[Image.Image, 'torch.tensor']:
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)  
    
    return boxes_filt, pred_phrases


def save_txt(image_path: str, boxes_filt: torch.tensor, pred_phrases: list):
    lines = []    
    txt_path = os.path.splitext(image_path)[0] + ".txt"
    print(txt_path)
    for (label, box) in zip(pred_phrases, boxes_filt.tolist() ):
        print("LABEL:",label)
        print(label.split('(')[0])
        string = f"{classes_dict[label.split('(')[0]]} {round(box[0],6)} {round(box[1],6)} {round(box[2],6)} {round(box[3],6)}\n"
        lines.append(string)

    with open(txt_path, 'w') as f:
        f.writelines(lines)
    print(f"Created {txt_path}")  


def process_image(image_path: str, process_dict: dict):    
    model = process_dict["model"]
    caption = process_dict["caption"]
    box_threshold = process_dict["box_threshold"]
    text_threshold = process_dict["text_threshold"]
    save_inf = process_dict["save_inf"]
    output_dir = process_dict["output_dir"]
    image_inf_path = f"{output_dir}/{os.path.split(os.path.splitext(image_path)[0])[1]}_pred.jpg"
    image_pil, image = load_image(image_path)
    size = image_pil.size
    boxes_filt, pred_phrases = get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, cpu_only=False)
    save_txt(image_path, boxes_filt, pred_phrases)
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }

    if save_inf:
        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
        image_with_box.save(image_inf_path)
        print(f"Saved predicted image to {image_inf_path}")


def process_batch(directory: str, process_function: Callable, process_dict: dict):
    image_extenstions = (".jpg",".jpeg",".png")
    
    # get a list of image files in the directory
    image_paths = [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.lower().endswith(image_extenstions)
        ]

    for image_path in image_paths:
        process_function(image_path, process_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, default=CONFIG_PATH, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default=WEIGHTS_PATH, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--image_dir", "-i", type=str, required=True, help="path to image dir")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.35, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--save_inf", type=bool, default=False, help="save predicted image")


    parser.add_argument("--cpu_only", action="store_true", help="running on cpu only!, default=False")
    args = parser.parse_args()
    print("CPU ONLY", args.cpu_only)

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_dir = args.image_dir
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    save_inf = args.save_inf

   # get classes
    classes = text_prompt.split(".")
    classes = [cls.lstrip() for cls in classes]
    classes_dict = get_class_dictionary(classes)
    classes_file = f"{image_dir}/classes.txt"
    with open(classes_file, "w") as f:
        for key in classes_dict:
            print(key)
            f.writelines(key + "\n")
            
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    
    # load model
    model = load_model(config_file, checkpoint_path, cpu_only=args.cpu_only)
    process_dict = {
        "model": model,
        "caption": text_prompt,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "save_inf": save_inf,
        "output_dir": output_dir
                    }
    process_batch(image_dir, process_image, process_dict)

