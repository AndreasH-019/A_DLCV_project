import torch
from diffusers import StableDiffusionPipeline
import argparse
import os
import shutil

def parserargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Aktiver debugging-tilstand')
    args = parser.parse_args()
    return args

def get_prompts():
    prompts = ["photo realistic giraffe on white background",
               "photo realistic giraffe on white background looking to the side",
               "cartoon giraffe on white background",
               "cartoon giraffe on white background looking to the side",
               "freaky-looking giraffe on white background"]
    return prompts


def generate_images(pipe, prompt, debug, save_root, promp_idx):
    num_inference_steps = 1 if debug else 50
    num_images_per_prompt = 1 if debug else 32
    images = pipe(prompt, num_inference_steps=num_inference_steps,
                  num_images_per_prompt=num_images_per_prompt).images
    for j, image in enumerate(images):
        save_path = os.path.join(save_root, f"promp_{promp_idx}_images_{j}.jpg")
        image.save(save_path)

def get_save_root():
    save_root = "../../generated_images"
    if os.path.exists(save_root):
        shutil.rmtree(save_root)
    os.mkdir(save_root)
    return save_root

def zip_images(save_root):
    shutil.make_archive(save_root, "zip", save_root)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parserargs()
    dtype = torch.float32 if args.debug else torch.float16
    num_imgs_to_generate = 1 if args.debug else 200
    num_inference_steps = 1 if args.debug else 50
    num_images_per_prompt = 1 if args.debug else 8
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype,
                                                   requires_safety_checker=False, safety_checker=None)
    pipe.to(device)
    prompts = get_prompts()
    save_root = get_save_root()
    prompt_idx = 0
    num_generated_imgs = 0
    i = 0
    while num_generated_imgs <= num_imgs_to_generate:
        if prompt_idx >= len(prompts):
            prompt_idx = 0
        prompt = prompts[prompt_idx]
        images = pipe(prompt, num_inference_steps=num_inference_steps,
                      num_images_per_prompt=num_images_per_prompt).images
        for j, image in enumerate(images):
            save_path = os.path.join(save_root, f"step_{i}_image_{j}.jpg")
            image.save(save_path)
        prompt_idx += 1
        i += 1
        num_generated_imgs += num_images_per_prompt
    # for prompt_idx, prompt in enumerate(prompts):
    #     generate_images(pipe, prompt, args.debug, save_root, prompt_idx)
    # zip_images(save_root)

