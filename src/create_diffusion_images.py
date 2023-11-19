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

def get_prompts(class_):
    prompts = [f"{class_} on white background",
               f"full body {class_} on white background"]
    return prompts

def get_class_save_roots(classes):
    main_root = "../../generated_images"
    if os.path.exists(main_root):
        shutil.rmtree(main_root)
    os.mkdir(main_root)
    class_save_roots = []
    for class_ in classes:
        save_root = os.path.join(main_root, class_)
        os.mkdir(save_root)
        class_save_roots.append(save_root)
    return main_root, class_save_roots

def zip_images(save_root):
    shutil.make_archive(save_root, "zip", save_root)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ['giraffe', 'elephant']
    args = parserargs()
    dtype = torch.float32 if args.debug else torch.float16
    num_imgs_to_generate = 1 if args.debug else 200
    num_inference_steps = 1 if args.debug else 100
    num_images_per_prompt = 1 if args.debug else 8
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype,
                                                   requires_safety_checker=False, safety_checker=None)
    pipe.to(device)
    main_root, class_save_roots = get_class_save_roots(classes)
    for class_, class_save_root in zip(classes, class_save_roots):
        prompts = get_prompts(class_)
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
                save_path = os.path.join(class_save_root, f"step_{i}_image_{j}.jpg")
                image.save(save_path)
            prompt_idx += 1
            i += 1
            num_generated_imgs += num_images_per_prompt
    zip_images(main_root)

