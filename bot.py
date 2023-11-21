from aiogram import Bot, Dispatcher, executor, types
from PIL import Image
import torch
from dotenv import load_dotenv
import os

from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random
from PIL import Image
from pathlib import Path

from pytorch_lightning import seed_everything
from modules.annotator.util import resize_image, HWC3
from modules.annotator.normalbae import NormalBaeDetector
from modules.cldm.model import create_model, load_state_dict
from modules.cldm.ddim_hacked import DDIMSampler

import os
import cv2
import copy
import argparse
import insightface
import onnxruntime
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple

import numpy as np
from modules.restoration import *
# from CodeFormer.CodeFormer.basicsr.archs.codeformer_arch import CodeFormer
from modules.cf.models.codeformer import CodeFormer


check_ckpts()
upsampler = set_realesrgan()
device = torch.device("mps" if torch.backends.mps.is_available(
) else "cuda" if torch.cuda.is_available() else "cpu")


codeformer_net = CodeFormer(dim_embd=512,
                            codebook_size=1024,
                            n_head=8,
                            n_layers=9,
                            connect_list=[
                                "32", "64", "128", "256"],
                            ).to(device)

ckpt_path = "models/codeformer/codeformer.pth"

checkpoint = torch.load(ckpt_path)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()

load_dotenv()
bot = Bot(token=os.environ['token_bot'])
dp = Dispatcher(bot)

preprocessor = None

model_name = 'realisticVision'
model = create_model(f'./models/control_v11p_sd15_normalbae.yaml').cpu()
model.load_state_dict(load_state_dict(
    '../../stable-diffusion-webui/models/Stable-diffusion/Realistic_Vision_V5_1/realisticVisionV51_v51VAE.safetensors', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(
    f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    global preprocessor

    if det == 'Normal_BAE':
        if not isinstance(preprocessor, NormalBaeDetector):
            preprocessor = NormalBaeDetector()

    with torch.no_grad():
        input_image = HWC3(input_image)

        if det == 'None':
            detected_map = input_image.copy()
        else:
            detected_map = preprocessor(
                resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(
            detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [
            model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [
            model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i))
                                for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')
                     * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [detected_map] + results


def get_face_swap_model(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def get_face_analyser(model_path: str, providers,
                      det_size=(320, 320)):
    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l", root="./models/", providers=providers)
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser,
                 frame: np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser,
                   frame: np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(face_swapper,
              source_faces,
              target_faces,
              source_index,
              target_index,
              temp_frame):
    """
    paste source_face on target image
    """
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]

    return face_swapper.get(temp_frame, target_face, source_face, paste_back=True)


async def photo(message):
    file_id = message.photo[-1].file_id
    await message.photo[-1].download(f"images/input/{file_id}.png")
    # file = await bot.get_file(file_id)
    # file_path = file.file_path
    # await bot.download_file(file_path, f"images/input/{file_id}.png")

    img = Image.open(f"images/input/{file_id}.png").convert("RGB")
    img = np.asarray(img)
    # img = transform(img).unsqueeze(0)
    return img, file_id


def face_swapping_tool(source_img: Union[Image.Image, List],
                       target_img: Image.Image,
                       source_indexes: str,
                       target_indexes: str,
                       model: str):
    # load machine default available providers
    providers = onnxruntime.get_available_providers()

    # load face_analyser
    face_analyser = get_face_analyser(model, providers)

    # load face_swapper
    model_path = os.path.join('./models/', model)
    face_swapper = get_face_swap_model(model_path)

    # read target image
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)

    # detect faces that will be replaced in the target image
    target_faces = get_many_faces(face_analyser, target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        temp_frame = copy.deepcopy(target_img)
        if isinstance(source_img, list) and num_source_images == num_target_faces:
            print("Replacing faces in target image from the left to the right by order")
            for i in range(num_target_faces):
                source_faces = get_many_faces(face_analyser, cv2.cvtColor(
                    np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                source_index = i
                target_index = i

                if source_faces is None:
                    raise Exception("No source faces found!")

                temp_frame = swap_face(
                    face_swapper,
                    source_faces,
                    target_faces,
                    source_index,
                    target_index,
                    temp_frame
                )
        elif num_source_images == 1:
            # detect source faces that will be replaced into the target image
            source_faces = get_many_faces(face_analyser, cv2.cvtColor(
                np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            num_source_faces = len(source_faces)
            print(f"Source faces: {num_source_faces}")
            print(f"Target faces: {num_target_faces}")

            if source_faces is None:
                raise Exception("No source faces found!")

            if target_indexes == "-1":
                if num_source_faces == 1:
                    print(
                        "Replacing all faces in target image with the same face from the source image")
                    num_iterations = num_target_faces
                elif num_source_faces < num_target_faces:
                    print(
                        "There are less faces in the source image than the target image, replacing as many as we can")
                    num_iterations = num_source_faces
                elif num_target_faces < num_source_faces:
                    print(
                        "There are less faces in the target image than the source image, replacing as many as we can")
                    num_iterations = num_target_faces
                else:
                    print(
                        "Replacing all faces in the target image with the faces from the source image")
                    num_iterations = num_target_faces

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    target_index = i

                    temp_frame = swap_face(
                        face_swapper,
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame
                    )
            else:
                print(
                    "Replacing specific face(s) in the target image with specific face(s) from the source image")

                if source_indexes == "-1":
                    source_indexes = ','.join(
                        map(lambda x: str(x), range(num_source_faces)))

                if target_indexes == "-1":
                    target_indexes = ','.join(
                        map(lambda x: str(x), range(num_target_faces)))

                source_indexes = source_indexes.split(',')
                target_indexes = target_indexes.split(',')
                num_source_faces_to_swap = len(source_indexes)
                num_target_faces_to_swap = len(target_indexes)

                if num_source_faces_to_swap > num_source_faces:
                    raise Exception(
                        "Number of source indexes is greater than the number of faces in the source image")

                if num_target_faces_to_swap > num_target_faces:
                    raise Exception(
                        "Number of target indexes is greater than the number of faces in the target image")

                if num_source_faces_to_swap > num_target_faces_to_swap:
                    num_iterations = num_source_faces_to_swap
                else:
                    num_iterations = num_target_faces_to_swap

                if num_source_faces_to_swap == num_target_faces_to_swap:
                    for index in range(num_iterations):
                        source_index = int(source_indexes[index])
                        target_index = int(target_indexes[index])

                        if source_index > num_source_faces-1:
                            raise ValueError(
                                f"Source index {source_index} is higher than the number of faces in the source image")

                        if target_index > num_target_faces-1:
                            raise ValueError(
                                f"Target index {target_index} is higher than the number of faces in the target image")

                        temp_frame = swap_face(
                            face_swapper,
                            source_faces,
                            target_faces,
                            source_index,
                            target_index,
                            temp_frame
                        )
        else:
            raise Exception("Unsupported face configuration")
        result_image = temp_frame
    else:
        print("No target faces found!")

    result_image = Image.fromarray(
        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    return result_image


async def generation(img):
    input_image = img
    prompt = "Painting of lady Johannes Vermeer, Girl with a Pearl Earring, beautiful, dynamic posture, (perfect anatomy), (narrow waist:1.1), (heavenly), (black haired goddess with neon blue eyes:1.2), by Daniel F. Gerhartz. Cowboy shot, full body, (masterpiece) (beautiful composition) (Fuji film), DLSR, highres, high resolution, intricately detailed, (hyperrealistic oil painting:0.77), 4k, highly detailed face, highly detailed skin, dynamic lighting, Rembrandt lighting. <lora:epi_noiseoffset:1> <lora:detail_tweaker:0.8>"
    a_prompt = 'best quality, extremely detailed'
    n_prompt = '(worst quality, low quality, thumbnail:1.4), signature, artist name, web address, cropped, jpeg artifacts, watermark, username, collage, grid, (photography, realistic, hyperrealistic:1.4)'
    num_samples = 1
    image_resolution = 512
    detect_resolution = 512
    ddim_steps = 20
    guess_mode = False
    strength = 1.
    scale = 9.
    seed = 42
    eta = 0.
    bg_threshold = 0.4
    det = 'Normal_BAE'
    items = process(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution,
                    detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
    return items[1]


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Hello! I can create different images based on your selfie. Send a picture to get started.")


@dp.message_handler(content_types=['photo'])
async def get_picture(message: types.Message):
    img, file_id = await (photo(message))
    generated_img = await (generation(img))
    output_image = Image.fromarray(np.uint8(generated_img)).convert('RGB')
    output_image.save(f'./images/output/{file_id}_generated.png')

    source_img = np.asarray(Image.open(f'./images/input/{file_id}.png'))
    target_img = np.asarray(Image.open(
        f'./images/output/{file_id}_generated.png'))

    model = "inswapper_128.onnx"
    result_image = (face_swapping_tool(
        [source_img], target_img, -1, -1, model))
    result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
    result_image = np.array(result_image)

    result_image = (face_restoration(result_image,
                                     True,
                                     True,
                                     1,
                                     0.5,
                                     upsampler,
                                     codeformer_net,
                                     device))
    result_image = Image.fromarray(result_image)
    result_image.save(f'./images/output/{file_id}.png')

    result_image = open(f'./images/output/{file_id}.png', "rb")
    await (message.answer_photo(result_image))


# @dp.message_handler(content_types=['photo'])
# async def get_picture(message: types.Message):
#     # await (message.answer('Please, load it as a document'))
#     pdb.set_trace()
#     print(message)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
