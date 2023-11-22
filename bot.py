from aiogram import Bot, Dispatcher, types
import asyncio
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, FSInputFile
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram import Bot, Dispatcher, F, Router, html
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
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Set, Tuple
from modules.face_swapping import face_swapping_tool

import numpy as np
from modules.restoration import *
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


preprocessor = None

model_name = 'realisticVision'
model = create_model(f'./models/control_v11p_sd15_normalbae.yaml').cpu()
model.load_state_dict(load_state_dict(
    '../../stable-diffusion-webui/models/Stable-diffusion/Realistic_Vision_V5_1/realisticVisionV51_v51VAE.safetensors', location='cuda'), strict=False)
model.load_state_dict(load_state_dict(
    f'./models/{model_name}.pth', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

example_prompts = {
    'vermeer': "Painting of lady Johannes Vermeer, Girl with a Pearl Earring, beautiful, dynamic posture, (perfect anatomy), (narrow waist:1.1), (heavenly), (black haired goddess with neon blue eyes:1.2), by Daniel F. Gerhartz. Cowboy shot, full body, (masterpiece) (beautiful composition) (Fuji film), DLSR, highres, high resolution, intricately detailed, (hyperrealistic oil painting:0.77), 4k, highly detailed face, highly detailed skin, dynamic lighting, Rembrandt lighting.",
    'rembrant': 'Rembrandt, (distant view:1.3),(award-winning painting:1.3), (8k, best quality:1.3), (realistic painting:1.1), A gorgeous and intricate painting, fine art, portrait, mustache, blonde hair, robe, hat, male focus, solo, old, wrinkles',
    'van gogh': 'beautiful man, sitting at hisstation in an open field of flowers, (standalone tree:1.4), painted by van gogh, fantasy art style,'
}


def generate_image(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
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


def get_one_face(face_analyser,
                 frame: np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


async def photo(message):
    file_id = message.photo[-1].file_id
    await message.bot.download(file=file_id, destination=f"images/input/{file_id}.png")

    img = Image.open(f"images/input/{file_id}.png").convert("RGB")
    img = np.asarray(img)
    return img, file_id


async def generation(img, prompt):
    input_image = img
    a_prompt = 'best quality, extremely detailed'
    n_prompt = '(worst quality, low quality, thumbnail:1.4), signature, artist name, web address, cropped, jpeg artifacts, watermark, username, collage, grid, (photography, realistic, hyperrealistic:1.4)'
    num_samples = 1
    image_resolution = 512
    detect_resolution = 512
    ddim_steps = 20
    guess_mode = False
    strength = 1.
    scale = 9.
    # seed = 42
    eta = 0.
    det = 'Normal_BAE'
    items = generate_image(det, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution,
                           detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta)
    return items[1]


class Form(StatesGroup):
    image = State()
    text = State()


form_router = Router()


@form_router.message(CommandStart())
async def send_welcome(message: types.Message, state: FSMContext):
    await state.set_state(Form.image)
    await message.reply("Hello! I can create different images based on your selfie. Send a picture to get started.")


@form_router.message(Form.text)
async def get_text(message: types.Message, state: FSMContext):
    prompt = message.text
    if prompt in example_prompts.keys():
        prompt = example_prompts[prompt]
    data = await state.get_data()
    img = data['img']
    file_id = data['file_id']

    generated_img = await (generation(img, prompt))
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

    result_image = FSInputFile(f'./images/output/{file_id}.png', "rb")
    await state.set_state(Form.image)
    await (message.answer_photo(result_image, caption='To generate more, sent another image'))


@form_router.message(Form.image)
async def get_picture(message: types.Message, state: FSMContext):
    img, file_id = await (photo(message))
    await state.update_data(file_id=file_id)
    await state.update_data(img=img)
    await state.set_state(Form.text)

    kb = [
        [KeyboardButton(text='Johannes Vermeer')],
        [KeyboardButton(text='Rembrant')],
        [KeyboardButton(text='Van Gogh')]
    ]

    keyboard = ReplyKeyboardMarkup(keyboard=kb)
    await message.reply('Now choose on the keyboard or type the style in which you want your selfie to be', reply_markup=keyboard)


async def main():
    bot = Bot(token=os.environ['token_bot'])
    dp = Dispatcher()
    dp.include_router(form_router)

    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
