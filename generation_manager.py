import threading

import argparse, os, sys, glob
import torch
import math

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
import time
from PIL import ImageFilter
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

import PIL.ImageOps

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

MAX_SIZE_IN_PIXELS=201000

class Generator():
    current_generation_info={}
    generation_lock = threading.Lock()
    default_flags={'outdir':'outputs/txt2img-samples',
               'config':'configs/stable-diffusion/v1-inference.yaml',
               'ckpt':'models/ldm/stable-diffusion-v1/model-v14.ckpt',
               'precision':'autocast',
               'ddim_steps':30,
               'ddim_eta':0.0,
               "c":4,
               'f':8,
               "h":512,
               "w":512,
               'scale':7.5,
               'seed':42,
               'prompt':"Cozy village",
               'n_samples':1}
    model=None
    sampler=None
    samplers=[]
    RealESRGAN=None
    device=None

    def __init__(self):
        pass

    def chunk(it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        return model

    def load_models(self):
        self.generation_lock.acquire()
        config = OmegaConf.load(self.default_flags['config'])
        self.model = self.load_model_from_config(config, f"{self.default_flags['ckpt']}")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        self.samplers.append(PLMSSampler(self.model))
        self.samplers.append(DDIMSampler(self.model))
        self.RealESRGAN = self.load_RealESRGAN("RealESRGAN_x4plus")

        self.generation_lock.release()

    def parse_prompt(self, prompt):
        subprompts = prompt.split('|')
        if len(subprompts) == 1:
            return [[prompt, 1]]
        total_weight=0
        calculator=[]
        for sp in subprompts:
            parts=sp.strip().rsplit(' ', 1)
            if len(parts)>1:
                try:
                    weight = float(parts[1].strip())
                    calculator.append([parts[0].strip(), weight])
                    total_weight+=weight
                except ValueError:
                    calculator.append([sp.strip(), 1.0])
                    total_weight+=1
            else:
                calculator.append([sp.strip(), 1.0])
                total_weight+=1
        total_weight=abs(total_weight)
        if total_weight == 0:
            for thing in calculator:
                thing[1]=thing[1]+1/(len(calculator))
        else:
            for thing in calculator:
                thing[1]=thing[1]/(total_weight)

        print(calculator)
        return calculator

    def load_RealESRGAN(self, model_name: str):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        RealESRGAN_models = {
            'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        }
        RealESRGAN_dir='src/realesrgan/'
        model_path = os.path.join(RealESRGAN_dir, 'experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            raise Exception(model_name + ".pth not found at path " + model_path)

        sys.path.append(os.path.abspath(RealESRGAN_dir))
        from realesrgan import RealESRGANer

        instance = RealESRGANer(scale=2, model_path=model_path, model=RealESRGAN_models[model_name], pre_pad=0,
                                half=True)
        instance.device = torch.device(f'cuda:0')
        instance.model.to('cuda')

        instance.model.name = model_name
        return instance

    def processRealESRGAN(self, image):
        image = image.convert("RGB")
        result, res = self.RealESRGAN.enhance(np.array(image, dtype=np.uint8))
        result = Image.fromarray(result)
        return result

    def load_img(self, data):
        image = Image.fromqimage(data)

        image=image.convert("RGB")

        init_width, init_height = image.size
        width, height, tooBig = self.check_and_resize(init_width, init_height)
        if tooBig < 0:
            image = self.processRealESRGAN(image).resize((width, height), Image.LANCZOS)
        if tooBig > 0:
            image = image.resize((width, height), Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1., (init_width, init_height), tooBig

    def load_img_with_alpha_mask(self, data):
        image = Image.fromqimage(data)
        init_image=image
        image=image.convert("RGBA") #Image.open(io.BytesIO(data)).convert("RGB")
        image.getchannel("A").save("test1.png")
        init_width, init_height = image.size
        width, height, tooBig = self.check_and_resize(init_width, init_height)
        test=image.resize((width//8, height//8), Image.LANCZOS)

        if tooBig < 0:
            print("tooBig < 0")
            image = self.processRealESRGAN(image).resize((width, height), Image.LANCZOS)
        if tooBig > 0:
            print("tooBig > 0")
            image = image.resize((width, height), Image.LANCZOS)

        mask_channel = test.getchannel("A")

        mask = np.array(mask_channel).astype(np.float32) / 255.0
        mask = (1 - mask)

        mask = np.tile(mask, (4, 1, 1))
        mask = mask[None].transpose(0, 1, 2, 3)

        image=image.convert("RGB")

        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        return init_image, 2.*image - 1., mask, (init_width, init_height), tooBig

    def calculate_max_dimensions(self, maxTotal, w, h):
        ratio=w/h
        maxWidth=False
        if ratio >=1:
            maxWidth=True
            normalized_ratio = ratio
        else:
            normalized_ratio = 1/ratio
        min_side=math.floor(math.sqrt(maxTotal/normalized_ratio))
        max_side=math.floor(min_side*normalized_ratio)

        if maxWidth:
            w=max_side
            h=min_side
        else:
            w=min_side
            h=max_side
        print(f"new size {int(w//64)*64}x{int(h//64)*64}")
        return int(w//64)*64, int(h//64)*64

    def check_and_resize(self, w, h):
        print(f"old size {w}x{h}")
        neww,newh=self.calculate_max_dimensions(MAX_SIZE_IN_PIXELS, w, h)
        if neww*newh>w*h:
            return neww, newh, -1
        if neww*newh<w*h:
            return neww, newh, 1
        return w, h, 0

    def get_progress(self, visual):
        result, progress = self.sampler.progress()
        if progress==0:
            return None, progress
        if visual or progress==100:
            x_samples_ddim = self.model.decode_first_stage(result)
            x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)[0]

            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            result = Image.fromarray(x_sample.astype(np.uint8))
            if self.current_generation_info['tooBig'] < 0:
                result = result.resize((self.current_generation_info['w'], self.current_generation_info['h']), Image.LANCZOS)
        if visual and self.current_generation_info['tooBig'] > 0 and progress < 100:
            result = result.resize((self.current_generation_info['w'], self.current_generation_info['h']), Image.LANCZOS)
        if progress==100:
            self.sampler.image_stage=None
            self.sampler.progress_percent=0

            if self.current_generation_info['tooBig'] > 0:
                result = self.processRealESRGAN(result).resize((self.current_generation_info['w'], self.current_generation_info['h']), Image.LANCZOS)
            if self.current_generation_info['mode']=='inpaint':
                result = Image.composite(self.current_generation_info['orig_image'], result, self.current_generation_info['orig_image'].getchannel("A"))
            torch.cuda.empty_cache()
        return result, progress

    def img2imgInpainting(self, flags, image_data):
        self.generation_lock.acquire()
        self.sampler = self.samplers[1]
        final_flags=self.default_flags.copy()
        final_flags.update(flags)

        seed_everything(int(final_flags['seed']))
        batch_size=1
        prompt = final_flags['prompt']
        assert prompt is not None
        data = [[prompt]]

        orig_image, init_image, mask, (init_width, init_height), tooBig = self.load_img_with_alpha_mask(image_data)

        self.current_generation_info={'w':int(flags['w']),'h':int(flags['h']),'tooBig':tooBig,"mode":'inpaint', 'orig_image':orig_image}

        init_image = init_image.to(self.device)
        mask = torch.from_numpy(mask).to(self.device)

        init_image = repeat(init_image, '1 ... -> b ...', b=1)
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

        self.sampler.make_schedule(ddim_num_steps=int(final_flags['ddim_steps']), ddim_eta=final_flags['ddim_eta'], verbose=False)

        assert 0. <= float(final_flags['strength']) <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(float(final_flags['strength']) * int(final_flags['ddim_steps']))
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast if final_flags['precision'] == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    tic = time.time()
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if float(final_flags['scale']) != 1.0:
                            uc = self.model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self.device))

                        random = torch.randn(mask.shape, device=self.device)
                        z_enc = (mask * random) + ((1 - mask) * z_enc)

                        # decode it
                        samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=float(final_flags['scale']),
                                                unconditional_conditioning=uc, z_mask=mask, x0=init_latent)

                        self.generation_lock.release()
                        return final_flags #result, final_flags

    def img2img(self, flags, image_data):
        self.generation_lock.acquire()
        self.sampler = self.samplers[1]
        final_flags=self.default_flags.copy()
        final_flags.update(flags)

        seed_everything(int(final_flags['seed']))
        batch_size=1
        prompt = final_flags['prompt']
        assert prompt is not None
        data = [[prompt]]

        init_image, (init_width, init_height), tooBig = self.load_img(image_data)

        self.current_generation_info={'w':int(flags['w']),'h':int(flags['h']),'tooBig':tooBig,"mode":'img2img'}

        init_image = init_image.to(self.device)

        init_image = repeat(init_image, '1 ... -> b ...', b=1)
        init_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(init_image))  # move to latent space

        self.sampler.make_schedule(ddim_num_steps=int(final_flags['ddim_steps']), ddim_eta=final_flags['ddim_eta'], verbose=False)

        assert 0. <= float(final_flags['strength']) <= 1., 'can only work with strength in [0.0, 1.0]'
        t_enc = int(float(final_flags['strength']) * int(final_flags['ddim_steps']))
        print(f"target t_enc is {t_enc} steps")

        precision_scope = autocast if final_flags['precision'] == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    tic = time.time()
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if float(final_flags['scale']) != 1.0:
                            uc = self.model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.model.get_learned_conditioning(prompts)

                        # encode (scaled latent)
                        z_enc = self.sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(self.device))
                        # decode it
                        samples = self.sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=float(final_flags['scale']),
                                                unconditional_conditioning=uc,)

                        self.generation_lock.release()
                        return final_flags #result, final_flags


    def generate(self, flags):
        self.generation_lock.acquire()
        self.sampler = self.samplers[0]
        final_flags=self.default_flags.copy()
        final_flags.update(flags)

        width, height, tooBig = self.check_and_resize(int(final_flags['w']), int(final_flags['h']))
        self.current_generation_info={'w':int(flags['w']),'h':int(flags['h']),'tooBig':tooBig,"mode":'txt2img'}

        seed_everything(int(final_flags['seed']))

        batch_size = 1

        prompts = final_flags['prompt']
        assert prompts is not None

        precision_scope = autocast if final_flags['precision']=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    tic = time.time()
                    for n in trange(1, desc="Sampling"):
                        uc = None
                        if float(final_flags['scale']) != 1.0:
                            uc = self.model.get_learned_conditioning(batch_size * [""])

                        c = torch.zeros_like(uc)
                        for p in self.parse_prompt(prompts):
                            c = torch.add(c, self.model.get_learned_conditioning(p[0]), alpha=p[1])

                        shape = [int(final_flags['c']), height // int(final_flags['f']), width // int(final_flags['f'])]
                        samples_ddim, _ = self.sampler.sample(S=int(final_flags['ddim_steps']),
                                                        conditioning=c,
                                                        batch_size=int(self.default_flags['n_samples']),
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=float(final_flags['scale']),
                                                        unconditional_conditioning=uc,
                                                        eta=float(final_flags['ddim_eta']),
                                                        x_T=None)

                        self.generation_lock.release()
                        return final_flags #result, final_flags
