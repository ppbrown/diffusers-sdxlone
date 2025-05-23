# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import unittest

import numpy as np
import torch

from diffusers import AutoPipelineForText2Image, SDXLonePipeline
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
    torch_device,
)
from diffusers.utils import load_image # For image comparison if needed, or PIL.Image

# Import the base test class or relevant utilities if there's a common structure for pipeline tests
# For example, from tests.pipelines.pipeline_params import TEXT_TO_IMAGE_PARAMS
# from tests.pipelines.test_pipelines_common import PipelineTesterMixin # If applicable

enable_full_determinism()

# Minimal SDXL model ID for testing (community model with minimal components)
# Using a tiny SDXL model would be best, but stabilityai/stable-diffusion-xl-base-1.0 is canonical.
# For faster tests, if a 'tiny' or 'ci-friendly' SDXL variant exists, use that.
# Otherwise, we'll have to rely on 'slow' decorator.
# Let's use a known small model if possible, or a standard one if not.
# For now, this is a placeholder. A real small SDXL-compatible model would be `diffusers/stable-diffusion-xl-base-1.0-tiny` if it existed.
# Using runwayml/stable-diffusion-v1-5 as a placeholder for a small model for structure,
# but this will need to be replaced with an actual SDXL-compatible small model or run with full SDXL.
# For now, let's assume we will run this with a full model under @slow decorator.
BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# If there's a very small, minimal confi for SDXL for CI, that would be better.
# e.g., "hf-internal-testing/tiny-stable-diffusion-xl-pipe" if it exists.
# For now, stick to the full model and mark test as slow.

class SDXLonePipelineFastTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_sdxl_one_pipeline_loading(self):
        pipe = SDXLonePipeline.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16)
        self.assertIsNotNone(pipe)
        self.assertIsNone(pipe.text_encoder_2)
        self.assertIsNone(pipe.tokenizer_2)
        self.assertIsNotNone(pipe.text_encoder)
        self.assertIsNotNone(pipe.tokenizer)
        self.assertIsNotNone(pipe.unet)
        self.assertIsNotNone(pipe.vae)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    def test_sdxl_one_pipeline_instance_from_auto_pipeline(self):
        pipe = AutoPipelineForText2Image.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            # Explicitly use the custom pipeline key if needed, or ensure the model's config.json points to it
            # For now, we assume the custom_pipeline_type can be inferred or is set in config
            custom_pipeline="stable-diffusion-xl-one-encoder" 
        )
        self.assertIsInstance(pipe, SDXLonePipeline)
        self.assertIsNone(pipe.text_encoder_2)
        self.assertIsNone(pipe.tokenizer_2)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


@slow
@require_torch_gpu
class SDXLonePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_sdxl_one_pipeline_generation_tiny(self):
        # This test ideally uses a tiny SDXL model. If not available, it's a full model test.
        # For now, assume BASE_MODEL_ID is a full model.
        pipe = SDXLonePipeline.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "A small cat"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        
        # Generate with fewer steps for faster testing
        images = pipe(
            prompt, 
            generator=generator, 
            num_inference_steps=2, 
            output_type="np"
        ).images

        self.assertIsNotNone(images)
        self.assertEqual(images.shape[0], 1) # batch size
        self.assertEqual(images.shape[1], 1024) # default height for SDXL
        self.assertEqual(images.shape[2], 1024) # default width for SDXL
        self.assertEqual(images.shape[3], 3) # channels

        # It's hard to check for specific image content with so few steps on a large model.
        # The main goal here is to ensure no crashes and correct output shape.
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    def test_sdxl_one_pipeline_negative_prompt(self):
        pipe = SDXLonePipeline.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "A dog"
        negative_prompt = "blur, low quality"
        generator = torch.Generator(device=torch_device).manual_seed(0)
        
        images = pipe(
            prompt, 
            negative_prompt=negative_prompt,
            generator=generator, 
            num_inference_steps=2, 
            output_type="np"
        ).images

        self.assertIsNotNone(images)
        self.assertEqual(images.shape[0], 1)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
