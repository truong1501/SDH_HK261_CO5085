"""Optional integration check; uses random tiny models and synthetic images, no Hub download."""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["PYTHONUTF8"] = "1"
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionInstructPix2PixPipeline
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="ip2p_smoke_") as temp:
    temp = Path(temp)
    (temp/"vocab.json").write_text('{"<|startoftext|>": 0, "<|endoftext|>": 1}')
    (temp/"merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(temp/"vocab.json"), str(temp/"merges.txt"), model_max_length=77)
    text = CLIPTextModel(CLIPTextConfig(vocab_size=2, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                      num_attention_heads=4, max_position_embeddings=77, bos_token_id=0, eos_token_id=1, pad_token_id=1))
    unet = UNet2DConditionModel(sample_size=8, in_channels=8, out_channels=4, block_out_channels=(32,64),
                               down_block_types=("CrossAttnDownBlock2D","DownBlock2D"),
                               up_block_types=("UpBlock2D","CrossAttnUpBlock2D"), layers_per_block=1,
                               cross_attention_dim=32, norm_num_groups=8)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",)*2,
                        up_block_types=("UpDecoderBlock2D",)*2, block_out_channels=(32,64),
                        latent_channels=4, norm_num_groups=8, sample_size=16)
    # Random synthetic-model test only; production comparison retains the pretrained safety checker.
    pipe = StableDiffusionInstructPix2PixPipeline(vae=vae, text_encoder=text, tokenizer=tokenizer,
            unet=unet, scheduler=DDPMScheduler(num_train_timesteps=10), safety_checker=None,
            feature_extractor=None, requires_safety_checker=False)
    pipe.save_pretrained(temp/"base")
    images = [Image.new("RGB",(16,16),(i*20,50,100)) for i in range(10)]
    ds = Dataset.from_dict(dict(input_image=images, edited_image=images, edit_prompt=["test"]*10))
    DatasetDict(train=ds.select(range(6)), validation=ds.select([6,7]), test=ds.select([8,9])).save_to_disk(str(temp/"data"))
    command = [sys.executable, str(root/"train_instruct_pix2pix.py"),
            "--pretrained_model_name_or_path", str(temp/"base"), "--dataset_name", str(temp/"data"),
            "--output_dir", str(temp/"run"), "--resolution", "16", "--train_batch_size", "2",
            "--max_train_steps", "2", "--checkpointing_steps", "1", "--checkpoints_total_limit", "2",
            "--report_to", "tensorboard", "--lr_warmup_steps", "0", "--seed", "42", "--conditioning_dropout_prob", "0.05"]
    subprocess.run(command, check=True)
    assert (temp/"run"/"checkpoint-2"/"unet"/"config.json").is_file()
    command[command.index("--max_train_steps")+1] = "3"
    subprocess.run(command+["--resume_from_checkpoint", "latest"], check=True)
    assert (temp/"run"/"checkpoint-3"/"unet"/"config.json").is_file()
    print("TINY CPU TRAIN + CHECKPOINT + RESUME PASS", flush=True)
