import argparse
import os
import torch
from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
import random
import numpy as np
from accelerate import Accelerator
from level_dataset import visualize_samples
import json
from datetime import datetime
from torch.distributions import Categorical
from models.block2vec_model import Block2Vec
import glob
import models.general_training_helper as gen_train_help
import re
from models.pipeline_loader import get_pipeline
from models.latent_diffusion_pipeline import UnconditionalDDPMPipeline

MM2_TILESET = 'mm2_tileset.json'
MM2_NUM_INFERENCE_STEPS = 50


def mse_loss(pred, target, scene_oh=None, noisy_scenes=None, **kwargs):
    return torch.nn.functional.mse_loss(pred, target)


def reconstruction_loss(pred, target, scene_oh, noisy_scenes, timesteps=None, scheduler=None, **kwargs):
    if timesteps is None or scheduler is None:
        raise ValueError("timesteps and scheduler must be provided for reconstruction_loss")
    alpha_hat = scheduler.alphas_cumprod[timesteps].to(pred.device)
    sqrt_alpha_hat = torch.sqrt(alpha_hat)[:, None, None, None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1. - alpha_hat)[:, None, None, None]
    logits = (1.0 / sqrt_alpha_hat) * (noisy_scenes - sqrt_one_minus_alpha_hat * pred)
    target_indices = scene_oh.argmax(dim=1)
    logits = logits.permute(0, 2, 3, 1)
    dist = Categorical(logits=logits)
    rec_loss = -dist.log_prob(target_indices).sum(dim=(1, 2)).mean()
    return rec_loss


def combined_loss(pred, target, scene_oh=None, noisy_scenes=None, timesteps=None, scheduler=None, **kwargs):
    mse = mse_loss(pred, target)
    rec = reconstruction_loss(pred, target, scene_oh, noisy_scenes, timesteps=timesteps, scheduler=scheduler)
    return mse + 0.001 * rec


def parse_args():
    parser = argparse.ArgumentParser(description="Train an unconditional diffusion model for MM2 tile-based level generation")

    # Dataset args
    parser.add_argument("--json", type=str, default="datasets/mm2_dataset.json", help="Path to dataset JSON file")
    parser.add_argument("--val_json", type=str, default=None, help="Optional path to validation dataset JSON file")
    parser.add_argument("--tileset", type=str, default=MM2_TILESET, help="Path to MM2 tileset JSON (used to infer num_tiles)")
    parser.add_argument("--num_tiles", type=int, default=None, help="Override tile count (inferred from tileset if not set)")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")

    # Model args
    parser.add_argument("--model_dim", type=int, default=128, help="Base dimension of UNet model")
    parser.add_argument("--dim_mults", nargs="+", type=int, default=[1, 2, 4], help="Dimension multipliers for UNet")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per downsampling")
    parser.add_argument("--down_block_types", nargs="+", type=str,
                        default=["DownBlock2D", "DownBlock2D", "DownBlock2D"],
                        help="Down block types for UNet")
    parser.add_argument("--up_block_types", nargs="+", type=str,
                        default=["UpBlock2D", "UpBlock2D", "UpBlock2D"],
                        help="Up block types for UNet")
    parser.add_argument("--attention_head_dim", type=int, default=8, help="Number of attention heads")

    # Training args
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr_warmup_percentage", type=float, default=0.05, help="Learning rate warmup portion")
    parser.add_argument("--lr_scheduler_cycles", type=float, default=0.5, help="Cosine LR scheduler cycles")
    parser.add_argument("--save_image_epochs", type=int, default=20, help="Save generated levels every N epochs")
    parser.add_argument("--save_model_epochs", type=int, default=20, help="Save model every N epochs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate_epochs", type=int, default=5, help="Calculate validation loss every N epochs")

    # Output args
    parser.add_argument("--output_dir", type=str, default="level-diffusion-output", help="Output directory")

    # Diffusion scheduler args
    parser.add_argument("--num_train_timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--num_inference_timesteps", type=int, default=MM2_NUM_INFERENCE_STEPS, help="Inference timesteps for sample generation")
    parser.add_argument("--beta_schedule", type=str, default="linear", help="Beta schedule type")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)

    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file with training parameters")

    # Block embedding model
    parser.add_argument("--block_embedding_model_path", type=str, default=None, help="Path to trained block embedding model (.pt)")

    # Loss function
    parser.add_argument("--loss_type", type=str, default="COMBO", choices=["MSE", "REC", "COMBO"],
                        help="Loss function: MSE, REC (reconstruction), or COMBO")

    parser.add_argument("--sprite_temperature_n", type=int, default=None,
                        help="Enable per-sprite temperature scaling with root n (e.g. 2, 4, 8)")

    parser.add_argument("--use_early_stopping", action="store_true", help="Stop training on val loss stagnation")
    parser.add_argument("--patience", type=int, default=30, help="Epochs without improvement before early stopping")

    return parser.parse_args()


def compute_sprite_scaling_factors(json_path, num_tiles, n):
    with open(json_path, 'r') as f:
        data = json.load(f)
    counts = [0] * num_tiles
    for entry in data:
        level = entry.get('level')
        if level is not None:
            for row in level:
                for tile in row:
                    counts[tile] += 1
    counts = [c if c > 0 else 1 for c in counts]
    scalings = [c ** (1 / n) for c in counts]
    min_scaling = min(scalings)
    scalings = [s / min_scaling for s in scalings]
    return torch.tensor(scalings, dtype=torch.float32)


def find_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None, None
    pattern = re.compile(r"checkpoint-(\d+)")
    epochs = [(int(pattern.search(os.path.basename(c)).group(1)), c) for c in checkpoints if pattern.search(os.path.basename(c))]
    if not epochs:
        return None, None
    latest_epoch, latest_ckpt = max(epochs, key=lambda x: x[0])
    return latest_ckpt, latest_epoch


def copy_log_up_to_epoch(output_dir, log_file, resume_epoch, log_pattern):
    log_files = [
        f for f in glob.glob(os.path.join(output_dir, log_pattern))
        if os.path.abspath(f) != os.path.abspath(log_file)
    ]
    if not log_files:
        raise RuntimeError(f"No previous log files found in {output_dir} matching pattern {log_pattern}.")
    prev_log_file = max(log_files, key=os.path.getmtime)
    print(f"Copying log entries from {prev_log_file} up to epoch {resume_epoch} into {log_file}")
    with open(prev_log_file, 'r') as fin, open(log_file, 'w') as fout:
        for line in fin:
            try:
                entry = json.loads(line)
                if entry.get("epoch", -1) <= resume_epoch:
                    fout.write(line)
            except Exception as e:
                raise RuntimeError(f"Malformed log line in {prev_log_file}: {line.strip()} ({e})")
    print(f"Truncated log file {log_file} to only include entries up to epoch {resume_epoch}")


def infer_global_step_from_log(log_file):
    global_step = 0
    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "step" in entry:
                        global_step = entry["step"]
                except Exception:
                    continue
    except Exception as e:
        raise RuntimeError(f"Could not read log file {log_file} to infer global step: {e}")
    return global_step


def main():
    args = parse_args()

    if args.loss_type == "MSE":
        loss_fn = mse_loss
    elif args.loss_type == "REC":
        loss_fn = reconstruction_loss
    elif args.loss_type == "COMBO":
        loss_fn = combined_loss
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    print(f"Using loss function: {args.loss_type}")

    # Infer num_tiles from tileset if not explicitly provided
    if args.num_tiles is None:
        with open(args.tileset, 'r', encoding='utf-8') as f:
            tileset_data = json.load(f)
        args.num_tiles = len(sorted(tileset_data['tiles'].keys())) + 1  # +1 for extra/void tile
        print(f"Inferred num_tiles={args.num_tiles} from {args.tileset}")

    if hasattr(args, 'config') and args.config:
        config = load_config_from_json(args.config)
        args = update_args_from_config(args, config)
        print("Training will use parameters from the config file.")

    if os.path.exists(args.output_dir):
        checkpoints = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
        if checkpoints:
            user_input = input(f"Output directory '{args.output_dir}' already exists and contains checkpoints. Resume training from last checkpoint? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Exiting. Please remove the directory or choose a different output directory.")
                exit()
            resume_training = True
        else:
            raise RuntimeError(f"Output directory '{args.output_dir}' already exists but contains no checkpoints. Please remove it or choose a different name.")
    else:
        os.makedirs(args.output_dir)
        resume_training = False

    sprite_scaling_factors = None
    if args.sprite_temperature_n is not None:
        raise ValueError("temperature scaling not currently implemented")
        sprite_scaling_factors = compute_sprite_scaling_factors(
            args.json, args.num_tiles, args.sprite_temperature_n
        )
        print(f"Sprite scaling factors: {sprite_scaling_factors}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    block_embeddings = None
    embedding_dim = None
    if args.block_embedding_model_path:
        try:
            block2vec = Block2Vec.from_pretrained(args.block_embedding_model_path)
            block_embeddings = block2vec.get_embeddings()
            embedding_dim = block_embeddings.shape[1]
            print(f"Loaded block embeddings from {args.block_embedding_model_path} with dimension {embedding_dim}")
        except Exception as e:
            print(f"Error loading block embedding model: {e}")
            raise
    else:
        print("No block embedding model specified. One-hot encoding enabled.")

    train_dataloader, val_dataloader = gen_train_help.create_dataloaders(
        json_path=args.json,
        val_json=args.val_json,
        augment=args.augment,
        num_tiles=args.num_tiles,
        block_embeddings=block_embeddings,
        batch_size=args.batch_size
    )

    first_sample = train_dataloader.dataset[0]
    scene = first_sample[0] if isinstance(first_sample, (list, tuple)) else first_sample
    scene_height = scene.shape[1]
    scene_width = scene.shape[2]
    print(f"Scene height: {scene_height}")
    print(f"Scene width: {scene_width}")

    in_channels = embedding_dim if args.block_embedding_model_path else args.num_tiles
    out_channels = in_channels

    model = UNet2DModel(
        sample_size=(scene_height, scene_width),
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=args.num_res_blocks,
        block_out_channels=[args.model_dim * mult for mult in args.dim_mults],
        down_block_types=args.down_block_types,
        up_block_types=args.up_block_types,
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    total_training_steps = (len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps
    warmup_steps = int(total_training_steps * args.lr_warmup_percentage)
    print(f"Warmup period will be {warmup_steps} steps out of {total_training_steps}")

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_cycles=args.lr_scheduler_cycles,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    progress_bar = tqdm(total=args.num_epochs * len(train_dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')
    log_file = os.path.join(args.output_dir, f"training_log_{formatted_date}.jsonl")
    config_file = os.path.join(args.output_dir, f"hyperparams_{formatted_date}.json")

    if accelerator.is_local_main_process:
        hyperparams = vars(args)
        with open(config_file, "w") as f:
            json.dump(hyperparams, f, indent=4)
        print(f"Saved configuration to: {config_file}")

    def log_metrics(epoch, loss, lr, step=None, val_loss=None):
        if accelerator.is_local_main_process:
            log_entry = {
                "epoch": epoch,
                "loss": loss,
                "lr": lr,
                "step": step if step is not None else epoch * len(train_dataloader),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if val_loss is not None:
                log_entry["val_loss"] = val_loss
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    plotter, plot_thread = None, None
    if accelerator.is_local_main_process:
        plotter, plot_thread = gen_train_help.start_plotter(
            log_file=log_file, output_dir=args.output_dir,
            left_key='loss', right_key='val_loss',
            left_label='Training Loss', right_label='Validation Loss',
            png_name='training_loss'
        )

    patience = args.patience if hasattr(args, 'patience') else 30
    early_stop = False
    epochs_since_improvement = 0

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    start_epoch = 0
    global_step = 0

    if resume_training:
        latest_ckpt, latest_epoch = find_latest_checkpoint(args.output_dir)
        copy_log_up_to_epoch(args.output_dir, log_file, latest_epoch, "training_log_*.jsonl")

        if latest_ckpt is not None:
            pipeline = get_pipeline(latest_ckpt)
            model = pipeline.unet
            noise_scheduler = pipeline.scheduler

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )

            optimizer_path = os.path.join(latest_ckpt, "optimizer.pt")
            if os.path.exists(optimizer_path):
                optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))

            lr_scheduler_config_path = os.path.join(latest_ckpt, "lr_scheduler_config.json")
            if os.path.exists(lr_scheduler_config_path):
                with open(lr_scheduler_config_path, "r") as f:
                    scheduler_config = json.load(f)
                lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_cycles=scheduler_config["num_cycles"],
                    num_warmup_steps=scheduler_config["num_warmup_steps"],
                    num_training_steps=scheduler_config["num_training_steps"],
                )
                lr_scheduler_path = os.path.join(latest_ckpt, "lr_scheduler.pt")
                if os.path.exists(lr_scheduler_path):
                    lr_scheduler.load_state_dict(torch.load(lr_scheduler_path, map_location="cpu"))
            else:
                raise RuntimeError("lr_scheduler_config.json not found in checkpoint. Cannot resume scheduler correctly.")

            model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, lr_scheduler
            )

            early_stop_path = os.path.join(latest_ckpt, "early_stop_state.json")
            if os.path.exists(early_stop_path):
                with open(early_stop_path, "r") as f:
                    early_stop_state = json.load(f)
                best_val_loss = early_stop_state.get("best_val_loss", float('inf'))
                best_epoch = early_stop_state.get("best_epoch", 0)
                epochs_since_improvement = early_stop_state.get("epochs_since_improvement", 0)
            else:
                best_val_loss = float('inf')
                best_epoch = 0
                epochs_since_improvement = 0

            start_epoch = latest_epoch + 1
            global_step = infer_global_step_from_log(log_file)
            print(f"Resumed training from epoch {start_epoch}, global_step {global_step}")
        else:
            raise RuntimeError(f"No checkpoint found in {args.output_dir}. Please check the directory or remove it to start fresh.")

    for epoch in range(start_epoch, args.num_epochs):
        if args.use_early_stopping and early_stop:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break
        model.train()
        train_loss = 0.0

        for batch in train_dataloader:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with accelerator.accumulate(model):
                loss = process_diffusion_batch(model, batch, noise_scheduler, loss_fn, accelerator)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            train_loss += loss.detach().item()

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.detach().item(), step=global_step)

            del loss
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            global_step += 1

        avg_train_loss = train_loss / len(train_dataloader)

        val_loss = None
        val_loss_improved = False
        if val_dataloader is not None and (epoch % args.validate_epochs == 0 or epoch == args.num_epochs - 1):
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_batch_loss = process_diffusion_batch(model, val_batch, noise_scheduler, loss_fn, accelerator)
                    val_loss += val_batch_loss.item()
                    del val_batch_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            val_loss /= len(val_dataloader)

            val_loss_improved = val_loss < best_val_loss
            if val_loss_improved:
                best_val_loss = val_loss
                best_epoch = epoch
                best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }

            if args.use_early_stopping and not val_loss_improved:
                epochs_since_improvement = epoch - best_epoch
                print(f"No improvement in val loss for {epochs_since_improvement}/{patience} epochs.")
                if epochs_since_improvement >= patience:
                    print(f"\nEarly stopping triggered. Best val loss: {best_val_loss:.4f}")
                    early_stop = True

            model.train()

        log_metrics(epoch, avg_train_loss, lr_scheduler.get_last_lr()[0], val_loss=val_loss, step=global_step)

        if val_dataloader is not None and (epoch % args.validate_epochs == 0 or epoch == args.num_epochs - 1):
            val_result = f"{val_loss:.4f}" if val_loss is not None else "N/A"
            status_message = f"Epoch {epoch+1} of {args.num_epochs}, Loss: {avg_train_loss:.4f}, Val Loss: {val_result}"
        else:
            status_message = f"Epoch {epoch+1} of {args.num_epochs}, Loss: {avg_train_loss:.4f}"
        if args.use_early_stopping:
            status_message += f", No improvement for {epochs_since_improvement} of {patience} epochs."
        print(status_message)

        if epoch % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            model.eval()
            pipeline = UnconditionalDDPMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
                block_embeddings=block_embeddings
            )
            if sprite_scaling_factors is not None:
                pipeline.give_sprite_scaling_factors(sprite_scaling_factors)
            with torch.no_grad():
                samples = pipeline(
                    batch_size=4,
                    height=scene_height,
                    width=scene_width,
                    generator=torch.Generator(device=accelerator.device).manual_seed(args.seed),
                    num_inference_steps=args.num_inference_timesteps,
                    output_type="tensor",
                    show_progress_bar=False,
                ).images
            visualize_samples(samples, os.path.join(args.output_dir, f"samples_epoch_{epoch}"))

        if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}")
            pipeline = UnconditionalDDPMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
                block_embeddings=block_embeddings
            )
            if sprite_scaling_factors is not None:
                pipeline.give_sprite_scaling_factors(sprite_scaling_factors)
            accelerator.wait_for_everyone()
            pipeline.save_pretrained(checkpoint_dir)
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
            torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "lr_scheduler.pt"))

            early_stop_state = {
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "epochs_since_improvement": epochs_since_improvement
            }
            with open(os.path.join(checkpoint_dir, "early_stop_state.json"), "w") as f:
                json.dump(early_stop_state, f)

            scheduler_config = {
                "num_warmup_steps": warmup_steps,
                "num_training_steps": total_training_steps,
                "num_cycles": args.lr_scheduler_cycles,
            }
            with open(os.path.join(checkpoint_dir, "lr_scheduler_config.json"), "w") as f:
                json.dump(scheduler_config, f)

    try:
        if accelerator.is_local_main_process and plotter:
            gen_train_help.kill_plotter(plotter, plot_thread)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
    finally:
        progress_bar.close()

        if best_model_state is not None:
            model.load_state_dict(best_model_state['model_state_dict'])
            best_model_info = {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
            }
            with open(os.path.join(args.output_dir, "best_model_info.json"), "w") as f:
                json.dump(best_model_info, f)
            print(f"\nSaved best model from epoch {best_epoch}, val loss: {best_val_loss:.4f}")

        pipeline = UnconditionalDDPMPipeline(
            unet=accelerator.unwrap_model(model),
            scheduler=noise_scheduler,
            block_embeddings=block_embeddings
        )
        if sprite_scaling_factors is not None:
            pipeline.give_sprite_scaling_factors(sprite_scaling_factors)
        pipeline.save_pretrained(args.output_dir)


def load_config_from_json(config_path):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Configuration loaded from {config_path}")
        for key, value in config.items():
            print(f"  {key}: {value}")
        return config
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading config file: {e}")
        raise e


def update_args_from_config(args, config):
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def process_diffusion_batch(model, batch, noise_scheduler, loss_fn, accelerator):
    scenes = batch[0] if isinstance(batch, (list, tuple)) else batch
    scenes = scenes.to(accelerator.device)

    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (scenes.shape[0],), device=accelerator.device
    ).long()

    noise = torch.randn_like(scenes)
    noisy_scenes = noise_scheduler.add_noise(scenes, noise, timesteps)
    noise_pred = model(noisy_scenes, timesteps).sample

    return loss_fn(noise_pred, noise, scenes, noisy_scenes, timesteps=timesteps, scheduler=noise_scheduler)


if __name__ == "__main__":
    main()
