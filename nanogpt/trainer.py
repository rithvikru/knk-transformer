import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import os
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrainerConfig:
    max_epochs = 2
    # micro-batch size per process (per-GPU)
    micro_batch_size = 32
    # desired global batch size across all GPUs and accumulation steps
    global_batch_size = None  # if set, overrides grad_accum_steps based on world size
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 231e6
    final_tokens = 231e9
    ckpt_path = None
    num_workers = 8
    persistent_workers = True
    prefetch_factor = 4
    pin_memory = True
    drop_last = True
    # precision
    use_bf16 = True
    use_fp16 = False
    grad_accum_steps = 1
    # validation/eval
    val_max_batches = 200
    # distributed
    ddp = True
    # wandb logging options
    use_wandb = False
    log_interval = 10
    wandb_project = "knk-transformer"
    wandb_run_name = None
    wandb_watch = False

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, eval_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = config

        # Distributed setup
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))

        self.is_distributed = self.world_size > 1 and bool(getattr(self.config, 'ddp', True))
        self.device = 'cpu'
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        
        if self.is_distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
            self.model = self.model.to(self.device)
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=False
            )
        else:
            self.model = self.model.to(self.device)

        # Setup wandb usage flag
        self.use_wandb = (self.rank == 0) and bool(getattr(self.config, "use_wandb", False)) and WANDB_AVAILABLE
        if (self.rank == 0) and bool(getattr(self.config, "use_wandb", False)) and not WANDB_AVAILABLE:
            logging.warning("use_wandb=True but wandb is not available; skipping W&B logging.")

        # precision/amp
        self.amp_dtype = torch.bfloat16 if getattr(self.config, 'use_bf16', True) and torch.cuda.is_available() else (torch.float16 if getattr(self.config, 'use_fp16', False) else None)
        # Use new torch.amp API to avoid deprecation warnings
        try:
            self.scaler = torch.amp.GradScaler(device='cuda', enabled=(self.amp_dtype == torch.float16))
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.amp_dtype == torch.float16))

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        # Initialize wandb run if requested
        if self.use_wandb and getattr(wandb, 'run', None) is None:
            wandb.init(project=getattr(config, 'wandb_project', 'knk-transformer'),
                       name=getattr(config, 'wandb_run_name', None))
            # push config for reproducibility
            try:
                wandb.config.update({k: v for k, v in vars(config).items() if not k.startswith('__')})
            except Exception:
                pass
        if self.use_wandb and getattr(config, 'wandb_watch', False):
            try:
                wandb.watch(raw_model, log='all', log_freq=getattr(config, 'log_interval', 10))
            except Exception:
                pass

        # compute grad accumulation to reach desired global batch size if provided
        if getattr(config, 'global_batch_size', None):
            per_step_world = self.world_size * getattr(config, 'micro_batch_size', 1)
            accum = max(1, int(math.ceil(config.global_batch_size / per_step_world)))
            self.config.grad_accum_steps = accum

        # restore from checkpoint if available
        if getattr(config, 'ckpt_path', None) and os.path.exists(config.ckpt_path):
            try:
                checkpoint = torch.load(config.ckpt_path, map_location='cpu')
                raw_model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint.get('optimizer', {}))
                self.tokens = checkpoint.get('tokens', 0)
                if self.scaler is not None and 'scaler' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler'])
                if self.rank == 0:
                    logger.info("Loaded checkpoint '%s' (tokens=%d)", config.ckpt_path, self.tokens)
            except Exception as e:
                if self.rank == 0:
                    logger.warning("Failed to load checkpoint '%s': %s", config.ckpt_path, e)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.eval_dataset
            sampler = None
            if self.is_distributed and hasattr(data, '__len__'):
                sampler = DistributedSampler(data, shuffle=is_train)
                if is_train and hasattr(sampler, 'set_epoch'):
                    # ensure different shuffles each epoch across ranks
                    sampler.set_epoch(epoch)
            loader = DataLoader(
                data,
                shuffle=(sampler is None and is_train),
                sampler=sampler,
                pin_memory=getattr(config, 'pin_memory', True),
                batch_size=getattr(config, 'micro_batch_size', 1),
                num_workers=getattr(config, 'num_workers', 8),
                persistent_workers=getattr(config, 'persistent_workers', True) and getattr(config, 'num_workers', 0) > 0,
                prefetch_factor=getattr(config, 'prefetch_factor', 2) if getattr(config, 'num_workers', 0) > 0 else None,
                drop_last=getattr(config, 'drop_last', True),
            )

            losses = []
            iterator = enumerate(loader)
            if is_train and self.rank == 0:
                iterator = tqdm(iterator, total=len(loader))
            max_val_batches = getattr(config, 'val_max_batches', None)
            for it, (x, y) in iterator:

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    if self.amp_dtype is not None:
                        with torch.cuda.amp.autocast(dtype=self.amp_dtype):
                            logits, loss = model(x, y)
                    else:
                        logits, loss = model(x, y)
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:
                    loss_to_scale = loss / max(1, getattr(config, 'grad_accum_steps', 1))
                    if self.scaler is not None:
                        self.scaler.scale(loss_to_scale).backward()
                    else:
                        loss_to_scale.backward()

                    if ((it + 1) % getattr(config, 'grad_accum_steps', 1)) == 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                        if self.scaler is not None:
                            self.scaler.step(optimizer)
                            self.scaler.update()
                        else:
                            optimizer.step()
                        model.zero_grad(set_to_none=True)

                    if config.lr_decay:
                        pad_id = getattr(raw_model, 'pad_token_id', 0)
                        self.tokens += (y != pad_id).sum().item()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # lighter-weight logging progress to reduce overhead
                    if self.use_wandb and (it % getattr(config, 'log_interval', 50) == 0):
                        try:
                            # handle grad_norm which may be a tensor
                            grad_norm_value = float(grad_norm) if 'grad_norm' in locals() and hasattr(grad_norm, '__float__') else float(getattr(locals().get('grad_norm', 0.0), 'item', lambda: 0.0)())
                            wandb.log({
                                'train/loss': loss.item(),
                                'train/lr': lr,
                                'train/epoch': epoch,
                                'train/iter': epoch * len(loader) + it,
                                'train/grad_norm': grad_norm_value,
                                'train/tokens': int(self.tokens) if hasattr(self, 'tokens') else 0,
                                'train/world_size': int(self.world_size),
                                'train/accum': int(getattr(self.config, 'grad_accum_steps', 1)),
                            })
                        except Exception:
                            pass

                    if self.rank == 0 and 'tqdm' in str(type(iterator)) and (it % 50 == 0):
                        iterator.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

                if (not is_train) and (max_val_batches is not None) and (it + 1 >= max_val_batches):
                    break

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                if self.use_wandb:
                    try:
                        wandb.log({'val/loss': test_loss, 'val/epoch': epoch})
                    except Exception:
                        pass
                return test_loss

        best_loss = float('inf')
        if not hasattr(self, 'tokens'):
            self.tokens = 0
        for epoch in range(config.max_epochs):

            if self.is_distributed:
                # re-set sampler epoch for shuffling
                pass
            run_epoch('train')
            if self.eval_dataset is not None:
                test_loss = run_epoch('test')

            if self.config.ckpt_path is not None and self.rank == 0:
                state = {
                    'model': (self.model.module if hasattr(self.model, 'module') else self.model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'tokens': self.tokens,
                }
                if self.scaler is not None:
                    state['scaler'] = self.scaler.state_dict()
                # always save latest
                torch.save(state, self.config.ckpt_path)
                # save best
                if self.eval_dataset is not None and test_loss < best_loss:
                    best_loss = test_loss
                    torch.save(state, self.config.ckpt_path + ".best")
                    if self.use_wandb:
                        try:
                            wandb.log({'val/best_loss': best_loss})
                        except Exception:
                            pass
        
        