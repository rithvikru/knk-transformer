import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrainerConfig:
    max_epochs = 2
    batch_size = 512 * 8
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 231e6
    final_tokens = 231e9
    ckpt_path = None
    num_workers = 8
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

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        # Setup wandb usage flag
        self.use_wandb = bool(getattr(self.config, "use_wandb", False)) and WANDB_AVAILABLE
        if bool(getattr(self.config, "use_wandb", False)) and not WANDB_AVAILABLE:
            logging.warning("use_wandb=True but wandb is not available; skipping W&B logging.")

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

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.eval_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:

                x = x.to(self.device)
                y = y.to(self.device)

                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean()
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
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

                    if self.use_wandb and (it % getattr(config, 'log_interval', 10) == 0):
                        try:
                            # handle grad_norm which may be a tensor
                            grad_norm_value = float(grad_norm) if hasattr(grad_norm, '__float__') else float(getattr(grad_norm, 'item', lambda: 0.0)())
                            wandb.log({
                                'train/loss': loss.item(),
                                'train/lr': lr,
                                'train/epoch': epoch,
                                'train/iter': epoch * len(loader) + it,
                                'train/grad_norm': grad_norm_value,
                                'train/tokens': int(self.tokens) if hasattr(self, 'tokens') else 0,
                            })
                        except Exception:
                            pass

                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

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
        self.tokens = 0
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.eval_dataset is not None:
                test_loss = run_epoch('test')

            if self.config.ckpt_path is not None:
                if self.eval_dataset is None:
                    self.save_checkpoint()
                    continue
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_checkpoint()
                    if self.use_wandb:
                        try:
                            wandb.log({'val/best_loss': best_loss})
                        except Exception:
                            pass
        
        