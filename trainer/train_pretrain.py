import os
import sys
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from tqdm import tqdm
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_model import DiffusionModel, DiffusionModelConfig
from dataset.lm_dataset import PretrainDataset


@dataclass
class ModelConfig:
    vocab_size: int = 6400
    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: int = 5504
    max_position_embeddings: int = 512
    mask_token_id: int = 3
    eps: float = 1e-3


@dataclass
class TrainConfig:
    data_path: str = "dataset/pretrain_hq.jsonl"
    tokenizer_path: str = "model"
    save_dir: str = "checkpoints"

    batch_size: int = 32
    grad_accum_steps: int = 1
    epochs: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    num_workers: int = 4
    save_every: int = 1
    sample_every: int = 1000  # 每多少步采样一次
    save_steps: int = 1000  # 每多少步保存checkpoint
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # wandb配置
    use_wandb: bool = True
    wandb_project: str = "diffusion-lm-pretrain"
    wandb_run_name: str = None

    # checkpoint恢复
    resume_from: str = None


def forward_process(input_ids, mask_token_id=3, eps=1e-3):
    b, l = input_ids.shape

    # 每个 batch 样本采样一个 mask 比例 t
    t = torch.rand(b, device=input_ids.device)

    # 防止 mask 概率为 0
    p_mask = t * (1 - eps) + eps      # shape: (b,)
    p_mask = p_mask[:, None].expand(-1, l)  # shape: (b, l)

    # 采样需要 mask 的位置
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask

    # 构造 mask token
    mask_token = torch.full_like(input_ids, mask_token_id)

    noisy_batch = torch.where(masked_indices, mask_token, input_ids)

    return noisy_batch, masked_indices, p_mask


def compute_loss(model, input_ids, mask_token_id=3, eps=1e-3):
    """
    计算扩散语言模型的损失
    只在被mask的位置计算交叉熵损失
    """
    noisy_batch, masked_indices, p_mask = forward_process(
        input_ids, mask_token_id=mask_token_id, eps=eps
    )

    logits = model(noisy_batch)
    assert logits.shape[:2] == input_ids.shape, \
        f"Logits shape {logits.shape} does not match input {input_ids.shape}"
    
    loss_all = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        input_ids.view(-1),
        reduction='none'
    )
    loss_all = loss_all.view(input_ids.shape)  # (b, l)

    # 只在 masked positions 求平均
    masked_loss = loss_all * masked_indices.float()
    num_masked = masked_indices.sum()

    if num_masked == 0:
        return masked_loss.sum()

    return masked_loss.sum() / num_masked

@torch.no_grad()
def sample_from_mask(model, seq_len, batch_size=1, mask_token_id=3, num_steps=50, device='cuda'):
    """
    从全mask序列开始采样
    Args:
        model: 扩散模型
        seq_len: 序列长度
        batch_size: 批次大小
        mask_token_id: MASK token的id
        num_steps: 采样步数
        device: 设备
    Returns:
        samples: 采样得到的序列
    """
    model.eval()

    # 初始化为全mask序列
    x = torch.full((batch_size, seq_len), mask_token_id, dtype=torch.long, device=device)

    # 逐步去噪
    for step in range(num_steps):
        # 计算当前mask比例 (从1逐渐减少到0)
        t = 1.0 - step / num_steps

        # 获取模型预测
        logits = model(x)
        probs = F.softmax(logits, dim=-1)

        # 采样预测的token
        pred_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(batch_size, seq_len)

        # 计算每个位置的置信度
        confidence = probs.max(dim=-1).values

        # 决定哪些位置保持mask（置信度低的位置更可能保持mask）
        # 随着步数增加，保持mask的比例逐渐降低
        mask_ratio = t
        num_to_mask = int(seq_len * mask_ratio)

        if num_to_mask > 0:
            # 按置信度排序，置信度低的位置继续mask
            _, indices = confidence.sort(dim=-1)
            mask_positions = indices[:, :num_to_mask]

            # 更新序列：先用预测token替换，再把低置信度位置恢复为mask
            x = pred_tokens.clone()
            for b in range(batch_size):
                x[b, mask_positions[b]] = mask_token_id
        else:
            x = pred_tokens

    return x


def validate_sampling(model, tokenizer, model_config, train_config, num_samples=2):
    """
    验证采样：从全mask序列生成文本
    """
    print("\n" + "=" * 50)
    print("Validation Sampling (from full mask sequence):")
    print("=" * 50)

    samples = sample_from_mask(
        model=model,
        seq_len=model_config.max_position_embeddings,
        batch_size=num_samples,
        mask_token_id=model_config.mask_token_id,
        num_steps=50,
        device=train_config.device
    )

    for i, sample in enumerate(samples):
        text = tokenizer.decode(sample.tolist(), skip_special_tokens=True)
        print(f"\nSample {i + 1}:")
        print(text[:200] + "..." if len(text) > 200 else text)

    print("=" * 50 + "\n")


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, best_loss, model_config, train_config, filename):
    """保存checkpoint"""
    checkpoint_path = os.path.join(train_config.save_dir, filename)
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'best_loss': best_loss,
        'model_config': model_config,
        'train_config': train_config
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def train_epoch(model, dataloader, optimizer, scheduler, model_config, train_config, epoch, global_step, best_loss, tokenizer):
    model.train()
    total_loss = 0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for step, input_ids in enumerate(pbar):
        input_ids = input_ids.to(train_config.device)

        loss = compute_loss(
            model, input_ids,
            mask_token_id=model_config.mask_token_id,
            eps=model_config.eps
        )
        loss = loss / train_config.grad_accum_steps
        loss.backward()

        if (step + 1) % train_config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # wandb logging
            if train_config.use_wandb:
                wandb.log({
                    'train/loss': loss.item() * train_config.grad_accum_steps,
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/global_step': global_step
                }, step=global_step)

            # 每sample_every步采样一次
            if global_step % train_config.sample_every == 0:
                validate_sampling(model, tokenizer, model_config, train_config, num_samples=2)
                model.train()

            # 每save_steps步保存checkpoint
            if global_step % train_config.save_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    loss.item() * train_config.grad_accum_steps, best_loss,
                    model_config, train_config,
                    f"checkpoint_step_{global_step}.pt"
                )

        total_loss += loss.item() * train_config.grad_accum_steps
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item() * train_config.grad_accum_steps:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    return total_loss / num_batches, global_step


def main():
    model_config = ModelConfig()
    train_config = TrainConfig()

    os.makedirs(train_config.save_dir, exist_ok=True)

    # 初始化wandb
    if train_config.use_wandb:
        wandb.init(
            project=train_config.wandb_project,
            name=train_config.wandb_run_name,
            config={
                'model_config': asdict(model_config),
                'train_config': asdict(train_config)
            }
        )

    print(f"Loading tokenizer from {train_config.tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer_path, trust_remote_code=True)

    print(f"Loading dataset from {train_config.data_path}...")
    dataset = PretrainDataset(
        data_path=train_config.data_path,
        tokenizer=tokenizer,
        max_length=model_config.max_position_embeddings
    )

    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {train_config.batch_size}")
    print(f"Total batches per epoch: {len(dataloader)}")

    diffusion_config = DiffusionModelConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings
    )

    print("Initializing model...")
    model = DiffusionModel(diffusion_config)
    model = model.to(train_config.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95)
    )

    total_steps = len(dataloader) * train_config.epochs // train_config.grad_accum_steps

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=train_config.lr * 0.1)

    # 从checkpoint恢复
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')

    if train_config.resume_from is not None:
        print(f"Resuming from checkpoint: {train_config.resume_from}")
        checkpoint = torch.load(train_config.resume_from, map_location=train_config.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', 0)
        best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('inf')))

        print(f"Resumed from epoch {start_epoch}, global_step {global_step}, best_loss {best_loss:.4f}")

    print(f"\nStarting training...")
    print(f"Device: {train_config.device}")
    print(f"Epochs: {train_config.epochs}")
    print(f"Learning rate: {train_config.lr}")
    print(f"Mask token ID: {model_config.mask_token_id}")
    print(f"Epsilon: {model_config.eps}")
    print("-" * 50)

    for epoch in range(start_epoch, train_config.epochs):
        print(f"\nEpoch {epoch + 1}/{train_config.epochs}")

        avg_loss, global_step = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            model_config=model_config,
            train_config=train_config,
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            tokenizer=tokenizer
        )

        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

        # wandb log epoch metrics
        if train_config.use_wandb:
            wandb.log({
                'epoch/loss': avg_loss,
                'epoch/epoch': epoch + 1
            }, step=global_step)

        # epoch结束时验证采样
        validate_sampling(model, tokenizer, model_config, train_config, num_samples=2)

        # 每save_every个epoch保存checkpoint
        if (epoch + 1) % train_config.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                avg_loss, best_loss, model_config, train_config,
                f"checkpoint_epoch_{epoch + 1}.pt"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(train_config.save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'loss': best_loss
            }, best_path)
            print(f"Saved best model to {best_path}")

    if train_config.use_wandb:
        wandb.finish()

    print("\nTraining completed!")
    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
