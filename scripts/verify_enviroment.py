import torch
import bitsandbytes as bnb

# 1. 验证 BF16 (创新点：Ampere 显卡专属，提升训练稳定性) [cite: 40, 431, 433]
print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}") 

# 2. 验证 8-bit 优化器 (创新点：降低优化器显存至 1/4) [cite: 62, 228, 443]
try:
    dummy_param = torch.nn.Parameter(torch.zeros(1).cuda())
    optimizer = bnb.optim.AdamW8bit([dummy_param], lr=1e-5)
    print("8-bit AdamW Ready: True")
except Exception as e:
    print(f"8-bit AdamW Error: {e}")

# 3. 验证 Flash Attention (创新点：解决序列长度平方级显存增长问题) [cite: 8, 450]
# 注意：flash-attn 安装较慢，可暂时跳过，使用 xformers 作为替代
print("Flash Attention: Skipped (use xformers instead)")