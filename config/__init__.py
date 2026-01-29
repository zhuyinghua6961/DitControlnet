"""
ControlNet-DiT Configuration Package
"""

import os
import yaml
from pathlib import Path


def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        # 默认配置文件路径
        config_path = Path(__file__).parent / 'config.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_path=None):
    """保存配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)