"""
水印采样器模块
职责：包含所有与行为采样相关的算法
"""

import random
import math
import torch
import hmac
import hashlib
import numpy as np
import os


# ==============================================================================
# ================ 上下文密钥生成 ================
# ==============================================================================

def generate_contextual_key(history_responses, num_bytes=32):
    """
    基于历史响应（上下文）生成确定性密钥
    
    Args:
        history_responses (list): 过去行为描述字符串的列表
        num_bytes (int): 生成密钥的字节数（默认 32，对应 SHA-256）

    Returns:
        bytes: 生成的密钥
        
    Example:
        >>> history = ["用户点赞了视频", "用户收藏了视频"]
        >>> key = generate_contextual_key(history)
        >>> len(key)
        32
    """
    if not history_responses:
        # 冷启动：如果历史为空，使用固定的初始字符串
        context_string = "INITIAL_CONTEXT_FOR_AGENT_WATERMARK"
    else:
        # 使用简单策略：使用最近的响应作为上下文
        # 这里使用最后一个响应，可以扩展为多个响应的连接
        context_string = history_responses[-1]
        
    # 使用 SHA-256 哈希函数将上下文字符串转换为固定长度的密钥
    hasher = hashlib.sha256()
    hasher.update(context_string.encode('utf-8'))
    return hasher.digest()[:num_bytes]


# ==============================================================================
# ================ 差分方案水印引擎 ================
# ==============================================================================

# 伪随机生成器（PRG/DRBG），确保发送方和接收方可以同步随机过程
class DRBG:
    def __init__(self, key, nonce):
        self.key = key
        self.nonce = nonce
        self.counter = 0

    def generate_random_bits(self, n):
        message = self.nonce + self.counter.to_bytes(4, 'big')
        hmac_sha512 = hmac.new(self.key, message, hashlib.sha512).digest()
        self.counter += 1
        
        bits = ''.join(format(byte, '08b') for byte in hmac_sha512)
        return bits[:n]

    def generate_random(self, n):
        # 从位串生成 (0,1) 范围内的浮点数
        random_bits = self.generate_random_bits(n)
        random_int = int(random_bits, 2)
        random_float = random_int / (2**n)
        return random_float

# 均匀循环移位编码器（根据秘密信息在选定的"桶"内选择一个项目）
# 标准版本 - 与 Artifacts 实现一致
def uni_cyclic_shift_enc(bit_stream, n, PRG, precision=52):
    """
    循环移位均匀隐写编码器（Artifacts 标准版本）
    
    Args:
        bit_stream (str): 要嵌入的位流
        n (int): 桶大小
        PRG: 伪随机生成器
        precision (int): 精度参数
        
    Returns:
        tuple: (选定的索引, 嵌入的位串)
    """
    if n == 1:
        PRG.generate_random(n=precision)
        return 0, ''
    
    ptr = PRG.generate_random(n=precision)
    R = math.floor(ptr * n)
    
    k = math.floor(math.log2(n))
    t = n - 2**k
    
    # 检查位流是否足够
    if len(bit_stream) < k:
        # 位流不足，随机选择但消耗 PRG 以保持同步
        return R, ''
    
    bits = bit_stream[:k]
    
    # 检查是否需要额外的位
    if len(bit_stream) < k + 1:
        bits_res = '0'  # 默认值
    else:
        bits_res = bit_stream[k]
    
    idx_sort = lsb_bits2int([int(b) for b in bits])
    
    if idx_sort < 2**k - t:
        return (idx_sort + R) % n, bits
    else:
        return (2 * (idx_sort - (2**k - t)) + (2**k - t) + R + int(bits_res)) % n, bits + bits_res

# 差分重组模块（核心创新：水平切片）
# V2：使用稳定排序处理相等概率
def differential_based_recombination(prob, indices):
    bins = []
    
    # ========================== 使用稳定排序 ==========================
    # torch.argsort 返回对输入张量进行排序的索引张量
    # stable=True 确保当 prob 中的值相等时，对应的索引保持其原始顺序
    # 这是保证编码/解码同步的关键！
    # [修复] 对概率进行四舍五入以避免浮点噪声改变"相等"值的顺序
    prob_rounded = torch.round(prob * 1e8) / 1e8
    sorted_order_indices = torch.argsort(prob_rounded, stable=True, descending=False)
    
    # 使用这个确定性顺序重新排列 prob 和 indices
    prob = prob[sorted_order_indices]
    indices = indices[sorted_order_indices]
    # ==================================================================

    mask = prob > 0
    prob_nonzero = prob[mask]
    indices_nonzero = indices[mask]
 
    diff = torch.cat((prob_nonzero[:1], torch.diff(prob_nonzero, n=1)))
    n = len(prob_nonzero)

    weights = torch.arange(n, 0, -1, device = prob.device) 
    diff_positive = diff > 0

    prob_new = diff[diff_positive] * weights[diff_positive] 
    bins = torch.arange(n, device = prob.device)[diff_positive]

    return indices_nonzero, bins, prob_new

# 差分编码器（引擎组装）
def differential_based_encoder(prob, indices, bit_stream, bit_index, PRG, precision = 52, **kwargs):
    indices_nonzero, bins, prob_new = differential_based_recombination(prob, indices)
    if prob_new.sum() == 0: # 避免除以零
        # 如果所有概率相等，随机选择一个
        random_idx = int(PRG.generate_random(precision) * len(indices))
        return indices[random_idx].view(1,1), 0

    prob_new = prob_new/prob_new.sum()

    random_p = PRG.generate_random(n = precision)
    cdf = torch.cumsum(prob_new, dim=0)
    bin_indice_idx = torch.searchsorted(cdf, random_p).item()
    
    selected_bin_start_index = bins[bin_indice_idx]
    bin_content = indices_nonzero[selected_bin_start_index:]

    idx, bits = uni_cyclic_shift_enc(bit_stream=bit_stream[bit_index:], n = len(bin_content), PRG = PRG, precision=precision)
    
    num = len(bits)
    prev = bin_content[idx].view(1,1)

    return prev, num


# ==============================================================================
# ================ 基本采样算法 ================
# ==============================================================================

def sample_behavior(probabilities, seed=None, round_num=0, strategy="weighted", temperature=1.0):
    """
    根据概率从列表中选择行为（无水印版本）
    
    Args:
        probabilities (dict): 行为及其对应概率的字典
        seed (int, optional): 随机种子以确保可重现性
        round_num (int, optional): 当前轮次编号，基于固定种子引入变化
        strategy (str): 采样策略，选项：
            - "weighted": 加权随机采样（原始概率分布）
            - "greedy": 贪婪选择（选择概率最高的）
            - "temperature": 温度采样（使用温度参数调整概率分布）
        temperature (float): 温度参数，仅在 strategy="temperature" 时使用
            - temperature < 1.0: 更倾向于高概率动作
            - temperature = 1.0: 等同于加权采样
            - temperature > 1.0: 更均匀的分布
        
    Returns:
        str: 选定的行为
        
    Example:
        >>> probs = {"点赞": 0.3, "收藏": 0.2, "转发": 0.5}
        >>> sample_behavior(probs, seed=42, round_num=1, strategy="greedy")
        '转发'  # 总是选择概率最高的
    """
    # 设置随机种子
    if seed is not None:
        combined_seed = seed + round_num
        random.seed(combined_seed)
    
    # 获取行为列表和对应的概率列表
    behaviors = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # 确保概率总和为 1
    total = sum(probs)
    if total != 1.0:
        probs = [p/total for p in probs]
    
    if strategy == "greedy":
        # 贪婪策略：选择概率最高的动作（类似于官方 ALFWorld 评估模式）
        max_idx = probs.index(max(probs))
        selected_behavior = behaviors[max_idx]
    
    elif strategy == "temperature":
        # 温度采样：调整概率分布的"锐度"
        # 较低的温度意味着更倾向于高概率动作
        if temperature <= 0:
            raise ValueError("温度必须为正数")
        
        # 应用温度缩放
        scaled_probs = [p ** (1.0 / temperature) for p in probs]
        total_scaled = sum(scaled_probs)
        scaled_probs = [p / total_scaled for p in scaled_probs]
        
        # 使用缩放后的概率进行加权采样
        selected_behavior = random.choices(behaviors, weights=scaled_probs, k=1)[0]
    
    else:  # "weighted" 或默认
        # 加权随机采样：根据原始概率分布采样
        selected_behavior = random.choices(behaviors, weights=probs, k=1)[0]
    
    return selected_behavior


# ==============================================================================
# ================ 传统水印采样算法 ================
# ==============================================================================

def sample_behavior_watermark(probabilities, seed=None, round_num=0, prob_bias=0.5, ratio=0.5, BEHAVIOR_TYPES=[]):
    """
    根据概率从列表中随机选择行为，为某些行为添加概率偏置（旧概率偏置水印）
    
    Args:
        probabilities (dict): 行为及其对应概率的字典
        seed (int, optional): 随机种子以确保可重现性
        round_num (int, optional): 当前轮次编号，基于固定种子引入变化
        prob_bias (float, optional): 用于调整概率的概率偏置
        ratio (float, optional): 要偏置的行为比例，0-1，控制有多少 BEHAVIOR_TYPES 获得 prob_bias 添加
        BEHAVIOR_TYPES (list, optional): 行为类型列表

    Returns:
        tuple: (选定的行为, 添加了概率偏置的行为列表)
        
    Example:
        >>> probs = {"点赞": 0.3, "收藏": 0.2, "转发": 0.5}
        >>> behavior, biased_list = sample_behavior_watermark(probs, seed=42, round_num=1, prob_bias=0.5, ratio=0.5, BEHAVIOR_TYPES=['点赞', '收藏', '转发'])
        >>> print(f"选定：{behavior}，偏置：{biased_list}")
    """
    # 行为数量
    behavior_num = len(BEHAVIOR_TYPES)
    
    # 设置随机种子
    if seed is not None:
        # 组合种子和轮次编号创建新种子
        combined_seed = seed + round_num
        random.seed(combined_seed)
    
    # 根据 combined_seed 和 ratio 划分需要偏置的行为
    # 计算要偏置的行为数量
    biased_count = int(behavior_num * ratio)
    # 随机选择要偏置的行为
    add_logits_behavior_list = random.sample(BEHAVIOR_TYPES, biased_count)
    
    # 获取行为列表和对应的概率列表
    behaviors = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # 为选定的行为添加概率偏置
    modified_probs = []
    for behavior, prob in zip(behaviors, probs):
        if behavior in add_logits_behavior_list:
            modified_probs.append(prob + prob_bias)
        else:
            modified_probs.append(prob)
    
    # 确保概率总和为 1
    total = sum(modified_probs)
    if total != 1.0:
        modified_probs = [p/total for p in modified_probs]
    
    # 使用 random.choices 进行加权随机选择
    selected_behavior_watermark = random.choices(behaviors, weights=modified_probs, k=1)[0]
    
    return selected_behavior_watermark, add_logits_behavior_list


def sample_behavior_watermark_uncertainty(probabilities, seed=None, round_num=0, prob_bias=0.5, ratio=0.5, BEHAVIOR_TYPES=[], uncertainty_threshold=0.5):
    """
    根据概率随机选择行为，为某些行为添加偏置，并评估行为不确定性
    
    Args:
        probabilities (dict): 行为及其对应概率的字典
        seed (int, optional): 随机种子以确保可重现性
        round_num (int, optional): 当前轮次编号，基于固定种子引入变化
        prob_bias (float, optional): 用于调整概率的概率偏置
        ratio (float, optional): 要偏置的行为比例，0-1，控制有多少 BEHAVIOR_TYPES 获得 prob_bias 添加
        BEHAVIOR_TYPES (list, optional): 行为类型列表
        uncertainty_threshold (float, optional): 不确定性阈值，超过此值的行为被认为不稳定

    Returns:
        tuple: (选定的行为, 添加了概率偏置的行为列表, 是否应用水印, 不确定性分数)
        
    Example:
        >>> probs = {"点赞": 0.3, "收藏": 0.2, "转发": 0.5}
        >>> behavior, biased_list, is_stable, unc = sample_behavior_watermark_uncertainty(
        ...     probs, seed=42, round_num=1, prob_bias=0.5, ratio=0.5,
        ...     BEHAVIOR_TYPES=['点赞', '收藏', '转发'], uncertainty_threshold=0.5
        ... )
        >>> print(f"选定：{behavior}，偏置：{biased_list}，稳定：{is_stable}，不确定性：{unc}")
    """
    # Number of behaviors
    behavior_num = len(BEHAVIOR_TYPES)
    
    # Set random seed
    if seed is not None:
        # Combine seed and round number to create new seed
        combined_seed = seed + round_num
        random.seed(combined_seed)
    
    # Partition behaviors needing bias based on combined_seed and ratio
    # Calculate count of behaviors to bias
    biased_count = int(behavior_num * ratio)
    # Randomly select behaviors to bias
    add_logits_behavior_list = random.sample(BEHAVIOR_TYPES, biased_count)
    
    # Get behavior list and corresponding probability list
    behaviors = list(probabilities.keys())
    probs = list(probabilities.values())
    
    # Record original probabilities for later comparison
    original_probs = probs.copy()
    
    # Add probability bias to selected behaviors
    modified_probs = []
    for behavior, prob in zip(behaviors, probs):
        if behavior in add_logits_behavior_list:
            modified_probs.append(prob + prob_bias)
        else:
            modified_probs.append(prob)
    
    # Ensure probabilities sum to 1
    total = sum(modified_probs)
    if total != 1.0:
        modified_probs = [p/total for p in modified_probs]
    
    # Use random.choices for weighted random selection
    selected_behavior_watermark = random.choices(behaviors, weights=modified_probs, k=1)[0]
    
    # 计算不确定性
    # 注意 方法 1：计算修改后概率的前 1 和前 2 之间的差异
    # 对概率排序，第 1 和第 2 之间的差异表示置信度。差异越大意味着不确定性越低
    # 例如，如果最大概率 0.8，第 2 为 0.1，差异 0.7，选择很有信心
    sorted_probs = sorted(modified_probs, reverse=True)
    max_prob_diff = sorted_probs[0] - sorted_probs[1]
    
    # 注意 方法 2：计算原始概率和修改后概率之间的最大变化
    # 变化越大意味着水印影响越大，因此不确定性越高
    # 例如，原始 0.2，修改后 0.7，变化 0.5，影响很大
    prob_changes = [abs(m - o) for m, o in zip(modified_probs, original_probs)]
    max_prob_change = max(prob_changes)
    
    # 注意 方法 3：计算熵
    # 熵越高意味着分布越平坦，不确定性越高
    # 例如，均匀 [0.33, 0.33, 0.34]，熵 ~1，非常不确定
    # 集中 [0.9, 0.05, 0.05]，熵 ~0，非常确定
    entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in modified_probs)
    normalized_entropy = entropy / math.log2(len(behaviors))  # 归一化熵
    
    # 综合不确定性指标（可根据需要调整）
    uncertainty = (
        (1 - max_prob_diff) +  # 概率差异越小意味着不确定性越高
        max_prob_change +      # 概率变化越大意味着不确定性越高
        normalized_entropy     # 熵越高意味着不确定性越高
    ) / 3
    
    # 确定是否应该应用水印。不确定性越低意味着稳定性越高，更有可能开始水印
    is_stable = uncertainty < uncertainty_threshold
    
    return selected_behavior_watermark, add_logits_behavior_list, is_stable, uncertainty


# ==============================================================================
# ================ 差分水印采样 ================
# ==============================================================================

def sample_behavior_differential(probabilities, bit_stream, bit_index, context_for_key=None, history_responses=None, seed=None, round_num=0):
    """
    使用差分方案引擎选择行为并嵌入秘密信息（新差分水印方案）
    新引擎的适配器函数，支持基于上下文的动态密钥生成

    Args:
        probabilities (dict): 行为及其对应概率的字典
        bit_stream (str): 要嵌入的秘密信息位流
        bit_index (int): 位流中的起始索引
        context_for_key (str, optional): 用于密钥生成的显式上下文字符串（推荐）
        history_responses (list, optional): [已弃用] 历史响应列表，仅在 context_for_key 为 None 时使用
        seed (int, optional): 随机种子（回退，当前实现使用上下文密钥）
        round_num (int, optional): 当前轮次编号

    Returns:
        tuple: (选定的行为, 用于检测的目标行为列表, 嵌入的位数, 用于密钥的实际上下文)
        
    Example:
        >>> probs = {"点赞": 0.3, "收藏": 0.2, "转发": 0.5}
        >>> context = "response1||response2"
        >>> behavior, targets, bits, ctx = sample_behavior_differential(probs, "10110", 0, context_for_key=context, round_num=1)
        >>> print(f"选定：{behavior}，目标：{targets}，位数：{bits}")
    """
    # --- 1. 数据格式转换（为新引擎适配输入）---
    # 确保固定的行为顺序以保持索引一致
    behaviors = sorted(probabilities.keys())
    probs_list = [probabilities[b] for b in behaviors]
    
    # 转换为 PyTorch 张量
    # 强制使用 CPU 以避免大规模并行运行中的 CUDA 初始化开销
    device = 'cpu'
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)
    indices_tensor = torch.arange(len(behaviors), device=device)
    
    # --- 2. 初始化 PRG（基于上下文的动态密钥生成）---
    # 决定上下文：优先使用 context_for_key，否则从 history_responses 构建
    if context_for_key is not None:
        # 使用显式上下文字符串
        context_used = context_for_key
    else:
        # 向后兼容：从 history_responses 构建上下文
        if history_responses is None:
            history_responses = []
        # 使用滑动窗口（最近 3 个响应）构建上下文
        window_size = 3
        recent_responses = history_responses[-window_size:] if len(history_responses) > 0 else []
        context_used = "||".join(recent_responses) if recent_responses else ""
    
    # === 新方法：基于显式上下文字符串的密钥 ===
    key = generate_contextual_key([context_used])  # 作为列表传递以保持兼容性
    # nonce 使用轮次编号以确保每轮不同的随机序列
    nonce = str(round_num).encode('utf-8') 
    
    # === 旧方法：基于预共享种子的静态密钥（保留为注释）===
    # 如果需要回退到静态密钥，取消下面的注释：
    # if seed is None:
    #     seed = 42
    # combined_seed_str = str(seed)
    # round_num_str = str(round_num)
    # key = combined_seed_str.encode('utf-8')
    # nonce = round_num_str.encode('utf-8')
    
    PRG = DRBG(key, nonce)

    # --- 3. 调用新引擎核心 ---
    selected_idx_tensor, num_bits_embedded = differential_based_encoder(
        prob=probs_tensor,
        indices=indices_tensor,
        bit_stream=bit_stream,
        bit_index=bit_index,
        PRG=PRG
    )
    selected_idx = selected_idx_tensor.item()
    
    # --- 4. 转换输出并生成用于检测的"目标列表"---
    # 将选定的索引 ID 转换回行为字符串
    selected_behavior = behaviors[selected_idx]
    
    # 为了让检测器（detect_watermark.py）工作，我们需要重新计算选择了哪个"桶"
    # 检测器需要知道"目标范围"是什么
    PRG_for_detection = DRBG(key, nonce)  # 使用相同参数重新创建 PRG
    
    indices_nonzero, bins, prob_new = differential_based_recombination(probs_tensor, indices_tensor)
    prob_new = prob_new / prob_new.sum()
    
    random_p = PRG_for_detection.generate_random(n=52)
    cdf = torch.cumsum(prob_new, dim=0)
    bin_indice_idx = torch.searchsorted(cdf, random_p).item()

    selected_bin_start_index = bins[bin_indice_idx]
    bin_content_indices = indices_nonzero[selected_bin_start_index:]
    
    # 这相当于旧引擎中的"绿名单"
    target_behavior_list = [behaviors[i] for i in bin_content_indices]

    if os.getenv("AGENTMARK_DEBUG_SAMPLER"):
        debug_payload = {
            "stage": "bin_select",
            "random_p": float(random_p),
            "cdf": [float(x) for x in cdf.tolist()],
            "bin_indice_idx": int(bin_indice_idx),
            "selected_bin_start_index": int(selected_bin_start_index),
            "bin_content": target_behavior_list,
        }
        print(f"[agentmark:sampler] {json.dumps(debug_payload, ensure_ascii=True)}")
    
    return selected_behavior, target_behavior_list, num_bits_embedded, context_used


# ==============================================================================
# ================ 差分水印解码器 ================
# ==============================================================================

def lsb_bits2int(bits):
    """
    将位列表转换为整数（LSB 优先）
    
    Args:
        bits (list): 位列表，例如 [1, 0, 1] 表示二进制 101（LSB 优先）
        
    Returns:
        int: 对应的整数值
        
    Example:
        >>> lsb_bits2int([1, 0, 1])  # LSB: 1*1 + 0*2 + 1*4 = 5
        5
    """
    result = 0
    for i, bit in enumerate(bits):
        result += bit * (2 ** i)
    return result


def lsb_int2bits(num, length):
    """
    将整数转换为位列表（LSB 优先）
    
    Args:
        num (int): 要转换的整数
        length (int): 位列表的长度
        
    Returns:
        list: 位列表（LSB 优先）
        
    Example:
        >>> lsb_int2bits(5, 3)  # 5 = 101(二进制) -> [1, 0, 1]（LSB 优先）
        [1, 0, 1]
    """
    bits = []
    for _ in range(length):
        bits.append(num % 2)
        num //= 2
    return bits


def uni_cyclic_shift_dec(idx, n, PRG, precision=52):
    """
    均匀循环移位解码器（Artifacts 标准版本）
    对应编码器 uni_cyclic_shift_enc，从选定的索引中提取秘密位
    
    必须与编码器的 PRG 调用顺序保持一致！
    
    Args:
        idx (int): 选定的索引位置（桶中的相对位置）
        n (int): 桶大小
        PRG: 伪随机生成器
        precision (int): 精度参数
        
    Returns:
        str: 提取的位串
    """
    if n == 1:
        PRG.generate_random(n=precision)
        return ''
    
    # 必须与编码器相同，首先生成 R
    ptr = PRG.generate_random(n=precision)
    R = math.floor(ptr * n)
    
    k = math.floor(math.log2(n))
    t = n - 2**k
    
    # 反向循环移位
    idx_sort = (idx - R) % n
    
    if idx_sort < 2**k - t:
        bits = lsb_int2bits(idx_sort, k)
        bits = "".join([str(_) for _ in bits])
        return bits
    else:
        s1 = idx_sort - 2**k + t
        s_last = s1 % 2
        
        bits = lsb_int2bits((s1 - s_last) // 2 + 2**k - t, k)
        bits = "".join([str(_) for _ in bits])
        
        if s_last == 0:
            return bits + '0'
        else:
            return bits + '1'


def differential_based_decoder(probabilities, selected_behavior, context_for_key=None, history_responses=None, round_num=0):
    """
    差分水印解码器 - 从选定的行为中提取嵌入的秘密位
    
    Args:
        probabilities (dict): 行为及其对应概率的字典
        selected_behavior (str): 实际选定的行为
        context_for_key (str, optional): 用于密钥生成的显式上下文字符串（推荐从日志中读取）
        history_responses (list, optional): [已弃用] 历史响应列表，仅在 context_for_key 为 None 时使用
        round_num (int): 当前轮次编号（必须与编码时相同）
        
    Returns:
        str: 提取的位串
        
    Example:
        >>> probs = {"点赞": 0.3, "收藏": 0.2, "转发": 0.5}
        >>> context = "response1||response2"
        >>> bits = differential_based_decoder(probs, "转发", context_for_key=context, round_num=1)
        >>> print(f"提取的位：{bits}")
    """
    # --- 1. 数据格式转换 ---
    behaviors = sorted(probabilities.keys())
    probs_list = [probabilities[b] for b in behaviors]
    
    # 转换为 PyTorch 张量
    # 强制使用 CPU 以避免大规模并行运行中的 CUDA 初始化开销
    device = 'cpu'
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)
    indices_tensor = torch.arange(len(behaviors), device=device)
    
    # 查找选定行为的索引
    try:
        selected_idx = behaviors.index(selected_behavior)
    except ValueError:
        print(f"警告：选定的行为 '{selected_behavior}' 不在行为列表中")
        return ''
    
    prev_tensor = torch.tensor([selected_idx], device=device)
    
    # --- 2. 初始化 PRG（必须与编码完全相同）---
    # 决定上下文：优先使用 context_for_key
    if context_for_key is not None:
        context_used = context_for_key
    else:
        # 向后兼容：从 history_responses 构建
        if history_responses is None:
            history_responses = []
        window_size = 3
        recent_responses = history_responses[-window_size:] if len(history_responses) > 0 else []
        context_used = "||".join(recent_responses) if recent_responses else ""
    
    key = generate_contextual_key([context_used])
    nonce = str(round_num).encode('utf-8')
    PRG = DRBG(key, nonce)
    
    # --- 3. 概率重组（与编码器相同）---
    indices_nonzero, bins, prob_new = differential_based_recombination(probs_tensor, indices_tensor)
    
    if prob_new.sum() == 0:
        return ''
    
    prob_new = prob_new / prob_new.sum()
    
    # --- 4. 桶采样（与编码器相同）---
    random_p = PRG.generate_random(n=52)
    cdf = torch.cumsum(prob_new, dim=0)
    bin_indice_idx = torch.searchsorted(cdf, random_p).item()
    
    selected_bin_start_index = bins[bin_indice_idx]
    bin_content = indices_nonzero[selected_bin_start_index:]
    
    # --- 5. 均匀隐写解码 ---
    # 在桶中查找选定索引的位置
    try:
        idx_in_bin = (bin_content == prev_tensor.item()).nonzero().item()
    except (RuntimeError, ValueError):
        # 如果选定的行为不在桶中，出了问题
        print(f"警告：选定的行为不在预期的桶中，无法解码")
        return ''
    
    # 使用循环移位解码器提取位
    bits = uni_cyclic_shift_dec(idx=idx_in_bin, n=len(bin_content), PRG=PRG, precision=52)
    
    return bits
# ==============================================================================
# ================ Red-Green List Sampling Algorithms ================
# ==============================================================================

def sample_behavior_red_green(probabilities, context_for_key=None, history_responses=None, seed=None, round_num=0, gamma=0.5, delta=2.0):
    """
    Use Red-Green List strategy (KGW Style) for behavior sampling.
    
    Args:
        probabilities (dict): Behavior and their raw probabilities.
        context_for_key (str): Context info, used to generate random seed.
        history_responses (list): Backup context.
        seed (int): Backup seed.
        round_num (int): Round number, introducing time variance.
        gamma (float): Green list ratio (0.0 - 1.0). E.g., 0.5 means half behaviors are green list.
        delta (float): Logit bias value. Green list behaviors' logits will increase by delta.
        
    Returns:
        tuple: (Selected behavior, Green list, 0 bits, context_used)
    """
    # 1. Prepare Data
    behaviors = sorted(probabilities.keys())
    probs_list = [probabilities[b] for b in behaviors]
    # Force CPU to avoid CUDA initialization overhead in massive parallel runs
    device = 'cpu'
    
    # Convert probabilities to Logits (Inverse Softmax is not unique, assume raw Logits is log(p))
    # Add a small value to avoid log(0)
    epsilon = 1e-9
    probs_tensor = torch.tensor(probs_list, dtype=torch.float32, device=device)
    logits = torch.log(probs_tensor + epsilon)
    
    # 2. Generate Random Seed (Hash Context)
    if context_for_key is not None:
        context_used = context_for_key
    else:
        window_size = 3
        recent_responses = history_responses[-window_size:] if history_responses else []
        context_used = "||".join(recent_responses)
        
    # Generate hash as pseudo-random source
    # Note: To make red-green list independent for each behavior, typically hash(context + behavior)
    # But for efficiency and convenience of list return, here we generate a context-based random vector
    
    key = generate_contextual_key([context_used])
    nonce = str(round_num).encode('utf-8')
    PRG = DRBG(key, nonce)
    
    # 3. Partition Red-Green List
    # Generate a random number in [0, 1] for each behavior
    # To ensure behavior order irrelevance, strictly should use hash(context + behavior_name)
    # But as long as behaviors list sort order is fixed, using PRG sequence is also deterministic and efficient
    
    green_list = []
    
    # Generate len(behaviors) random numbers
    random_vals = [PRG.generate_random(32) for _ in range(len(behaviors))]
    
    mask = torch.zeros_like(logits, device=device)
    
    for i, r_val in enumerate(random_vals):
        if r_val < gamma:
            # Enter Green List
            green_list.append(behaviors[i])
            mask[i] = 1.0
            
    # 4. Apply Watermark (Logit Bias)
    # Green List Logits increase by delta
    watermarked_logits = logits + (mask * delta)
    
    # 5. Sampling
    # Normalize using Softmax
    watermarked_probs = torch.softmax(watermarked_logits, dim=0)
    
    # Convert to Python list for weighted random
    final_probs = watermarked_probs.tolist()
    
    # Sampling
    # To maintain determinism, we can continue using PRG or use externally provided global seed
    # To match AgentMark style, use random.choices (depends on global seed or loop seed)
    # But considering sample_behavior_differential uses PRG efficiently, ideally PRG here too
    
    # Use next random number from PRG for sampling (Inverse Transform Sampling)
    rand_p = PRG.generate_random(52)
    cdf = torch.cumsum(watermarked_probs, dim=0)
    idx = torch.searchsorted(cdf, rand_p).item()
    idx = min(idx, len(behaviors) - 1) # Boundary protection
    
    selected_behavior = behaviors[idx]
    
    return selected_behavior, green_list, 0, context_used
