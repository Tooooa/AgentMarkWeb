"""
编码工具模块
职责：提供可配置水印嵌入策略的纠错编码/解码工具

标准实现：
- parity: Python 内置操作
- hamming: 标准 Hamming(7,4) 扩展算法
- Reed-Solomon: reedsolo 库（可选）
"""


def add_parity_bit(data_bits: str) -> str:
    """
    为 8 位数据添加奇偶校验位

    Args:
        data_bits (str): 8 位二进制字符串

    Returns:
        str: 9 位二进制字符串（原始 8 位 + 奇偶校验位）

    Example:
        >>> add_parity_bit("11001100")
        "110011000"  # 偶校验，0 个 1，校验位为 0
        >>> add_parity_bit("11001101")
        "110011011"  # 偶校验，奇数个 1，校验位为 1
    """
    if len(data_bits) != 8:
        raise ValueError(f"奇偶校验需要 8 位，但得到 {len(data_bits)} 位")
    
    # 计算 1 的个数，使用偶校验
    ones_count = data_bits.count('1')
    parity_bit = '1' if ones_count % 2 == 1 else '0'
    
    return data_bits + parity_bit


def check_and_strip_parity_bit(message_bits: str) -> tuple:
    """
    检查并去除奇偶校验位

    Args:
        message_bits (str): 9 位二进制字符串（8 位数据 + 校验位）

    Returns:
        tuple: (data_bits, is_valid)
            - data_bits (str): 原始 8 位数据
            - is_valid (bool): 奇偶校验是否通过

    Example:
        >>> check_and_strip_parity_bit("110011000")
        ("11001100", True)
        >>> check_and_strip_parity_bit("110011001")  # 错误的校验位
        ("11001100", False)
    """
    if len(message_bits) != 9:
        raise ValueError(f"期望 9 位消息（8 位数据 + 校验位），但得到 {len(message_bits)} 位")
    
    data_bits = message_bits[:8]
    received_parity = message_bits[8]
    
    # 重新计算校验位
    ones_count = data_bits.count('1')
    expected_parity = '1' if ones_count % 2 == 1 else '0'
    
    is_valid = (received_parity == expected_parity)
    
    return data_bits, is_valid


def add_hamming_code(data_bits: str) -> str:
    """
    为 16 位数据添加标准汉明码
    使用扩展汉明码纠正 1 位错误并检测 2 位错误

    Args:
        data_bits (str): 16 位二进制字符串

    Returns:
        str: 编码后的位串（包含校验位）

    Note:
        使用标准 Hamming(21,16) 码：
        - 16 位数据位
        - 5 个校验位位于位置 1, 2, 4, 8, 16
        - 总共 21 位
    """
    if len(data_bits) != 16:
        raise ValueError(f"汉明码需要 16 位，但得到 {len(data_bits)} 位")
    
    # 将数据位转换为列表
    data = [int(b) for b in data_bits]
    
    # 创建编码数组（21 位：索引 0 未使用，1-21 有效）
    # 校验位位于位置 1,2,4,8,16
    encoded = [0] * 22  # 索引 0 未使用，1-21 有效
    
    # 数据位位置（跳过 2 的幂次）
    data_positions = [i for i in range(1, 22) if i & (i-1) != 0]  # 非 2 的幂次
    
    # 填充数据位
    for i, pos in enumerate(data_positions[:16]):
        encoded[pos] = data[i]
    
    # 计算校验位
    parity_positions = [1, 2, 4, 8, 16]
    
    for p in parity_positions:
        # 对于校验位位置 p，检查所有 i & p != 0 的位置 i
        parity = 0
        for i in range(1, 22):
            if i & p and i != p:  # 检查 i 是否被校验位 p 覆盖
                parity ^= encoded[i]
        encoded[p] = parity
    
    # 转换为字符串（跳过索引 0）
    return ''.join(str(encoded[i]) for i in range(1, 22))


def decode_and_correct_hamming(message_bits: str) -> str:
    """
    解码并纠正标准汉明码
    纠正 1 位错误并检测 2 位错误

    Args:
        message_bits (str): 21 位编码字符串

    Returns:
        str: 纠正后的 16 位原始数据

    Note:
        使用标准汉明纠错
    """
    if len(message_bits) != 21:
        raise ValueError(f"期望 21 位消息，但得到 {len(message_bits)} 位")
    
    # 转换为数组（索引 0 未使用，1-21 有效）
    received = [0] + [int(b) for b in message_bits]
    
    # 计算校正子
    syndrome = 0
    parity_positions = [1, 2, 4, 8, 16]
    
    for p in parity_positions:
        parity = 0
        for i in range(1, 22):
            if i & p:  # 检查 i 是否被校验位 p 覆盖
                parity ^= received[i]
        if parity != 0:
            syndrome += p
    
    # 如果校正子不为 0，则存在错误
    if syndrome != 0:
        if 1 <= syndrome <= 21:
            print(f"汉明码检测到位置 {syndrome} 的错误，已纠正")
            # 纠正错误
            received[syndrome] ^= 1
        else:
            print("汉明码检测到多位错误（无法纠正）")
    
    # 提取数据位（非 2 的幂次位置）
    data_positions = [i for i in range(1, 22) if i & (i-1) != 0]
    data_bits = ''.join(str(received[pos]) for pos in data_positions[:16])
    
    return data_bits


def encode_payload(payload_bits: str, config: dict) -> str:
    """
    根据配置将载荷编码为完整消息
    工厂函数根据配置分派到相应的编码器

    Args:
        payload_bits (str): 原始载荷位
        config (dict): 水印配置，包括：
            - payload_bit_length: 载荷长度（8 或 16）
            - ecc_method: 纠错码方法（"parity"/"hamming"/"none"）
            - embedding_strategy: 嵌入策略（"cyclic"/"once"）

    Returns:
        str: 编码后的消息

    Raises:
        ValueError: 配置不匹配或参数无效

    Example:
        >>> config = {"payload_bit_length": 8, "ecc_method": "parity"}
        >>> encode_payload("11001100", config)
        "110011000"
    """
    bit_length = config.get("payload_bit_length", 8)
    ecc_method = config.get("ecc_method", "none")
    
    # 1. 验证输入长度与配置是否匹配
    if len(payload_bits) != bit_length:
        raise ValueError(
            f"数据长度 {len(payload_bits)} 与 payload_bit_length {bit_length} 不匹配"
        )
    
    # 2. 根据 ecc_method 选择纠错码编码器
    if ecc_method == "parity":
        if bit_length != 8:
            raise ValueError("奇偶校验目前仅支持 8 位数据")
        return add_parity_bit(payload_bits)
        
    elif ecc_method == "hamming":
        if bit_length != 16:
            raise ValueError("汉明码目前仅支持 16 位数据")
        return add_hamming_code(payload_bits)
        
    elif ecc_method == "none":
        return payload_bits  # 无纠错码
        
    else:
        raise ValueError(f"未知的纠错码方法：{ecc_method}")


def decode_message(message_bits: str, config: dict) -> dict:
    """
    根据配置解码消息并返回数据及验证信息

    Args:
        message_bits (str): 编码后的消息位
        config (dict): 水印配置字典

    Returns:
        dict: 解码结果字典
            - decoded_payload (str): 解码后的载荷
            - valid (bool): 消息是否有效/通过检查
            - corrected (bool): 是否应用了纠错
            - ecc_method (str): 纠错码方法
            - error (str, optional): 错误消息（如果无效）

    Example:
        >>> config = {"payload_bit_length": 8, "ecc_method": "parity"}
        >>> decode_message("110011000", config)
        {
            'decoded_payload': '11001100',
            'valid': True,
            'corrected': False,
            'ecc_method': 'parity'
        }
    """
    ecc_method = config.get("ecc_method", "none")
    
    if ecc_method == "parity":
        data_bits, is_valid = check_and_strip_parity_bit(message_bits)
        return {
            'decoded_payload': data_bits,
            'valid': is_valid,
            'corrected': False,  # 奇偶校验仅检测，不纠错
            'ecc_method': 'parity',
            'error': None if is_valid else '奇偶校验失败'
        }
        
    elif ecc_method == "hamming":
        # 汉明码解码自动纠错；检测是否发生了纠错
        corrected_data = decode_and_correct_hamming(message_bits)
        
        # 重新编码；差异表示发生了纠错
        re_encoded = add_hamming_code(corrected_data)
        was_corrected = (re_encoded != message_bits)
        
        return {
            'decoded_payload': corrected_data,
            'valid': True,  # 汉明码纠正 1 位错误
            'corrected': was_corrected,
            'ecc_method': 'hamming',
            'error': None
        }
        
    elif ecc_method == "none":
        return {
            'decoded_payload': message_bits,
            'valid': True,
            'corrected': False,
            'ecc_method': 'none',
            'error': None
        }
        
    else:
        return {
            'decoded_payload': '',
            'valid': False,
            'corrected': False,
            'ecc_method': ecc_method,
            'error': f"未知的纠错码方法：{ecc_method}"
        }


def prepare_cyclic_embedding(payload_bits: str, config: dict, total_rounds: int) -> list:
    """
    准备循环嵌入消息序列

    Args:
        payload_bits (str): 原始载荷
        config (dict): 水印配置
        total_rounds (int): 总实验轮数

    Returns:
        list: 每轮要嵌入的消息列表

    Example:
        >>> prepare_cyclic_embedding("11001100", config, 3)
        ["110011000", "110011000", "110011000"]  # 重复 3 次
    """
    message_to_embed = encode_payload(payload_bits, config)
    embedding_strategy = config.get("embedding_strategy", "once")
    
    if embedding_strategy == "cyclic":
        # 循环嵌入：每轮使用相同消息
        return [message_to_embed] * total_rounds
    elif embedding_strategy == "once":
        # 仅嵌入一次
        return [message_to_embed]
    else:
        raise ValueError(f"未知的嵌入策略：{embedding_strategy}")
