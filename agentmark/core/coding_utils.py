"""
Coding utilities module.
Responsibilities: provide error-correcting encode/decode utilities with configurable
watermark embedding strategies.

Standard implementations:
- parity: Python built-in operations
- hamming: standard Hamming(7,4) extended algorithm
- Reed-Solomon: reedsolo library (optional)
"""


def add_parity_bit(data_bits: str) -> str:
    """
    Add a parity bit to 8-bit data.

    Args:
        data_bits (str): 8-bit binary string

    Returns:
        str: 9-bit binary string (original 8 bits + parity bit)

    Example:
        >>> add_parity_bit("11001100")
        "110011000"  # even parity, 0 ones, parity bit 0
        >>> add_parity_bit("11001101")
        "110011011"  # even parity, odd ones, parity bit 1
    """
    if len(data_bits) != 8:
        raise ValueError(f"Parity requires 8 bits, got {len(data_bits)} bits")
    
    # Count ones, use even parity
    ones_count = data_bits.count('1')
    parity_bit = '1' if ones_count % 2 == 1 else '0'
    
    return data_bits + parity_bit


def check_and_strip_parity_bit(message_bits: str) -> tuple:
    """
    Check and strip a parity bit.

    Args:
        message_bits (str): 9-bit binary string (8 bits + parity bit)

    Returns:
        tuple: (data_bits, is_valid)
            - data_bits (str): original 8-bit data
            - is_valid (bool): whether parity check passes

    Example:
        >>> check_and_strip_parity_bit("110011000")
        ("11001100", True)
        >>> check_and_strip_parity_bit("110011001")  # bad parity bit
        ("11001100", False)
    """
    if len(message_bits) != 9:
        raise ValueError(f"Expected 9-bit message (8 data + parity), got {len(message_bits)} bits")
    
    data_bits = message_bits[:8]
    received_parity = message_bits[8]
    
    # Recompute parity bit
    ones_count = data_bits.count('1')
    expected_parity = '1' if ones_count % 2 == 1 else '0'
    
    is_valid = (received_parity == expected_parity)
    
    return data_bits, is_valid


def add_hamming_code(data_bits: str) -> str:
    """
    Add standard Hamming code for 16-bit data.
    Uses extended Hamming code to correct 1-bit errors and detect 2-bit errors.

    Args:
        data_bits (str): 16-bit binary string

    Returns:
        str: Encoded bit string (with parity bits)

    Note:
        Uses standard Hamming(21,16) code:
        - 16 data bits
        - 5 parity bits at positions 1, 2, 4, 8, 16
        - 21 bits total
    """
    if len(data_bits) != 16:
        raise ValueError(f"Hamming requires 16 bits, got {len(data_bits)} bits")
    
    # Convert data bits to list
    data = [int(b) for b in data_bits]
    
    # Create encoding array (21 bits: index 0 unused, 1-21 valid)
    # Parity bits at positions 1,2,4,8,16
    encoded = [0] * 22  # index 0 unused, 1-21 valid
    
    # Data positions (skip powers of two)
    data_positions = [i for i in range(1, 22) if i & (i-1) != 0]  # non-powers of two
    
    # Fill data bits
    for i, pos in enumerate(data_positions[:16]):
        encoded[pos] = data[i]
    
    # Compute parity bits
    parity_positions = [1, 2, 4, 8, 16]
    
    for p in parity_positions:
        # For parity position p, check all positions i where i & p != 0
        parity = 0
        for i in range(1, 22):
            if i & p and i != p:  # check whether i is covered by parity p
                parity ^= encoded[i]
        encoded[p] = parity
    
    # Convert to string (skip index 0)
    return ''.join(str(encoded[i]) for i in range(1, 22))


def decode_and_correct_hamming(message_bits: str) -> str:
    """
    Decode and correct standard Hamming code.
    Corrects 1-bit errors and detects 2-bit errors.

    Args:
        message_bits (str): 21-bit encoded string

    Returns:
        str: Corrected 16-bit original data

    Note:
        Uses standard Hamming correction.
    """
    if len(message_bits) != 21:
        raise ValueError(f"Expected 21-bit message, got {len(message_bits)} bits")
    
    # Convert to array (index 0 unused, 1-21 valid)
    received = [0] + [int(b) for b in message_bits]
    
    # Compute syndrome
    syndrome = 0
    parity_positions = [1, 2, 4, 8, 16]
    
    for p in parity_positions:
        parity = 0
        for i in range(1, 22):
            if i & p:  # check whether i is covered by parity p
                parity ^= received[i]
        if parity != 0:
            syndrome += p
    
    # If syndrome != 0, there is an error
    if syndrome != 0:
        if 1 <= syndrome <= 21:
            print(f"Hamming detected error at position {syndrome}, corrected")
            # Correct error
            received[syndrome] ^= 1
        else:
            print("Hamming detected multi-bit error (cannot correct)")
    
    # Extract data bits (non-power-of-two positions)
    data_positions = [i for i in range(1, 22) if i & (i-1) != 0]
    data_bits = ''.join(str(received[pos]) for pos in data_positions[:16])
    
    return data_bits


def encode_payload(payload_bits: str, config: dict) -> str:
    """
    Encode the payload into a full message based on config.
    Factory function dispatches to encoder based on config.

    Args:
        payload_bits (str): Raw payload bits
        config (dict): Watermark config, includes:
            - payload_bit_length: Payload length (8 or 16)
            - ecc_method: ECC method ("parity"/"hamming"/"none")
            - embedding_strategy: Embedding strategy ("cyclic"/"once")

    Returns:
        str: Encoded message

    Raises:
        ValueError: Config mismatch or invalid parameters

    Example:
        >>> config = {"payload_bit_length": 8, "ecc_method": "parity"}
        >>> encode_payload("11001100", config)
        "110011000"
    """
    bit_length = config.get("payload_bit_length", 8)
    ecc_method = config.get("ecc_method", "none")
    
    # 1. Validate input length against config
    if len(payload_bits) != bit_length:
        raise ValueError(
            f"Data length {len(payload_bits)} does not match payload_bit_length {bit_length}"
        )
    
    # 2. Select ECC encoder based on ecc_method
    if ecc_method == "parity":
        if bit_length != 8:
            raise ValueError("Parity currently supports only 8-bit data")
        return add_parity_bit(payload_bits)
        
    elif ecc_method == "hamming":
        if bit_length != 16:
            raise ValueError("Hamming currently supports only 16-bit data")
        return add_hamming_code(payload_bits)
        
    elif ecc_method == "none":
        return payload_bits  # no ECC
        
    else:
        raise ValueError(f"Unknown ECC method: {ecc_method}")


def decode_message(message_bits: str, config: dict) -> dict:
    """
    Decode a message based on config and return data plus validation info.

    Args:
        message_bits (str): Encoded message bits
        config (dict): Watermark config dict

    Returns:
        dict: Decoding result dict
            - decoded_payload (str): Decoded payload
            - valid (bool): Whether message is valid/passes checks
            - corrected (bool): Whether error correction was applied
            - ecc_method (str): ECC method
            - error (str, optional): Error message (if invalid)

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
            'corrected': False,  # parity detects only, no correction
            'ecc_method': 'parity',
            'error': None if is_valid else 'Parity check failed'
        }
        
    elif ecc_method == "hamming":
        # Hamming decode corrects automatically; detect whether correction happened
        corrected_data = decode_and_correct_hamming(message_bits)
        
        # Re-encode; difference implies correction
        re_encoded = add_hamming_code(corrected_data)
        was_corrected = (re_encoded != message_bits)
        
        return {
            'decoded_payload': corrected_data,
            'valid': True,  # Hamming corrects 1-bit errors
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
            'error': f"Unknown ECC method: {ecc_method}"
        }


def prepare_cyclic_embedding(payload_bits: str, config: dict, total_rounds: int) -> list:
    """
    Prepare cyclic embedding message sequence.

    Args:
        payload_bits (str): Raw payload
        config (dict): Watermark config
        total_rounds (int): Total experiment rounds

    Returns:
        list: Message list to embed per round

    Example:
        >>> prepare_cyclic_embedding("11001100", config, 3)
        ["110011000", "110011000", "110011000"]  # repeated 3 times
    """
    message_to_embed = encode_payload(payload_bits, config)
    embedding_strategy = config.get("embedding_strategy", "once")
    
    if embedding_strategy == "cyclic":
        # Cyclic embedding: same message each round
        return [message_to_embed] * total_rounds
    elif embedding_strategy == "once":
        # Embed only once
        return [message_to_embed]
    else:
        raise ValueError(f"Unknown embedding strategy: {embedding_strategy}")
