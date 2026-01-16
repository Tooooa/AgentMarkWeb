
import random
import numpy as np

class DeterministicRLNC:
    """
    GF(2) 上的确定性随机线性网络编码
    从固定的载荷密钥生成无限的编码位流
    
    第 i 个编码位是载荷位的线性组合，
    其中系数是基于流 ID（密钥）和位索引（i）确定性生成的
    """
    def __init__(self, payload_bits_str, stream_key=42):
        """
        Args:
            payload_bits_str (str): 要编码的载荷，例如 "10110011"
            stream_key (int/str): 用于随机化系数的种子/密钥
        """
        self.payload = [int(b) for b in payload_bits_str]
        self.n = len(self.payload)
        self.stream_key = stream_key
        # 如果需要，可以缓存生成的位，但简单生成已经足够快

    def get_bit(self, index):
        """
        获取第 i 个编码位
        c_i = sum(coeff_j * p_j) mod 2，其中 coeff_j 来自 PRG(stream_key, index)
        """
        coeffs = self._generate_coeffs(index)
        
        coded_val = 0
        for b_val, c_val in zip(self.payload, coeffs):
            coded_val ^= (b_val & c_val)
        return str(coded_val)

    def get_stream(self, start_index, length):
        """
        获取一系列编码位
        """
        return "".join([self.get_bit(i) for i in range(start_index, start_index + length)])

    def _generate_coeffs(self, index):
        """
        为给定索引确定性生成 n 个系数
        """
        # 使用 (stream_key, index) 的哈希作为种子，或使用固定种子的简单 Random
        # Python 的 random 是 Mersenne Twister，对于加密不安全但对于编码分布足够
        # 为了确保绝对的寻址能力而不需要回退，我们需要基于哈希的 PRG 或重新设置种子
        # 重新设置种子对于小 n 来说足够快
        
        # 将密钥和索引组合成唯一的种子
        # 高效的字符串种子
        seed_val = f"{self.stream_key}_{index}"
        rd = random.Random(seed_val)
        
        # 在 GF(2) 中，系数是 0 或 1
        # 理想情况下我们需要非零行，但随机对于 RLNC 来说是可以的（零行的概率是 1/2^n）
        # 如果需要，我们可以强制非零，但标准 RLNC 不严格要求每个数据包都这样
        return [rd.randint(0, 1) for _ in range(self.n)]

    def decode(self, received_indices, received_bits):
        """
        尝试从一组接收到的编码位解码载荷
        
        Args:
            received_indices (list[int]): 接收到的编码位的索引
            received_bits (list[int/str]): 接收到的编码位的值
            
        Returns:
            str: 如果成功则返回解码的载荷位，否则返回 None
        """
        m = len(received_indices)
        if m < self.n:
            return None
        
        # 构建系统 Matrix * Payload = Received
        matrix = []
        vector = []
        
        for idx, val in zip(received_indices, received_bits):
            coeffs = self._generate_coeffs(idx)
            matrix.append(coeffs)
            vector.append(int(val))
            
        matrix = np.array(matrix, dtype=int)
        vector = np.array(vector, dtype=int)
        
        # 求解
        # 我们可以使用之前分析中的高斯消元
        # 或者在这里重用自包含的求解器？
        # 让我们在这里包含求解器以保持完整性
        
        return self._solve_gf2(matrix, vector)

    def _solve_gf2(self, matrix, vector):
        rows, cols = matrix.shape
        augmented = np.hstack((matrix, vector.reshape(-1, 1)))
        
        pivot_row = 0
        pivot_cols = []
        
        for col in range(cols):
            if pivot_row >= rows:
                break
                
            candidates = [r for r in range(pivot_row, rows) if augmented[r, col] == 1]
            if not candidates:
                continue
            
            curr = candidates[0]
            augmented[[pivot_row, curr]] = augmented[[curr, pivot_row]]
            
            for r in range(rows):
                if r != pivot_row and augmented[r, col] == 1:
                    augmented[r] ^= augmented[pivot_row]
            
            pivot_cols.append(col)
            pivot_row += 1
            
        if len(pivot_cols) == cols:
            x = np.zeros(cols, dtype=int)
            # 回代很简单，因为我们已经对角化了（大部分）
            # 实际上上面的完整高斯-约旦使得主元是列中唯一的非零元素
            for i, p_col in enumerate(pivot_cols):
                # 主元行 'i' 对应变量 'p_col'
                x[p_col] = augmented[i, -1]
            return "".join(map(str, x))
        else:
            return None
