
import random
import numpy as np

class DeterministicRLNC:
    """
    Deterministic Random Linear Network Coding over GF(2).
    Generates an infinite stream of coded bits from a fixed payload key.
    
    The i-th coded bit is a linear combination of the payload bits,
    where the coefficients are generated deterministically based on the
    stream ID (key) and the bit index (i).
    """
    def __init__(self, payload_bits_str, stream_key=42):
        """
        Args:
            payload_bits_str (str): The payload to encode, e.g., "10110011".
            stream_key (int/str): A seed/key to randomize the coefficients.
        """
        self.payload = [int(b) for b in payload_bits_str]
        self.n = len(self.payload)
        self.stream_key = stream_key
        # Cache for generated bits if needed, but simple generation is fast enough.

    def get_bit(self, index):
        """
        Get the i-th coded bit.
        c_i = sum(coeff_j * p_j) mod 2, where coeff_j comes from PRG(stream_key, index).
        """
        coeffs = self._generate_coeffs(index)
        
        coded_val = 0
        for b_val, c_val in zip(self.payload, coeffs):
            coded_val ^= (b_val & c_val)
        return str(coded_val)

    def get_stream(self, start_index, length):
        """
        Get a sequence of coded bits.
        """
        return "".join([self.get_bit(i) for i in range(start_index, start_index + length)])

    def _generate_coeffs(self, index):
        """
        Generate n coefficients deterministically for a given index.
        """
        # Use a hash of (stream_key, index) to seed, or simple Random with fixed seed.
        # Python's random is Mersenne Twister, not safe for crypto but fine for coding distribution.
        # To ensure absolute seeking capability without rewinding, we need a hash-based PRG or re-seeding.
        # Re-seeding is fast enough for small n.
        
        # Combine key and index into a unique seed
        # efficient string seed
        seed_val = f"{self.stream_key}_{index}"
        rd = random.Random(seed_val)
        
        # In GF(2), coefficients are 0 or 1.
        # We need a non-zero row ideally, but random is fine for RLNC (prob of 0 row is 1/2^n).
        # We can force non-zero if we want, but standard RLNC doesn't strictly require it per packet.
        return [rd.randint(0, 1) for _ in range(self.n)]

    def decode(self, received_indices, received_bits):
        """
        Attempt to decode the payload from a set of received coded bits.
        
        Args:
            received_indices (list[int]): Indices of the received coded bits.
            received_bits (list[int/str]): Values of the received coded bits.
            
        Returns:
            str: Decoded payload bits if successful, None otherwise.
        """
        m = len(received_indices)
        if m < self.n:
            return None
        
        # Build the system Matrix * Payload = Received
        matrix = []
        vector = []
        
        for idx, val in zip(received_indices, received_bits):
            coeffs = self._generate_coeffs(idx)
            matrix.append(coeffs)
            vector.append(int(val))
            
        matrix = np.array(matrix, dtype=int)
        vector = np.array(vector, dtype=int)
        
        # Solve
        # We can use the Gaussian elimination from previous analysis
        # Or reuse self-contained solver here?
        # Let's include the solver here for completeness
        
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
            # Back sub is trivial because we diagonalized (mostly)
            # Actually full Gauss-Jordan above makes pivots the only non-zeros in columns
            for i, p_col in enumerate(pivot_cols):
                # The pivot row 'i' corresponds to variable 'p_col'
                x[p_col] = augmented[i, -1]
            return "".join(map(str, x))
        else:
            return None
