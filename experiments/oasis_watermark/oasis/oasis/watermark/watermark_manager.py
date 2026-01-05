# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
"""
WatermarkManager - Agent Watermark Integration for OASIS

This module provides a plug-and-play watermark manager that integrates
AgentMark watermarking technology into OASIS social simulation platform.

Features:
- Non-invasive integration (no core code modification)
- Supports lightweight and full modes  
- Automatic logging and tracking
- Error correction code support (none/parity/hamming)
- Context-based key generation
"""

import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import AgentMark modules
# We expect AgentMark modules to be in the workspace or in PYTHONPATH
try:
    # Try to import from agentmark.core (Standard AgentMark Structure)
    from agentmark.core.watermark_sampler import (
        sample_behavior_differential,
        differential_based_decoder,
        generate_contextual_key
    )
    from agentmark.core.coding_utils import encode_payload, decode_message
    from agentmark.core.rlnc_codec import DeterministicRLNC
    
    AGENTMARK_AVAILABLE = True
except ImportError as e:
    AGENTMARK_AVAILABLE = False
    print(f"Warning: AgentMark modules not found. Watermark features will be disabled.")
    print(f"   Error: {e}")
    print(f"   Please ensure AgentMark code is in PYTHONPATH or in the workspace.")


class WatermarkManager:
    """
    Watermark Manager for OASIS Social Agents
    
    This class manages the watermarking process for social agents, including:
    - Bit stream management
    - Watermark embedding via probability distribution modification
    - Logging and statistics
    - Watermark extraction and verification
    
    Args:
        enabled (bool): Whether watermarking is enabled
        mode (str): Watermark mode ("lightweight" or "full")
        config (dict): Watermark configuration dictionary
        bit_stream (str, optional): Custom bit stream to embed
        log_dir (str): Directory for log files
        
    Example:
        >>> wm = WatermarkManager(enabled=True, mode="lightweight")
        >>> # In agent action: modify probabilities based on watermark bit
        >>> selected, targets, bits, ctx = wm.sample_behavior_watermark(
        ...     probabilities={"like": 0.3, "share": 0.7},
        ...     round_num=0,
        ...     context=""
        ... )
    """
    
    def __init__(
        self,
        enabled: bool = True,
        mode: str = "lightweight",
        config: Optional[Dict[str, Any]] = None,
        bit_stream: Optional[str] = None,
        log_dir: str = "./log",
        log_level: str = "INFO",
        agent_id: Optional[int] = None
    ):
        """
        Initialize WatermarkManager
        
        Args:
            agent_id: Unique identifier for the agent. If provided, this WatermarkManager
                     becomes an independent tracing unit for that specific agent.
        """
        self.enabled = enabled and AGENTMARK_AVAILABLE
        self.mode = mode
        self.agent_id = agent_id  # New: Independent agent ID
        if log_dir == "./log":
            log_dir = os.getenv(
                "OASIS_WATERMARK_LOG_DIR",
                os.getenv("OASIS_LOG_DIR", log_dir),
            )
        
        if not AGENTMARK_AVAILABLE and enabled:
            print("Warning: Watermark requested but AgentMark not available. Disabling watermark.")
            self.enabled = False
        
        # Default configuration
        self.config = config or {
            "payload_bit_length": 8,
            "ecc_method": "parity",
            "embedding_strategy": "cyclic"
        }
        
        # Logging setup
        self.log_dir = os.path.abspath(log_dir)  # Convert to absolute path
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Include microseconds to avoid logger/log-file collisions when multiple
        # WatermarkManager instances are created within the same second.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        # New: Include agent_id in log filename
        if agent_id is not None:
            self.log_file = os.path.join(self.log_dir, f"watermark-agent{agent_id}-{timestamp}.log")
            logger_name = f"watermark-agent{agent_id}-{timestamp}"
        else:
            self.log_file = os.path.join(self.log_dir, f"watermark-{timestamp}.log")
            logger_name = f"watermark-{timestamp}"
        
        # Setup logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        if not self.logger.handlers:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            formatter = logging.Formatter(
                "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # RLNC Setup
        self.rlnc_encoder = None
        if self.config.get("ecc_method") == "rlnc" and AGENTMARK_AVAILABLE:
            # RLNC requires original payload bits, raw
            # For simplicity, if bit_stream is provided, we treat it as payload
            # If agent_id provided, use it as payload
            payload_bits = ""
            if bit_stream:
                # Assuming bit_stream input is raw payload when rlnc is selected
                payload_bits = bit_stream
            elif agent_id is not None:
                payload_bits = format(agent_id, '08b')
            else:
                payload_bits = "11001101"
            
            # Initialize RLNC Encoder
            # Use 'rlnc_key' from config or default to 42
            rlnc_key = self.config.get("rlnc_key", 42)
            self.rlnc_encoder = DeterministicRLNC(payload_bits, stream_key=rlnc_key)
            self.bit_stream = payload_bits # Store raw payload for reference
            self.logger.info(f"RLNC Enabled. Payload: {payload_bits}, Key: {rlnc_key}")
        
        # Bit stream management - If agent_id is provided, encode agent_id as watermark
        elif bit_stream:
            self.bit_stream = self._prepare_bit_stream(bit_stream)
        elif agent_id is not None:
            # New: Use agent_id as watermark content (8-bit can represent 0-255 agents)
            agent_bits = format(agent_id, '08b')  # Convert to 8-bit binary
            self.bit_stream = self._prepare_bit_stream(agent_bits)
        else:
            # Default: encode a simple message
            self.bit_stream = self._prepare_bit_stream("11001101")
        
        self.bit_index = 0
        self.bits_embedded_history = []
        
        # Context management
        self.history_responses = []
        
        # Statistics
        self.stats = {
            "total_actions": 0,
            "watermarked_actions": 0,
            "bits_embedded": 0,
            "rounds_completed": 0
        }
        
        if self.enabled:
            self.logger.info("=" * 70)
            if agent_id is not None:
                self.logger.info(f"WatermarkManager Initialized (Agent {agent_id})")
            else:
                self.logger.info("WatermarkManager Initialized")
            self.logger.info("=" * 70)
            if agent_id is not None:
                self.logger.info(f"Agent ID: {agent_id}")
                self.logger.info(f"Agent ID (binary): {format(agent_id, '08b')}")
            self.logger.info(f"Mode: {self.mode}")
            self.logger.info(f"Bit stream: {self.bit_stream}")
            self.logger.info(f"Bit stream length: {len(self.bit_stream)}")
            self.logger.info(f"Config: {json.dumps(self.config, indent=2)}")
            self.logger.info(f"Log file: {self.log_file}")
            self.logger.info("=" * 70)
    
    def _prepare_bit_stream(self, payload: str) -> str:
        """
        Prepare bit stream with error correction encoding
        
        Args:
            payload (str): Original payload bits
            
        Returns:
            str: Encoded bit stream with ECC
        """
        if not AGENTMARK_AVAILABLE:
            return payload
        
        return encode_payload(payload, self.config)
    
    def sample_behavior_watermark(
        self,
        probabilities: Dict[str, float],
        round_num: int,
        context_for_key: str = ""
    ) -> Tuple[str, List[str], int, str]:
        """
        Sample behavior with watermark embedding (differential scheme)
        
        This is the core integration point where watermark bits are embedded
        into the agent's action selection process.
        
        Args:
            probabilities (dict): Behavior probabilities from LLM
            round_num (int): Current round number
            context_for_key (str): Context string for key generation
            
        Returns:
            tuple: (selected_behavior, target_list, bits_embedded, context_used)
            
        Example:
            >>> probs = {"like": 0.3, "comment": 0.2, "share": 0.5}
            >>> behavior, targets, bits, ctx = wm.sample_behavior_watermark(
            ...     probabilities=probs,
            ...     round_num=0,
            ...     context_for_key=""
            ... )
        """
        if not self.enabled or not AGENTMARK_AVAILABLE:
            # Fallback: simple random selection without watermark
            import random
            selected = random.choices(
                list(probabilities.keys()),
                weights=list(probabilities.values()),
                k=1
            )[0]
            return selected, [], 0, context_for_key
        
        try:
            effective_bit_stream = ""
            effective_bit_index = 0
            
            # RLNC Logic: Generate stream dynamically
            if self.config.get("ecc_method") == "rlnc" and self.rlnc_encoder:
                # Generate enough bits for this round (e.g. 50 is safe upper bound for one behavior sample)
                # We generate from current global bit_index
                effective_bit_stream = self.rlnc_encoder.get_stream(self.bit_index, 50)
                effective_bit_index = 0 # We are passing a fresh chunk starting from 0 (relative to chunk)
                
            # Cyclic Logic
            elif self.config.get("embedding_strategy") == "cyclic":
                # Prepare bit stream for cyclic embedding
                # For cyclic strategy, extend the bit stream to avoid boundary issues
                # Estimate max bits that could be embedded: log2(6) approx 2.58 bits/round
                # Extend to cover at least 2 full cycles beyond current position
                extended_length = self.bit_index + len(self.bit_stream) * 2
                num_repeats = (extended_length // len(self.bit_stream)) + 1
                effective_bit_stream = self.bit_stream * num_repeats
                effective_bit_index = self.bit_index
            else:
                # Sequential strategy: use original bit stream
                effective_bit_stream = self.bit_stream
                effective_bit_index = self.bit_index
            
            # Call AgentMark's differential watermark sampler
            selected_behavior, target_list, bits_embedded, context_used = \
                sample_behavior_differential(
                    probabilities=probabilities,
                    bit_stream=effective_bit_stream,
                    bit_index=effective_bit_index,
                    context_for_key=context_for_key,
                    round_num=round_num
                )
            
            # Record the bit_index BEFORE embedding (for logging)
            bit_index_before = self.bit_index
            
            # Update bit index
            old_bit_index = self.bit_index
            self.bit_index += bits_embedded
            
            # Handle cyclic embedding (Only for non-RLNC)
            if self.config.get("ecc_method") != "rlnc":
                if self.bit_index >= len(self.bit_stream):
                    if self.config.get("embedding_strategy") == "cyclic":
                        self.bit_index = self.bit_index % len(self.bit_stream)
                        self.logger.info(f"Bit stream cycled. Reset from {old_bit_index} to {self.bit_index}")
                        self.stats["cycles"] = self.stats.get("cycles", 0) + 1
            
            # Log the watermarked action (AFTER updating bit_index so it shows correct state)
            self.log_watermark_action(
                round_num=round_num,
                probabilities=probabilities,
                selected_behavior=selected_behavior,
                target_list=target_list,
                bits_embedded=bits_embedded,
                context_for_key=context_used,
                bit_index_before=bit_index_before  # Record index before embedding
            )
            
            # Update statistics
            self.stats["watermarked_actions"] += 1
            self.stats["bits_embedded"] += bits_embedded
            
            return selected_behavior, target_list, bits_embedded, context_used
            
        except Exception as e:
            self.logger.error(f"Error in watermark sampling: {e}", exc_info=True)
            # Fallback
            import random
            selected = random.choices(
                list(probabilities.keys()),
                weights=list(probabilities.values()),
                k=1
            )[0]
            return selected, [], 0, context_for_key
    
    def log_watermark_action(
        self,
        round_num: int,
        probabilities: Dict[str, float],
        selected_behavior: str,
        target_list: List[str],
        bits_embedded: int,
        context_for_key: str,
        bit_index_before: int = None
    ):
        """Log watermarked action details"""
        log_entry = {
            "round_num": round_num,
            "probabilities_watermark": probabilities,
            "selected_behavior_watermark": selected_behavior,
            "target_list": target_list,
            "bits_embedded": bits_embedded,
            "context_for_key": context_for_key,
            "bit_index": bit_index_before if bit_index_before is not None else self.bit_index  # Record index before embedding
        }
        
        self.logger.info(f"Round {round_num}: {json.dumps(log_entry, indent=2)}")
        self.bits_embedded_history.append(bits_embedded)
    
    def get_next_bit(self) -> str:
        """
        Get the next bit from bit stream (for manual control)
        
        Returns:
            str: Next bit ('0' or '1')
        """
        if not self.enabled:
            return '0'
        
        if self.bit_index >= len(self.bit_stream):
            if self.config.get("embedding_strategy") == "cyclic":
                self.bit_index = 0
            else:
                return '0'
        
        bit = self.bit_stream[self.bit_index]
        self.bit_index += 1
        return bit
    
    def log_action(
        self,
        agent_id: int,
        action_name: str,
        action_args: Dict[str, Any],
        bit: str
    ):
        """
        Log action with watermark bit (for manual watermarking)
        
        Args:
            agent_id (int): Agent ID
            action_name (str): Action name
            action_args (dict): Action arguments
            bit (str): Watermark bit embedded
        """
        if not self.enabled:
            return
        
        log_entry = {
            "agent_id": agent_id,
            "action_name": action_name,
            "action_args": action_args,
            "watermark_bit": bit,
            "bit_index": self.bit_index - 1
        }
        
        self.logger.info(f"Action: {json.dumps(log_entry)}")
        self.stats["total_actions"] += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get watermark statistics
        
        Returns:
            dict: Statistics dictionary
        """
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "bit_stream_length": len(self.bit_stream),
            "current_bit_index": self.bit_index,
            "bits_remaining": len(self.bit_stream) - self.bit_index,
            "total_actions": self.stats["total_actions"],
            "watermarked_actions": self.stats["watermarked_actions"],
            "bits_embedded": self.stats["bits_embedded"],
            "rounds_completed": self.stats["rounds_completed"],
            "cycles": self.stats.get("cycles", 0),
            "log_file": self.log_file
        }
    
    def extract_watermark_from_log(
        self,
        log_path: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extract watermark from log file
        
        Args:
            log_path (str, optional): Path to log file. If None, use current log.
            
        Returns:
            tuple: (extracted_bits, statistics)
        """
        if not self.enabled or not AGENTMARK_AVAILABLE:
            return "", {"error": "Watermark not enabled or AgentMark not available"}
        
        if log_path is None:
            log_path = self.log_file
        
        try:
            # Parse log file
            extracted_bits = []
            round_data_list = []
            
            with open(log_path, 'r') as f:
                content = f.read()
            
            # Parse multi-line JSON from log
            # Look for "Round X: {" patterns and extract the JSON
            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i]
                if 'Round ' in line and ': {' in line:
                    # Found start of JSON block
                    json_start = line.find('{')
                    if json_start != -1:
                        # Collect all lines until we find the closing }
                        json_lines = [line[json_start:]]
                        i += 1
                        brace_count = 1
                        while i < len(lines) and brace_count > 0:
                            current_line = lines[i]
                            json_lines.append(current_line)
                            brace_count += current_line.count('{') - current_line.count('}')
                            i += 1
                            if brace_count == 0:
                                break
                        
                        # Parse the collected JSON
                        json_str = '\n'.join(json_lines)
                        try:
                            data = json.loads(json_str)
                            round_data_list.append(data)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse JSON: {e}")
                            pass
                else:
                    i += 1
            
            # Extract bits from each round
            for round_data in round_data_list:
                # Only extract from rounds that actually embedded bits
                bits_embedded = round_data.get('bits_embedded', 0)
                if bits_embedded > 0:
                    bits = differential_based_decoder(
                        probabilities=round_data['probabilities_watermark'],
                        selected_behavior=round_data['selected_behavior_watermark'],
                        context_for_key=round_data['context_for_key'],
                        round_num=round_data['round_num']
                    )
                    extracted_bits.append(bits)
            
            extracted_bit_stream = "".join(extracted_bits)
            
            # Calculate expected length based on config
            payload_length = self.config.get('payload_bit_length', 8)
            ecc_method = self.config.get('ecc_method', 'none')
            
            # RLNC Decoding Branch
            if ecc_method == "rlnc":
                # Re-iterate through round_data_list to get indices
                received_indices = []
                received_bits = []
                
                # We need to reconstruct the EXACT indices.
                # Since extracted_bit_stream is just a concat, we need the structure.
                # Luckily we have extracted_bits list which aligns with rounds that had output.
                # But we actually need to match bits to their global indices.
                # The log stores 'bit_index' which is the start index for that round.
                
                for round_data in round_data_list:
                    bits_embedded = round_data.get('bits_embedded', 0)
                    if bits_embedded > 0:
                        start_idx = round_data.get('bit_index', 0)
                        # Extract bits for this round
                        bits = differential_based_decoder(
                            probabilities=round_data['probabilities_watermark'],
                            selected_behavior=round_data['selected_behavior_watermark'],
                            context_for_key=round_data['context_for_key'],
                            round_num=round_data['round_num']
                        )
                        
                        # Add to list
                        for i, bit_val in enumerate(bits):
                            received_indices.append(start_idx + i)
                            received_bits.append(bit_val)
                
                # Perform RLNC Decoding
                rlnc_key = self.config.get("rlnc_key", 42)
                # Recreate decoder (stateless mostly)
                # We assume payload length 8 if not specified (default)
                dummy_payload = "0" * payload_length
                decoder_instance = DeterministicRLNC(dummy_payload, stream_key=rlnc_key)
                
                decoded_payload = decoder_instance.decode(received_indices, received_bits)
                
                is_valid = decoded_payload is not None
                
                stats = {
                    "actions_processed": len(round_data_list),
                    "received_bits_count": len(received_bits),
                    "decoded_payload": decoded_payload if is_valid else "FAIL",
                    "valid": is_valid,
                    "ecc_method": "rlnc",
                    "error": None if is_valid else "RLNC decoding failed (not enough linear independent packets)"
                }
                
                self.logger.info("=" * 70)
                self.logger.info("RLNC Watermark Extraction Complete")
                self.logger.info("=" * 70)
                self.logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
                
                return decoded_payload if is_valid else "", stats
            
            if ecc_method == 'parity':
                expected_length = payload_length + 1  # 8 + 1 = 9
            elif ecc_method == 'hamming':
                expected_length = 21  # Hamming(21,16)
            else:
                expected_length = payload_length
            
            # Decode sequentially by block size (cyclic validation)
            num_messages = len(extracted_bit_stream) // expected_length
            remainder = len(extracted_bit_stream) % expected_length
            
            if num_messages == 0 and remainder > 0:
                # Less than a full block, perform partial decode
                self.logger.warning(f"Extracted only {len(extracted_bit_stream)} bits, expected {expected_length}, performing partial decode")
                result = decode_message(extracted_bit_stream, self.config)
                decoded_payloads = [result.get('decoded_payload', '')]
                validation_results = [result]
                total_corrections = 0
                failed_validations = 1 if not result.get('valid') else 0
                partial_is_valid = False  # Initialize to False (less than a full block)
            else:
                # Have complete blocks, decode sequentially
                self.logger.info(f"Decoding {num_messages} complete messages ({expected_length} bits each)")
                if remainder > 0:
                    self.logger.warning(f"Remaining {remainder} bits will be decoded as partial message")
                
                decoded_payloads = []
                validation_results = []
                total_corrections = 0
                failed_validations = 0
                
                # Decode all complete blocks
                for i in range(num_messages):
                    start_idx = i * expected_length
                    end_idx = start_idx + expected_length
                    encoded_message = extracted_bit_stream[start_idx:end_idx]
                    
                    result = decode_message(encoded_message, self.config)
                    decoded_payloads.append(result.get('decoded_payload', ''))
                    validation_results.append(result)
                    
                    if result.get('corrected', False):
                        total_corrections += 1
                    if not result.get('valid'):
                        failed_validations += 1
                
                # If there is a remainder, decode the last incomplete block
                partial_is_valid = None  # Default to None, only set when remainder exists
                if remainder > 0:
                    partial_is_valid = False  # Initialize to False
                    partial_message = extracted_bit_stream[num_messages * expected_length:]
                    result = decode_message(partial_message, self.config)
                    decoded_payloads.append(result.get('decoded_payload', ''))
                    validation_results.append(result)
                    
                    # Improved remainder validation: cyclic match with original bit_stream
                    # Calculate where the remainder should correspond in the original bit stream
                    total_embedded_bits = num_messages * payload_length + len(result.get('decoded_payload', ''))
                    expected_partial_position = (total_embedded_bits - len(result.get('decoded_payload', ''))) % len(self.bit_stream)
                    
                    # Extract corresponding part from original bit stream
                    expected_partial = ""
                    for i in range(len(result.get('decoded_payload', ''))):
                        expected_partial += self.bit_stream[(expected_partial_position + i) % len(self.bit_stream)]
                    
                    # Compare extracted part with expected part
                    if result.get('decoded_payload', '') == expected_partial:
                        partial_is_valid = True
                        self.logger.info(f"[OK] Partial bits validated: {result.get('decoded_payload', '')} matches expected {expected_partial}")
                    else:
                        # Warning: Mismatch in partial block is logged but does not count as validation failure
                        # As long as at least one complete block succeeds
                        self.logger.warning(f"[WARN] Partial bits mismatch: got {result.get('decoded_payload', '')}, expected {expected_partial} (does NOT affect validation result)")
            
            decoded_bit_stream = "".join(decoded_payloads)
            
            # Improved accuracy calculation: Cyclic comparison of extracted raw bits with bit_stream
            # Note: extracted_bit_stream is the raw extraction (with ECC), compare with bit_stream (also with ECC)
            original_bits = self.bit_stream
            accuracy = 0.0
            if extracted_bit_stream:
                # Cyclic comparison, handling cases exceeding original length
                matches = 0
                for i, bit in enumerate(extracted_bit_stream):
                    expected_bit = original_bits[i % len(original_bits)]
                    if bit == expected_bit:
                        matches += 1
                accuracy = matches / len(extracted_bit_stream) * 100
            
            # Improved validation logic:
            # - If complete blocks > 0, pass if at least one complete block validates
            # - If no complete blocks, fail (partial is not enough)
            # - Partial failure doesn't fail the whole result if complete blocks pass
            
            # Count failed complete blocks (only count validation failures of complete blocks)
            complete_blocks_failed = sum(1 for i, result in enumerate(validation_results[:num_messages]) if not result.get('valid'))
            
            # Validation condition:
            # 1. Have at least one complete block
            # 2. Not all complete blocks failed (i.e., at least one succeeded)
            # For strictness we could say ALL complete blocks must pass, but given channel noise, > threshold is better.
            # Here keeping original logic: fail if complete_blocks_failed == num_messages? 
            # Original code: `complete_blocks_failed < num_messages` -> at least one passed.
            is_valid = num_messages > 0 and complete_blocks_failed < num_messages
            
            # Error message only reports failures of complete blocks
            error_msg = None
            if num_messages == 0:
                error_msg = f"Extracted {len(extracted_bit_stream)} bits, not enough for one complete block ({expected_length} bits)"
            elif complete_blocks_failed > 0:
                error_msg = f"{complete_blocks_failed} out of {num_messages} complete blocks failed validation"
            
            stats = {
                "actions_processed": len(round_data_list),
                "successful_extractions": len([b for b in extracted_bits if b]),
                "extracted_bit_stream": extracted_bit_stream,
                "decoded_payload": decoded_bit_stream,  # All decoded payloads concatenated
                "num_messages": num_messages + (1 if remainder > 0 else 0),
                "complete_messages": num_messages,
                "partial_bits": remainder,
                "partial_is_valid": partial_is_valid if remainder > 0 else None,
                "total_corrections": total_corrections,
                "failed_validations": complete_blocks_failed,  # Only count complete block failures
                "valid": is_valid,  # New logic: At least one complete block validated successfully
                "corrected": total_corrections > 0,
                "accuracy": accuracy,  # New accuracy field
                "ecc_method": self.config.get('ecc_method', 'none'),
                "error": error_msg
            }
            
            self.logger.info("=" * 70)
            self.logger.info("Watermark Extraction Complete")
            self.logger.info("=" * 70)
            self.logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
            
            return extracted_bit_stream, stats
            
        except Exception as e:
            self.logger.error(f"Error extracting watermark: {e}", exc_info=True)
            return "", {"error": str(e)}
    
    def update_context(self, response: str):
        """
        Update history context with new response
        
        Args:
            response (str): New response to add to history
        """
        self.history_responses.append(response)
        
        # Keep only recent history (sliding window)
        window_size = 3
        if len(self.history_responses) > window_size:
            self.history_responses = self.history_responses[-window_size:]
    
    def get_context_for_key(self) -> str:
        """
        Get context string for key generation
        
        Returns:
            str: Context string (concatenated recent responses)
        """
        if not self.history_responses:
            return ""
        
        return "||".join(self.history_responses)
    
    def reset(self):
        """Reset watermark manager state"""
        self.bit_index = 0
        self.history_responses = []
        self.bits_embedded_history = []
        self.stats = {
            "total_actions": 0,
            "watermarked_actions": 0,
            "bits_embedded": 0,
            "rounds_completed": 0
        }
        
        if self.enabled:
            self.logger.info("WatermarkManager reset")
    
    def __repr__(self) -> str:
        return f"WatermarkManager(enabled={self.enabled}, mode={self.mode}, bits={len(self.bit_stream)})"
