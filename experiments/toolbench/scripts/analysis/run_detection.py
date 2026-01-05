"""
Watermark extraction/decoding script.
Responsibility: extract the embedded secret bitstream from experiment logs and verify watermark integrity.
Supports ECC validation and error correction.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

# Import log parser and decoder utilities
from agentmark.core.log_parser import parse_log_files, validate_round_data
from agentmark.core.watermark_sampler import differential_based_decoder
from agentmark.core.coding_utils import decode_message

# Absolute path to this script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / 'data'
OUTPUT_DIR = SCRIPT_DIR / 'output'
LOG_DIR = OUTPUT_DIR / 'log'


def extract_operation_description(behavior_response):
    """
    Extract the action description from a behavior response.

    Args:
        behavior_response (str): Full behavior response text.

    Returns:
        str: Extracted action description.
    """
    import re
    
    # Try to match a generic "> <label>: ..." pattern.
    match = re.search(r'>\s*[^:\n]{1,40}[:]\s*(.+?)(?:\n|$)', behavior_response)
    if match:
        operation_desc = match.group(1).strip()
        return operation_desc
    
    # Fallback: return a truncated snippet.
    return behavior_response[:200] if len(behavior_response) > 200 else behavior_response


def main():
    print(" Starting watermark extraction (offline mode)...")
    print("=" * 70)
    
    # --- 1. Load config ---
    print("\n Step 1: Load config file")
    config_path = SCRIPT_DIR / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f" Config loaded: {config_path}")
    
    # Load watermark config (for ECC decoding)
    watermark_config = config.get('watermark_config', {
        'payload_bit_length': 8,
        'ecc_method': 'none',
        'embedding_strategy': 'cyclic'
    })
    embedding_strategy = watermark_config.get('embedding_strategy', 'once')
    print(f"  ECC config: {watermark_config['ecc_method']} (payload length: {watermark_config['payload_bit_length']})")
    print(f"  Embedding strategy: {embedding_strategy}")
    
    # Compute expected encoded length
    if watermark_config['ecc_method'] == 'parity':
        expected_encoded_length = watermark_config['payload_bit_length'] + 1
    elif watermark_config['ecc_method'] == 'hamming':
        expected_encoded_length = 21  # Hamming(21,16) requires a 16-bit payload
    else:
        expected_encoded_length = watermark_config['payload_bit_length']
    
    print(f"  Expected encoded length: {expected_encoded_length} bits/message")

    # --- 2. Parse log files ---
    print("\n Step 2: Parse log files")
    # Use the new directory structure
    log_path = LOG_DIR / 'watermark_log.txt'
    verbose_log_path = LOG_DIR / 'watermark_verbose.log'
    
    print(f"  Standard log: {log_path}")
    print(f"  Verbose log: {verbose_log_path}")
    
    all_log_data = parse_log_files(str(log_path), str(verbose_log_path))
    
    if not all_log_data:
        print(" Error: no log data parsed")
        return
    
    # --- 3. Load original bit stream and encode ---
    print("\n Step 3: Load original bit stream")
    bit_stream_path = DATA_DIR / 'bit_stream.txt'
    try:
        with open(bit_stream_path, 'r') as f:
            original_bit_stream = f.read().strip()
        print(f" Original bit stream length: {len(original_bit_stream)} bits")
        print(f"  First 50 bits: {original_bit_stream[:50]}...")
        
        # If using cyclic embedding, encode the original payload.
        if embedding_strategy == 'cyclic':
            from agentmark.core.coding_utils import encode_payload
            payload_bit_length = watermark_config.get('payload_bit_length', 8)
            payload_bits = original_bit_stream[:payload_bit_length]
            
            if len(payload_bits) < payload_bit_length:
                print(f" Warning: bit stream shorter than {payload_bit_length} bits")
                payload_bits = payload_bits.ljust(payload_bit_length, '0')
            
            original_encoded_message = encode_payload(payload_bits, watermark_config)
            print(f"  Encoded message: '{original_encoded_message}' ({len(original_encoded_message)} bits)")
            print("  Cyclic embedding: expected decode is repetitions of this message")
        else:
            original_encoded_message = original_bit_stream
            
    except FileNotFoundError:
        print(f" Warning: {bit_stream_path} not found; cannot validate decoding result")
        original_bit_stream = ""
        original_encoded_message = ""

    # --- 4. Per-round decoding (no LLM calls) ---
    print(f"\n Step 4: Start decoding ({len(all_log_data)} rounds)")
    print("-" * 70)
    
    extracted_bits_list = []
    watermark_history_responses = []  # For dynamic key generation
    skipped_rounds = 0

    for i, round_data in enumerate(tqdm(all_log_data, desc="Decoding progress")):
        # Validate data integrity
        is_valid, missing_keys = validate_round_data(round_data, i + 1)
        if not is_valid:
            print(f"   Round {i+1} incomplete, missing: {missing_keys}. Skipping.")
            skipped_rounds += 1
            continue
        
        # Fetch required info from parsed data
        probabilities = round_data['probabilities_watermark']
        selected_behavior = round_data['selected_behavior_watermark']
        
        # *** New method: read context_for_key from logs ***
        context_for_key = round_data.get('context_for_key', None)
        
        # If context_for_key is missing (legacy logs), rebuild from history.
        if context_for_key is None:
            print(f"   Round {i+1} missing context_for_key; rebuilding from history")
            behavior_response = round_data['behavior_response_watermark']
            
            # Extract action descriptions from history
            window_size = 3
            recent_responses = watermark_history_responses[-window_size:] if len(watermark_history_responses) > 0 else []
            operation_descriptions = [extract_operation_description(resp) for resp in recent_responses]
            context_for_key = "||".join(operation_descriptions) if operation_descriptions else ""
            
            print(f"     Rebuilt context length: {len(context_for_key)} chars")
        
        # --- Run decoding (always uses context_for_key) ---
        try:
            bits = differential_based_decoder(
                probabilities=probabilities,
                selected_behavior=selected_behavior,
                context_for_key=context_for_key,  # Context read/rebuilt from logs
                round_num=i  # round_num must match encoding
            )
            
            extracted_bits_list.append(bits)
            
            # Show decoding details (first 3 rounds only)
            if i < 3:
                print(f"\n  Round {i+1}:")
                print(f"    Selected action: {selected_behavior}")
                print(f"    Context length: {len(context_for_key) if context_for_key else 'N/A'}")
                print(f"    Extracted bits: {bits if bits else '(no bits)'}")
        except Exception as e:
            print(f"\n   Round {i+1} decoding failed: {e}")
            skipped_rounds += 1
        
        # --- Legacy rebuild mode: update history for next round ---
        if round_data.get('context_for_key', None) is None:
            behavior_response = round_data['behavior_response_watermark']
            watermark_history_responses.append(behavior_response)

    # --- 5. ECC decoding and validation ---
    print("\n" + "=" * 70)
    print(" Step 5: ECC decoding and validation")
    print("-" * 70)
    
    extracted_bit_stream = "".join(extracted_bits_list)
    print(f"\nExtracted encoded bitstream length: {len(extracted_bit_stream)} bits")
    print(f"Skipped rounds: {skipped_rounds}/{len(all_log_data)}")
    
    # Decode in blocks based on encoded length
    decoded_payloads = []
    validation_results = []
    total_corrections = 0
    failed_validations = 0
    
    if watermark_config['ecc_method'] != 'none':
        print(f"\n Start ECC decoding (method: {watermark_config['ecc_method']})")
        print(f"  Encoded block length: {expected_encoded_length} bits")
        
        num_messages = len(extracted_bit_stream) // expected_encoded_length
        print(f"  Expected message count: {num_messages}")
        
        for i in range(num_messages):
            start_idx = i * expected_encoded_length
            end_idx = start_idx + expected_encoded_length
            encoded_message = extracted_bit_stream[start_idx:end_idx]
            
            # Call ECC decoding
            result = decode_message(encoded_message, watermark_config)
            
            decoded_payloads.append(result['decoded_payload'])
            validation_results.append(result)
            
            if result.get('corrected', False):
                total_corrections += 1
            
            if not result['valid']:
                failed_validations += 1
                if failed_validations <= 3:  # Only show the first 3 failures
                    print(f"   Message #{i+1} validation failed: {result.get('error', 'Unknown')}")
        
        decoded_bit_stream = "".join(decoded_payloads)
        
        # Summarize ECC results
        print("\n ECC summary:")
        print(f"  Decoded successfully: {num_messages - failed_validations}/{num_messages}")
        print(f"  Validation failures: {failed_validations}")
        print(f"  Corrections: {total_corrections}")
        
        if watermark_config['ecc_method'] == 'parity':
            print(f"  Parity pass rate: {((num_messages - failed_validations) / num_messages * 100):.1f}%")
        elif watermark_config['ecc_method'] == 'hamming':
            print(f"  Hamming correction rate: {(total_corrections / num_messages * 100):.1f}%")
    else:
        decoded_bit_stream = extracted_bit_stream
        print("\n ECC disabled; using raw bitstream")
    
    # --- 6. Validate decoded results ---
    print("\n" + "=" * 70)
    print(" Step 6: Validate decoded results")
    print("-" * 70)
    
    # Validate cyclic embedding strategy
    if embedding_strategy == 'cyclic' and len(original_encoded_message) > 0:
        # Compute cycle counts based on decoded payload length
        cycles = len(decoded_bit_stream) // len(original_bit_stream)
        remainder = len(decoded_bit_stream) % len(original_bit_stream)
        
        print("\n Detected cyclic embedding strategy:")
        print(f"  Core payload: '{original_bit_stream}' ({len(original_bit_stream)} bits)")
        print(f"  Encoded message: '{original_encoded_message}' ({len(original_encoded_message)} bits)")
        print(f"  Total decoded length: {len(decoded_bit_stream)} bits")
        print(f"  Full cycles: {cycles}")
        if remainder > 0:
            print(f"  Partial cycle length: {remainder} bits")
        
        # Extend original payload to match decoded length
        if cycles > 0 or remainder > 0:
            expected_bit_stream = (original_bit_stream * (cycles + 1))[:len(decoded_bit_stream)]
        else:
            expected_bit_stream = original_bit_stream
    else:
        expected_bit_stream = original_bit_stream
    
    print(f"\nOriginal bitstream (first 50): {original_bit_stream[:50]}...")
    print(f"Decoded bitstream (first 50): {decoded_bit_stream[:50]}...")
    if embedding_strategy == 'cyclic':
        print(f"Expected bitstream (first 50): {expected_bit_stream[:50]}...")
    print(f"\nDecoded payload length: {len(decoded_bit_stream)} bits")
    print(f"Original bitstream length: {len(original_bit_stream)} bits")
    if embedding_strategy == 'cyclic':
        print(f"Expected bitstream length: {len(expected_bit_stream)} bits (after cyclic expansion)")
    
    # Compute BER based on decoded payload
    if original_bit_stream:
        errors = 0
        min_len = min(len(expected_bit_stream), len(decoded_bit_stream))
        
        for j in range(min_len):
            if expected_bit_stream[j] != decoded_bit_stream[j]:
                errors += 1
        
        ber = (errors / min_len) * 100 if min_len > 0 else 0
        
        print("\n" + "=" * 70)
        print(" Validation result")
        print("-" * 70)
        print(f"Compared length: {min_len} bits")
        print(f"Error bits: {errors}")
        print(f"Bit error rate (BER): {ber:.2f}%")
        
        if embedding_strategy == 'cyclic' and len(decoded_bit_stream) >= len(original_bit_stream):
            print("\n Cycle validation details:")
            print(f"  Core payload: '{original_bit_stream}'")
            print(f"  Decoded result: '{decoded_bit_stream}'")
            
            # Validate each cycle segment based on original payload length
            original_len = len(original_bit_stream)
            num_cycles = (len(decoded_bit_stream) + original_len - 1) // original_len
            
            for cycle_idx in range(num_cycles):
                start_idx = cycle_idx * original_len
                end_idx = min(start_idx + original_len, len(decoded_bit_stream))
                segment = decoded_bit_stream[start_idx:end_idx]
                expected_segment = original_bit_stream[:len(segment)]
                
                segment_errors = sum(1 for i in range(len(segment)) if segment[i] != expected_segment[i])
                segment_ber = (segment_errors / len(segment) * 100) if len(segment) > 0 else 0
                
                status = "" if segment_errors == 0 else ""
                print(f"  Cycle {cycle_idx + 1}: {status} '{segment}' (BER: {segment_ber:.1f}%)")
        
        # If ECC is enabled, compute post-correction BER stats
        if watermark_config['ecc_method'] != 'none' and len(validation_results) > 0:
            # Summarize corrections
            corrected_errors = sum(1 for r in validation_results if r.get('corrected', False))
            print("\n ECC correction summary:")
            print(f"  Corrected errors: {corrected_errors}")
            print(f"  Post-ECC BER: {ber:.2f}%")
            
            if corrected_errors > 0:
                print(f"   ECC successfully corrected {corrected_errors} errors")
        
        # Special handling for cyclic embedding results
        if embedding_strategy == 'cyclic':
            # Only check the first cycle (core payload)
            first_cycle_len = min(len(original_bit_stream), len(decoded_bit_stream))
            first_cycle_errors = sum(1 for i in range(first_cycle_len) 
                                    if original_bit_stream[i] != decoded_bit_stream[i])
            first_cycle_ber = (first_cycle_errors / first_cycle_len * 100) if first_cycle_len > 0 else 0
            
            # Count errors across full cycles
            num_full_cycles = len(decoded_bit_stream) // len(original_bit_stream)
            cycle_errors = []
            for cycle_idx in range(num_full_cycles):
                start = cycle_idx * len(original_bit_stream)
                end = start + len(original_bit_stream)
                cycle_segment = decoded_bit_stream[start:end]
                errors = sum(1 for i in range(len(cycle_segment)) if cycle_segment[i] != original_bit_stream[i])
                cycle_errors.append(errors)
            
            if first_cycle_ber == 0.0:
                print("\n Congratulations! Core message extracted perfectly; decoding succeeded!")
                print(" Watermark embed/extract verified (first cycle)")
                
                # Check whether subsequent cycles are also correct
                if all(e == 0 for e in cycle_errors):
                    print(f"\n Perfect! All {num_full_cycles} cycles are correct!")
                    print(" Cyclic embedding works as expected")
                elif ber > 0:
                    failed_cycles = [i+1 for i, e in enumerate(cycle_errors) if e > 0]
                    print(f"\n Errors detected in later cycles (overall BER={ber:.2f}%)")
                    print(f"  Failed cycles: {failed_cycles}")
                    print("  Possible causes:")
                    print("  - Transmission noise/bit flips")
                    print("  - PRG randomness unstable in some rounds")
                    print("  - Context key drift in some rounds")
                    print("\n Suggestion: check the logs for failed cycles")
            elif first_cycle_ber < 5.0:
                print(f"\n Core message has minor errors (first-cycle BER={first_cycle_ber:.2f}%)")
                print("Possible causes:")
                print("  - PRG randomness differences")
                print("  - History context mismatch")
                print("  - Probability distribution rounding")
            else:
                print(f"\n Core message decoding failed (first-cycle BER={first_cycle_ber:.2f}%)")
                print("Please check:")
                print("  - Log files are complete")
                print("  - Encoding and decoding are fully synchronized")
                print("  - round_num parameter matches")
        else:
            # Non-cyclic validation logic
            if ber == 0.0 and min_len > 0:
                print("\n Congratulations! Bitstream extracted perfectly; decoding succeeded!")
                print(" Watermark embed/extract verified")
            elif ber < 5.0:
                print(f"\n Minor decoding errors (BER={ber:.2f}%)")
                print("Possible causes:")
                print("  - PRG randomness differences")
                print("  - History context mismatch")
                print("  - Probability distribution rounding")
                if watermark_config['ecc_method'] != 'none':
                    print("  - Errors exceed ECC capability")
            else:
                print(f"\n High decoding error rate (BER={ber:.2f}%)")
                print("Please check:")
                print("  - Log files are complete")
                print("  - Encoding and decoding are fully synchronized")
                print("  - round_num parameter matches")
                print("  - ECC config matches")
    else:
        print("\n Original bitstream not loaded; cannot validate decoding accuracy")
    
    # Save extracted bitstreams (pre- and post-ECC)
    output_file_encoded = OUTPUT_DIR / 'extracted_with_ecc.txt'
    output_file_decoded = OUTPUT_DIR / 'extracted_data_only.txt'
    
    with open(output_file_encoded, 'w') as f:
        f.write(extracted_bit_stream)
    print(f"\n Extracted encoded bitstream saved to: {output_file_encoded}")
    print("   (includes ECC bits)")
    
    with open(output_file_decoded, 'w') as f:
        f.write(decoded_bit_stream)
    print(f" Decoded data saved to: {output_file_decoded}")
    print("   (payload with ECC removed)")
    
    # Save validation details
    if validation_results:
        validation_summary = {
            'total_messages': len(validation_results),
            'valid_messages': sum(1 for r in validation_results if r['valid']),
            'corrected_messages': sum(1 for r in validation_results if r.get('corrected', False)),
            'failed_messages': sum(1 for r in validation_results if not r['valid']),
            'ecc_method': watermark_config['ecc_method'],
            'details': validation_results
        }
        
        validation_file = OUTPUT_DIR / 'validation_report.json'
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_summary, f, indent=2, ensure_ascii=False)
        print(f" Validation report saved to: {validation_file}")
    
    print("\n" + "=" * 70)
    print(" Decoding complete")


if __name__ == "__main__":
    main()
