"""
ALFWorld main experiment runner.
Responsibilities: load config and bit stream, initialize environment and agent,
run baseline and watermarked groups, compute metrics, and generate reports.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 7.4
"""

import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime

# Ensure modules are discoverable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from openai import OpenAI
from tqdm import tqdm

# sys.path[0] is usually the scripts directory; use PROJECT_ROOT for absolute imports

from agentmark.environments.alfworld.adapter import ALFWorldAdapter
from agentmark.environments.alfworld.agent import ALFWorldAgent
from agentmark.environments.alfworld.logger import (
    initialize_alfworld_log,
    log_experiment_start,
    log_task_result,
    calculate_experiment_metrics,
    log_experiment_summary,
    print_experiment_summary
)
# Add current script directory to Python path for sibling imports
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from experiment_controller import (
    run_baseline_experiment,
    run_watermarked_experiment,
    calculate_metrics,
    generate_report
)
from decode_report import decode_task_bits

# Default CJK font path for visualization
DEFAULT_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not os.environ.get("AGENTMARK_CJK_FONT_PATH") and os.path.exists(DEFAULT_CJK_FONT_PATH):
    os.environ["AGENTMARK_CJK_FONT_PATH"] = DEFAULT_CJK_FONT_PATH

# Default CJK font path for visualization
DEFAULT_CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not os.environ.get("AGENTMARK_CJK_FONT_PATH") and os.path.exists(DEFAULT_CJK_FONT_PATH):
    os.environ["AGENTMARK_CJK_FONT_PATH"] = DEFAULT_CJK_FONT_PATH


def setup_logging(log_level: str = "INFO"):
    """
    Configure logging.

    Args:
        log_level: Log level ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Reduce httpx/openai chatter
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    # Silence agent step-by-step logs
    logging.getLogger("modules.alfworld.agent").setLevel(logging.WARNING)


def load_config(config_path: str) -> dict:
    """
    Load JSON config.

    Args:
        config_path: Path to config file

    Returns:
        config: Configuration dict
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    logger.info(f"Config loaded: {config_path}")
    return config


def resolve_api_config(config: dict, logger: logging.Logger) -> dict:
    """
    Fill API config: prefer config file, fall back to env vars.

    Supported env vars:
    - OPENAI_API_KEY
    - DEEPSEEK_API_KEY
    - OPENAI_BASE_URL
    - OPENAI_MODEL
    """
    api_key = (config.get("api_key") or "").strip()
    # Check for ${VAR} syntax
    if api_key.startswith("${") and api_key.endswith("}"):
        env_var_name = api_key[2:-1]
        env_value = os.environ.get(env_var_name)
        if env_value:
            config["api_key"] = env_value
            logger.info(f"Expanded api_key from env var: {env_var_name}")
        else:
            # Fallback for empty env var, treat as invalid
            api_key = ""

    if not api_key or api_key.lower() in {"your-api-key-here", "sk-xxx", "sk-xxxxxxxx"}:
        env_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if env_key:
            config["api_key"] = env_key
            logger.info("Loaded api_key from env (not written to config file)")
        else:
            logger.warning("No api_key found in config or env (OPENAI_API_KEY/DEEPSEEK_API_KEY)")

    base_url = (config.get("base_url") or "").strip()
    if not base_url:
        env_base_url = os.environ.get("OPENAI_BASE_URL")
        if env_base_url:
            config["base_url"] = env_base_url
            logger.info("Loaded base_url from env (OPENAI_BASE_URL)")

    model = (config.get("model") or "").strip()
    if not model:
        env_model = os.environ.get("OPENAI_MODEL")
        if env_model:
            config["model"] = env_model
            logger.info("Loaded model from env (OPENAI_MODEL)")

    return config


def sanitize_config_for_report(config: dict) -> dict:
    """Create a redacted config copy for reports/logs (avoid writing secrets to disk)."""
    safe_config = copy.deepcopy(config)
    if isinstance(safe_config, dict) and "api_key" in safe_config:
        safe_config["api_key"] = "***REDACTED***"
    return safe_config


def load_bit_stream(bit_stream_path: str = None) -> str:
    """
    Load watermark bit stream.

    Args:
        bit_stream_path: Optional path to bit stream file

    Returns:
        bit_stream: Bit stream string
    """
    logger = logging.getLogger(__name__)

    if bit_stream_path and os.path.exists(bit_stream_path):
        with open(bit_stream_path, 'r', encoding='utf-8') as f:
            bit_stream = f.read().strip()
        logger.info(f"Loaded bit stream from file: {bit_stream_path}, length: {len(bit_stream)}")
    else:
        # Generate default bit stream for testing
        import random
        bit_length = 1000
        bit_stream = ''.join(random.choice('01') for _ in range(bit_length))
        logger.info(f"Generated default bit stream, length: {len(bit_stream)}")

    return bit_stream


def encode_watermark_payload(bit_stream: str, watermark_config: dict, logger: logging.Logger) -> str:
    """Prepare payload bits based on ECC mode (rlnc or repetition)."""
    if not watermark_config:
        logger.warning("No watermark_config provided, using raw bit stream")
        return bit_stream

    payload_length = watermark_config.get('payload_bit_length')
    if not payload_length:
        logger.warning("watermark_config missing payload_bit_length, using raw bit stream")
        return bit_stream

    payload_bits = bit_stream[:payload_length]
    if len(payload_bits) < payload_length:
        logger.warning(
            "Bit stream shorter than %d, padding with zeros", payload_length
        )
        payload_bits = payload_bits.ljust(payload_length, '0')

    ecc_method = (watermark_config.get('ecc_method') or 'repetition').lower()
    if ecc_method not in {'rlnc', 'repetition'}:
        raise ValueError(
            f"Unsupported ecc_method for ALFWorld: {ecc_method}. "
            "Use 'rlnc' or 'repetition'."
        )

    if ecc_method == 'repetition':
        repetition_factor = int(watermark_config.get('repetition_factor', 1))
        if repetition_factor < 1:
            raise ValueError("repetition_factor must be >= 1")
        if repetition_factor > 1:
            payload_bits = ''.join(bit * repetition_factor for bit in payload_bits)
        logger.info(
            "Payload prepared: core %d bits -> message %d bits (ECC=repetition, factor=%d)",
            payload_length,
            len(payload_bits),
            repetition_factor
        )
    else:
        logger.info(
            "Payload prepared: core %d bits (ECC=rlnc)",
            payload_length
        )

    return payload_bits


def build_effective_bit_stream(
    base_message: str,
    config: dict,
    num_tasks: int,
    logger: logging.Logger
) -> str:
    """Build the final bit stream for the current round based on ECC and embedding strategy."""
    if not base_message:
        return base_message

    watermark_config = config.get('watermark_config', {})
    ecc_method = (watermark_config.get('ecc_method') or 'repetition').lower()
    if ecc_method == 'rlnc':
        from agentmark.core.rlnc_codec import DeterministicRLNC
        alfworld_config = config.get('alfworld_config', {})
        experiment_config = config.get('experiment_config', {})
        stream_key = watermark_config.get('rlnc_stream_key')
        if stream_key is None:
            stream_key = experiment_config.get('random_seed')
        if stream_key is None:
            stream_key = alfworld_config.get('random_seed', 2025)

        max_steps = alfworld_config.get('max_steps_per_task', 50)
        estimated_steps = max(1, num_tasks) * max_steps
        bits_per_step = int(watermark_config.get('rlnc_bits_per_step', 5))
        min_length = int(watermark_config.get('rlnc_min_stream_length', 8192))
        needed_len = max(estimated_steps * bits_per_step, min_length)

        encoder = DeterministicRLNC(base_message, stream_key=stream_key)
        effective_stream = encoder.get_stream(0, needed_len)

        logger.info(
            "RLNC stream ready: payload_bits=%d, stream_key=%s, length=%d",
            len(base_message),
            str(stream_key),
            len(effective_stream)
        )

        round_dir = config.get('experiment_context', {}).get('round_dir')
        if round_dir:
            meta_path = os.path.join(round_dir, "rlnc_meta.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'payload_bits': base_message,
                        'stream_key': stream_key,
                        'stream_length': len(effective_stream),
                        'is_rlnc': True
                    },
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.info(f"RLNC metadata saved: {meta_path}")

        return effective_stream

    strategy = watermark_config.get('embedding_strategy', 'one_time')
    if strategy != 'cyclic':
        return base_message

    alfworld_config = config.get('alfworld_config', {})
    max_steps = alfworld_config.get('max_steps_per_task', 50)
    estimated_steps = max(1, num_tasks) * max_steps
    # Differential scheme can embed up to log2(n) bits per step; estimate 3 bits per step.
    estimated_bits_needed = max(estimated_steps * 3, len(base_message))
    repeats = max(1, (estimated_bits_needed // len(base_message)) + 2)
    effective_stream = base_message * repeats
    logger.debug(
        "Cyclic embedding: need %d bits, generated %d bits (repeats=%d)",
        estimated_bits_needed,
        len(effective_stream),
        repeats
    )
    return effective_stream


def initialize_environment(config: dict) -> ALFWorldAdapter:
    """
    Initialize ALFWorld environment.

    Args:
        config: Configuration dict

    Returns:
        env_adapter: ALFWorld environment adapter
    """
    logger = logging.getLogger(__name__)

    alfworld_config = config.get('alfworld_config', {})
    config_path = alfworld_config.get('config_path', 'alfworld_base_config.yaml')
    train_eval = alfworld_config.get('train_eval', 'eval_in_distribution')

    logger.info(f"Initializing ALFWorld env: config={config_path}, train_eval={train_eval}")

    env_adapter = ALFWorldAdapter(
        config_path=config_path,
        train_eval=train_eval
    )

    logger.info(f"ALFWorld env ready, num_tasks={env_adapter.get_num_games()}")

    return env_adapter


def initialize_agent(
    client,
    config: dict,
    env_adapter: ALFWorldAdapter,
    use_watermark: bool,
    bit_stream: str = None
) -> ALFWorldAgent:
    """
    Initialize ALFWorld agent.

    Args:
        client: OpenAI client
        config: Configuration dict
        env_adapter: ALFWorld environment adapter
        use_watermark: Whether to enable watermarking
        bit_stream: Bit stream (required for watermarked mode)

    Returns:
        agent: ALFWorld agent controller
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Initializing agent: use_watermark={use_watermark}")

    agent = ALFWorldAgent(
        client=client,
        config=config,
        env_adapter=env_adapter,
        use_watermark=use_watermark,
        bit_stream=bit_stream
    )

    logger.info("Agent initialized")

    return agent


def select_tasks(env_adapter: ALFWorldAdapter, num_tasks: int, random_seed: int) -> list:
    """
    Select task IDs to run.

    Args:
        env_adapter: ALFWorld environment adapter
        num_tasks: Number of tasks
        random_seed: Random seed

    Returns:
        task_ids: List of task IDs
    """
    logger = logging.getLogger(__name__)

    total_games = env_adapter.get_num_games()

    if num_tasks is None or num_tasks <= 0:
        logger.info(
            f"num_tasks={num_tasks}; using all {total_games} tasks"
        )
        num_tasks = total_games

    if num_tasks > total_games:
        logger.warning(
            f"Requested {num_tasks} tasks exceeds available {total_games}, using all"
        )
        num_tasks = total_games

    # Reproducibility
    import random
    random.seed(random_seed)

    # Randomly sample tasks
    task_ids = random.sample(range(total_games), num_tasks)
    task_ids.sort()

    logger.info(f"Selected {len(task_ids)} tasks, random_seed={random_seed}")

    # Print task details
    print("\n" + "="*80)
    print("Task List Details")
    print("="*80)

    for idx, task_id in enumerate(task_ids, 1):
        task_desc = env_adapter.get_task_description(task_id)
        task_type = task_desc['task_type']
        game_file = os.path.basename(task_desc['game_file'])

        print(f"{idx:2d}. task_id={task_id:4d} | type={task_type:30s} | file={game_file}")

    print("="*80 + "\n")

    return task_ids


def main():
    """
    Main entry: run the full ALFWorld experiment pipeline.

    Steps:
    1. Parse CLI args
    2. Load config and bit stream
    3. Initialize environment and logging
    4. Run baseline (no watermark)
    5. Run watermarked
    6. Compute metrics
    7. Generate reports and charts

    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 7.4
    """
    # === 1. Parse CLI args ===
    parser = argparse.ArgumentParser(
        description='ALFWorld AgentMark experiment: evaluate watermark impact on agent performance'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config_alfworld.json',
        help='Config file path (default: config_alfworld.json)'
    )

    parser.add_argument(
        '--bit-stream',
        type=str,
        default=None,
        help='Bit stream file path (optional; auto-generate if omitted)'
    )

    parser.add_argument(
        '--num-tasks',
        type=int,
        default=None,
        help='Number of tasks to run (optional; overrides config)'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=None,
        help='Random seed (optional; overrides config)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (optional; overrides config)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level (default: INFO)'
    )

    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline run (watermarked only)'
    )

    parser.add_argument(
        '--skip-watermarked',
        action='store_true',
        help='Skip watermarked run (baseline only)'
    )

    parser.add_argument(
        '--task-ids-list',
        type=str,
        default=None,
        help='Comma-separated task IDs (e.g., "1,2,3"); overrides num-tasks and random selection'
    )

    parser.add_argument(
        '--num-rounds',
        type=int,
        default=None,
        help='Number of rounds to run (overrides config)'
    )

    parser.add_argument(
        '--eval-split',
        type=str,
        default=None,
        choices=['valid_seen', 'seen', 'id', 'valid_unseen', 'unseen', 'ood'],
        help='Eval split (overrides config; e.g., valid_seen, valid_unseen)'
    )

    args = parser.parse_args()

    # === 2. Logging ===
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("ALFWorld AgentMark experiment start")
    logger.info("="*80)

    try:
        # === 3. Load config ===
        config = load_config(args.config)
        if args.num_rounds is not None:
            config['experiment_config']['num_rounds'] = args.num_rounds

        experiment_config = config.get('experiment_config', {})
        auto_decode_enabled = experiment_config.get('auto_decode', False)
        num_rounds = max(1, experiment_config.get('num_rounds', 1))
        resample_tasks = experiment_config.get('resample_tasks_each_round', False)
        logger.info(
            f"Auto decode: {'enabled' if auto_decode_enabled else 'disabled'}"
            f" (decoder_task_index={experiment_config.get('decoder_task_index', 'all tasks')})"
        )
        logger.info(
            f"Planned rounds: {num_rounds}, resample_tasks_each_round: {resample_tasks}"
        )

        # CLI overrides
        if args.num_tasks is not None:
            config['alfworld_config']['num_tasks'] = args.num_tasks

        if args.random_seed is not None:
            config['alfworld_config']['random_seed'] = args.random_seed

        if args.output_dir is not None:
            config['experiment_config']['output_dir'] = args.output_dir

        # Map eval_split to train_eval
        alfworld_cfg = config.get('alfworld_config', {})
        eval_split = args.eval_split or alfworld_cfg.get('eval_split') or ''
        eval_split = eval_split.lower()
        if eval_split:
            split_map = {
                'valid_seen': 'eval_in_distribution',
                'seen': 'eval_in_distribution',
                'id': 'eval_in_distribution',
                'valid_unseen': 'eval_out_of_distribution',
                'unseen': 'eval_out_of_distribution',
                'ood': 'eval_out_of_distribution'
            }
            mapped = split_map.get(eval_split)
            if mapped:
                alfworld_cfg['train_eval'] = mapped
                config['alfworld_config'] = alfworld_cfg
                logger.info(f"Eval split: {eval_split} -> {mapped}")
            else:
                logger.warning(
                    f"Unknown eval_split: {eval_split}, keep train_eval={alfworld_cfg.get('train_eval')}"
                )

        # === 4. Load and prepare watermark bit stream ===
        bit_stream_path = args.bit_stream or experiment_config.get('bit_stream_path')
        raw_bit_stream = load_bit_stream(bit_stream_path)
        watermark_config = config.get('watermark_config', {})
        try:
            base_watermark_message = encode_watermark_payload(
                raw_bit_stream,
                watermark_config,
                logger
            )
        except ValueError:
            logger.error("Watermark payload encoding failed, aborting")
            sys.exit(1)
        if not base_watermark_message:
            logger.warning("Encoded watermark message is empty; embedding will be disabled")

        # === 4.5 Resolve API config ===
        config = resolve_api_config(config, logger)

        # === 5. Init OpenAI client ===
        logger.info("Initializing OpenAI client")
        client = OpenAI(
            api_key=config.get('api_key'),
            base_url=config.get('base_url')
        )

        # === 6. Init environment ===
        env_adapter = initialize_environment(config)

        # === 7. Task selection and multi-round control ===
        alfworld_config = config.get('alfworld_config', {})
        num_tasks = alfworld_config.get('num_tasks', 10)
        random_seed = alfworld_config.get('random_seed', 2025)

        if args.task_ids_list:
            try:
                base_task_ids = [int(x.strip()) for x in args.task_ids_list.split(',') if x.strip()]
                base_task_ids.sort()
                logger.info(f"Using CLI task list ({len(base_task_ids)}): {base_task_ids}")
            except ValueError:
                logger.error("Invalid --task-ids-list format; expected comma-separated integers")
                sys.exit(1)
        else:
            base_task_ids = select_tasks(env_adapter, num_tasks, random_seed)
        base_output_dir = experiment_config.get('output_dir', 'output/alfworld_experiments')
        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create session output directory
        output_dir = os.path.join(base_output_dir, base_timestamp)
        experiment_root = os.path.join(output_dir, 'reports')
        os.makedirs(experiment_root, exist_ok=True)
        round_records = []
        aggregate_info = None
        last_log_paths = None
        groups_to_run = int(not args.skip_baseline) + int(not args.skip_watermarked)
        total_tasks = len(base_task_ids) * num_rounds * groups_to_run
        progress_bar = tqdm(total=total_tasks, desc="Overall progress", unit="task", disable=(total_tasks == 0))

        for round_idx in range(num_rounds):
            round_label = round_idx + 1
            run_timestamp = base_timestamp if num_rounds == 1 else f"{base_timestamp}_r{round_label:02d}"
            round_dir = os.path.join(experiment_root, f"round_{round_label:02d}")
            os.makedirs(round_dir, exist_ok=True)
            context = config.setdefault('experiment_context', {})
            context.update({
                'run_timestamp': run_timestamp,
                'round_dir': round_dir,
                'experiment_root': experiment_root,
                'round_label': round_label
            })

            log_dir = os.path.join(output_dir, 'logs')
            log_paths = initialize_alfworld_log(
                log_dir=log_dir,
                experiment_name='alfworld_agentmark',
                timestamp=run_timestamp
            )
            last_log_paths = log_paths

            # Incremental report path
            partial_report_path = os.path.join(
                round_dir,
                f"evaluation_report_{run_timestamp}.json"
            )

            partial_state = {'baseline': [], 'watermarked': []}

            def write_partial_report():
                data = {
                    'metadata': {
                        'timestamp': run_timestamp,
                        'experiment_name': 'alfworld_agentmark_evaluation',
                        'config': sanitize_config_for_report(config)
                    },
                    'baseline_results': partial_state['baseline'],
                    'watermarked_results': partial_state['watermarked']
                }
                with open(partial_report_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            def on_baseline_update(results):
                partial_state['baseline'] = results
                write_partial_report()

            def on_watermarked_update(results):
                partial_state['watermarked'] = results
                write_partial_report()

            if round_idx == 0 or resample_tasks:
                seed_for_round = (random_seed + round_idx) if random_seed is not None else None
                task_ids = select_tasks(env_adapter, num_tasks, seed_for_round)
            else:
                task_ids = base_task_ids

            logger.info(f"Round {round_label}/{num_rounds} - task_ids: {task_ids}")

            baseline_results = []
            if not args.skip_baseline:
                logger.info("\n" + "="*80)
                logger.info(f"Starting baseline run - round {round_label}")
                logger.info("="*80)
                baseline_agent = initialize_agent(
                    client=client,
                    config=config,
                    env_adapter=env_adapter,
                    use_watermark=False,
                    bit_stream=None
                )
                baseline_results = run_baseline_experiment(
                    env_adapter=env_adapter,
                    agent=baseline_agent,
                    task_ids=task_ids,
                    config=config,
                    log_paths=log_paths,
                    progress_callback=progress_bar.update,
                    on_task_complete=on_baseline_update,
                    quiet=True
                )
                logger.info(
                    f"Baseline completed: {sum(1 for r in baseline_results if r['success'])}/{len(baseline_results)}"
                )
            else:
                logger.info("Skipping baseline run")

            watermarked_results = []
            if not args.skip_watermarked:
                logger.info("\n" + "="*80)
                logger.info(f"Starting watermarked run - round {round_label}")
                logger.info("="*80)
                effective_bit_stream = build_effective_bit_stream(
                    base_watermark_message,
                    config,
                    len(task_ids),
                    logger
                )
                watermarked_agent = initialize_agent(
                    client=client,
                    config=config,
                    env_adapter=env_adapter,
                    use_watermark=True,
                    bit_stream=effective_bit_stream
                )
                watermarked_results = run_watermarked_experiment(
                    env_adapter=env_adapter,
                    agent=watermarked_agent,
                    task_ids=task_ids,
                    config=config,
                    log_paths=log_paths,
                    progress_callback=progress_bar.update,
                    on_task_complete=on_watermarked_update,
                    quiet=True
                )
                logger.info(
                    f"Watermarked completed: {sum(1 for r in watermarked_results if r['success'])}/{len(watermarked_results)}"
                )
            else:
                logger.info("Skipping watermarked run")

            if baseline_results and watermarked_results:
                logger.info("\n" + "="*80)
                logger.info(f"Computing metrics - round {round_label}")
                logger.info("="*80)
                metrics = calculate_metrics(baseline_results, watermarked_results)
                log_experiment_summary(
                    log_path=log_paths['experiment'],
                    summary_path=log_paths['summary'],
                    metrics=metrics
                )
                print_experiment_summary(metrics)
                logger.info("\n" + "="*80)
                logger.info(f"Generating report - round {round_label}")
                logger.info("="*80)
                report_info = generate_report(
                    baseline_results=baseline_results,
                    watermarked_results=watermarked_results,
                    metrics=metrics,
                    output_dir=round_dir,
                    config=config
                )
                logger.info(f"Report saved to: {report_info['report_dir']}")
                if auto_decode_enabled:
                    task_index = experiment_config.get('decoder_task_index')
                    try:
                        decoded_path = auto_decode_report(
                            report_path=report_info['report_path'],
                            summary_path=report_info['summary_path'],
                            report_dir=report_info['report_dir'],
                            timestamp=report_info['timestamp'],
                            task_index=task_index
                        )
                        logger.info(f"Auto decode complete, output: {decoded_path}")
                    except Exception as e:
                        logger.error(f"Auto decode failed: {e}")
                round_records.append({
                    'round_index': round_label,
                    'metrics': metrics,
                    'report_info': report_info
                })
            else:
                logger.warning("Missing baseline or watermarked results; skipping metrics/report")

        if len(round_records) > 1:
            aggregate_info = generate_multi_round_summary(round_records, experiment_root)
            logger.info(f"Multi-round summary generated: {aggregate_info['summary_path']}")

        logger.info("\n" + "="*80)
        logger.info("ALFWorld AgentMark experiment complete")
        logger.info("="*80)
        if last_log_paths:
            logger.info(f"Last run log: {last_log_paths['experiment']}")
            logger.info(f"Last run summary: {last_log_paths['summary']}")
            logger.info(f"Last run log dir: {last_log_paths.get('run_dir')}")
        logger.info(f"Reports root: {experiment_root}")
        if aggregate_info:
            logger.info(f"Multi-round summary: {aggregate_info['summary_path']}")

    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\nExperiment failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()


def auto_decode_report(
    report_path: str,
    summary_path: str,
    report_dir: str,
    timestamp: str,
    task_index: int = None
):
    """
    Run auto decode and append results to the summary file.
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    watermarked_results = report.get('watermarked_results', [])
    if not watermarked_results:
        raise ValueError("Report missing watermarked_results; cannot decode")

    if task_index is not None:
        target_indices = [task_index]
    else:
        target_indices = list(range(len(watermarked_results)))

    decode_outputs = []
    for idx in target_indices:
        if idx < 0 or idx >= len(watermarked_results):
            raise IndexError(f"Decode task index {idx} out of range (total {len(watermarked_results)})")

        task = watermarked_results[idx]
        decode_info = decode_task_bits(task)
        expected_bits = (task.get('watermark_stats') or {}).get('total_bits_embedded', 0)
        verified = (decode_info['total_bits'] == expected_bits) and not decode_info.get('errors')
        decode_outputs.append({
            'task_index': idx,
            'task_id': task.get('task_id'),
            'task_type': task.get('task_type'),
            'decoded_bit_stream': decode_info['bit_stream'],
            'total_bits': decode_info['total_bits'],
            'expected_bits': expected_bits,
            'verified': verified,
            'errors': decode_info.get('errors', [])
        })

    output_path = os.path.join(report_dir, f"decoded_bits_{timestamp}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(decode_outputs, f, ensure_ascii=False, indent=2)

    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "-" * 80 + "\n")
        f.write("Auto decode verification\n")
        f.write("-" * 80 + "\n")
        for item in decode_outputs:
            status = "PASS" if item['verified'] else "FAIL"
            f.write(
                f"Task {item['task_id']} (index {item['task_index']}): "
                f"decoded {item['total_bits']} bits / expected {item['expected_bits']} -> {status}\n"
            )
            if item['errors']:
                for err in item['errors']:
                    f.write(f"    warning: {err}\n")
        f.write("-" * 80 + "\n")

    return output_path


def generate_multi_round_summary(round_records, experiment_root):
    """Generate a multi-round summary report."""
    from statistics import mean, pstdev
    aggregate_dir = os.path.join(experiment_root, 'aggregate')
    os.makedirs(aggregate_dir, exist_ok=True)
    summary_path = os.path.join(aggregate_dir, 'multi_round_summary.txt')
    json_path = os.path.join(aggregate_dir, 'multi_round_metrics.json')
    metrics_fields = [
        ('baseline_success_rate', 'Baseline success rate', 100, True),
        ('watermarked_success_rate', 'Watermarked success rate', 100, True),
        ('success_rate_diff', 'Success rate diff', 100, True),
        ('baseline_avg_steps', 'Baseline avg steps', 1, False),
        ('watermarked_avg_steps', 'Watermarked avg steps', 1, False),
        ('avg_steps_diff', 'Avg steps diff', 1, False),
        ('baseline_avg_duration', 'Baseline avg duration', 1, False),
        ('watermarked_avg_duration', 'Watermarked avg duration', 1, False),
        ('duration_diff', 'Duration diff', 1, False),
        ('step_increase_rate', 'Step increase rate (successful paths)', 1, True)
    ]
    aggregated_stats = {}

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Multi-round summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Rounds: {len(round_records)}\n\n")
        for field, label, scale, is_percent in metrics_fields:
            values = [rec['metrics'].get(field, 0) for rec in round_records]
            avg = mean(values)
            std = pstdev(values) if len(values) > 1 else 0.0
            min_v = min(values)
            max_v = max(values)
            if is_percent:
                f.write(
                    f"{label}: avg {avg*scale:+.2f}%  / std {std*scale:.2f}%  / range {min_v*scale:+.2f}% ~ {max_v*scale:+.2f}%\n"
                )
            else:
                f.write(
                    f"{label}: avg {avg*scale:+.2f}  / std {std*scale:.2f}  / range {min_v*scale:+.2f} ~ {max_v*scale:+.2f}\n"
                )
            aggregated_stats[field] = {
                'values': values,
                'mean': avg,
                'std': std,
                'min': min_v,
                'max': max_v
            }
        f.write("\nPer-round:\n")
        for rec in round_records:
            f.write(f"- round {rec['round_index']}: report {rec['report_info']['report_dir']}\n")

        chart_path = None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            rounds = [rec['round_index'] for rec in round_records]
            baseline_success = [rec['metrics'].get('baseline_success_rate', 0) * 100 for rec in round_records]
            watermarked_success = [rec['metrics'].get('watermarked_success_rate', 0) * 100 for rec in round_records]
            baseline_steps = [rec['metrics'].get('baseline_avg_steps', 0) for rec in round_records]
            watermarked_steps = [rec['metrics'].get('watermarked_avg_steps', 0) for rec in round_records]
            baseline_duration = [rec['metrics'].get('baseline_avg_duration', 0) for rec in round_records]
            watermarked_duration = [rec['metrics'].get('watermarked_avg_duration', 0) for rec in round_records]

            fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
            axes[0].plot(rounds, baseline_success, marker='o', label='Baseline success rate')
            axes[0].plot(rounds, watermarked_success, marker='o', label='Watermarked success rate')
            axes[0].set_ylabel('Success rate (%)')
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].plot(rounds, baseline_steps, marker='s', label='Baseline steps')
            axes[1].plot(rounds, watermarked_steps, marker='s', label='Watermarked steps')
            axes[1].set_ylabel('Avg steps')
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            axes[2].plot(rounds, baseline_duration, marker='^', label='Baseline duration')
            axes[2].plot(rounds, watermarked_duration, marker='^', label='Watermarked duration')
            axes[2].set_ylabel('Avg duration (s)')
            axes[2].set_xlabel('Round')
            axes[2].legend()
            axes[2].grid(alpha=0.3)

            fig.suptitle('Multi-round variability')
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            chart_path = os.path.join(aggregate_dir, 'multi_round_variability.png')
            plt.savefig(chart_path, dpi=200)
            plt.close(fig)
            f.write(f"\nChart: {chart_path}\n")
        except Exception as chart_err:
            logger = logging.getLogger(__name__)
            logger.warning(f"Multi-round chart generation failed: {chart_err}")
            chart_path = None

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'round_records': round_records,
                'metrics_stats': aggregated_stats,
                'chart_path': chart_path
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    return {'summary_path': summary_path, 'metrics_path': json_path, 'chart_path': chart_path}


if __name__ == "__main__":
    main()
