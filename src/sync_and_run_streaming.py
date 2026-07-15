"""
Remote Execution Helper for Streaming TTS Benchmarks.
Connects to a remote Lightning AI Studio, uploads local script updates,
and triggers the evaluation remotely on the target GPU node.
"""

import argparse
import base64
import logging
import os
from typing import List

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sync_and_run(
    studio_name: str,
    teamspace_name: str,
    user_name: str,
    local_base_dir: str,
    project_dir: str,
    files_to_sync: List[str]
) -> None:
    """
    Connects to the specified remote Studio, uploads files, and executes the streaming benchmark.
    
    Args:
        studio_name: Name of the remote Lightning AI Studio.
        teamspace_name: Name of the Lightning AI Teamspace.
        user_name: Username of the Studio owner.
        local_base_dir: Local project base directory.
        project_dir: Path to the project root folder on the remote Studio.
        files_to_sync: List of relative file paths to synchronize.
    """
    try:
        from lightning_sdk import Studio
    except ImportError:
        logger.error(
            "The 'lightning-sdk' package is required but could not be imported. "
            "Please install it using: pip install lightning-sdk"
        )
        return

    logger.info(f"[CONNECT] Connecting to remote Lightning AI Studio '{studio_name}'...")
    try:
        studio = Studio(name=studio_name, teamspace=teamspace_name, user=user_name)
    except Exception as e:
        logger.error(f"Failed to connect to Lightning AI Studio: {e}")
        raise

    # Ensure remote target directory structure exists
    studio.run(f"mkdir -p {project_dir}/src")

    for rel_path in files_to_sync:
        local_path = os.path.join(local_base_dir, rel_path)
        logger.info(f"Reading local file: {local_path}...")
        if not os.path.exists(local_path):
            logger.error(f"Local file does not exist: {local_path}")
            continue

        with open(local_path, "rb") as f:
            content = f.read()

        encoded_content = base64.b64encode(content).decode("utf-8")
        remote_path = os.path.join(project_dir, rel_path)
        
        logger.info(f"Uploading {rel_path} to remote Studio: {remote_path}...")
        # Write base64 encoded chunks to avoid string shell escaping issues
        write_cmd = (
            f"python3 -c \"import base64; "
            f"open('{remote_path}', 'wb').write(base64.b64decode('{encoded_content}'))\""
        )
        studio.run(write_cmd)

    logger.info("[SUCCESS] All files uploaded successfully!")

    # Run the streaming benchmark remotely using the venv interpreter
    logger.info("[RUN] Running streaming benchmark remotely on Tesla T4...")
    run_cmd = (
        f"cd {project_dir} && "
        f"export COQUI_TOS_AGREED=1 && "
        f"/home/zeus/miniconda3/envs/venv310/bin/python src/benchmark_streaming.py"
    )
    
    try:
        output = studio.run(run_cmd)
        logger.info("\n================== REMOTE EXECUTION OUTPUT ==================")
        print(output)
        logger.info("=============================================================")
    except Exception as e:
        logger.error(f"Remote command execution failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync local files and run streaming benchmarks on remote Lightning AI GPU Studio."
    )
    parser.add_argument("--studio", type=str, required=True, help="Lightning Studio name")
    parser.add_argument("--teamspace", type=str, required=True, help="Lightning Teamspace name")
    parser.add_argument("--user", type=str, required=True, help="Lightning AI username")
    parser.add_argument(
        "--local-dir", 
        type=str, 
        default=os.getcwd(), 
        help="Local project root directory"
    )
    parser.add_argument(
        "--remote-dir", 
        type=str, 
        required=True,
        help="Remote project root directory in Lightning Studio (e.g., /teamspace/studios/this_studio/voice)"
    )
    args = parser.parse_args()

    sync_files = ["src/utils.py", "src/benchmark_streaming.py"]
    
    sync_and_run(
        studio_name=args.studio,
        teamspace_name=args.teamspace,
        user_name=args.user,
        local_base_dir=args.local_dir,
        project_dir=args.remote_dir,
        files_to_sync=sync_files
    )
