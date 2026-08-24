"""Optional lifecycle manager for the project-local Gazebo launcher."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parent / "scripts" / "launch_gazebo.sh"


class GazeboStack:
    def __init__(self, *, headless: bool, mode: str, seed: int, socket_path: str):
        command = [
            "bash", str(LAUNCHER), "--mode", mode, "--seed", str(seed),
            "--socket", socket_path,
        ]
        if headless:
            command.append("--headless")
        self.process = subprocess.Popen(command, start_new_session=True)

    def close(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def maybe_launch(args, socket_path: str) -> GazeboStack | None:
    return GazeboStack(
        headless=bool(args.headless), mode=args.mode, seed=args.seed,
        socket_path=socket_path,
    ) if args.launch else None
