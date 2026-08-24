"""Display or save the exact 640x640 RGB tensor seen by the Actor."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from aerointercept.gazebo.client import GazeboBridgeClient
from aerointercept.gazebo.config import load_gazebo_config
from aerointercept.gazebo.protocol import image_from_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--socket", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    if not args.display and not args.output:
        parser.error("select --display and/or --output")
    cfg = load_gazebo_config(args.config)
    client = GazeboBridgeClient(args.socket or cfg.gazebo.bridge.socket)
    client.connect(float(cfg.gazebo.bridge.startup_timeout_s))
    sequence = -1
    try:
        while True:
            snapshot = client.snapshot(sequence, timeout=5.0)
            sequence = int(snapshot["sequence"])
            rgb = np.transpose(image_from_snapshot(snapshot), (1, 2, 0))
            bgr = rgb[..., ::-1]
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), bgr)
                if not args.display:
                    print(path)
                    break
            if args.display:
                cv2.imshow("AeroIntercept Actor RGB 640x640", bgr)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        client.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
