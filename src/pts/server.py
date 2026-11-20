"""Serve the existing OpenPI model with a PTS decoder, without patching OpenPI."""

import argparse
import logging
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from .model_contract import TASKS, load_config
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config", help="Existing OpenPI training config name")
    selection.add_argument("--task", choices=TASKS, help="Verified flip/polish training configuration")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decoder", choices=["zero_noise_one_step", "stochastic_10_step"], default="zero_noise_one_step")
    parser.add_argument("--action-space", choices=["absolute_tcp", "relative_increment"], default="absolute_tcp")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8025)
    parser.add_argument("--inspect", action="store_true", help="Validate config/transforms/checkpoint assets without loading weights")
    args = parser.parse_args()
    config_name = TASKS[args.task][0] if args.task else args.config
    if args.inspect:
        config = load_config(config_name, args.checkpoint, action_space=args.action_space)
        print(json.dumps({"config": config_name, "checkpoint": args.checkpoint,
                          "metadata": config.policy_metadata}, indent=2))
        return
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from .openpi_policy import create_decoder

    logging.basicConfig(level=logging.INFO)
    policy = create_decoder(config_name, args.checkpoint, decoder=args.decoder, action_space=args.action_space)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()

