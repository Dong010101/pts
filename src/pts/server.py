"""Serve the existing OpenPI model with a PTS decoder, without patching OpenPI."""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Existing OpenPI training config name")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--decoder", choices=["zero_noise_one_step", "stochastic_10_step"], default="zero_noise_one_step")
    parser.add_argument("--action-space", choices=["absolute_tcp", "relative_increment"], default="absolute_tcp")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8025)
    args = parser.parse_args()
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from .openpi_policy import create_decoder

    logging.basicConfig(level=logging.INFO)
    policy = create_decoder(args.config, args.checkpoint, decoder=args.decoder, action_space=args.action_space)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()

