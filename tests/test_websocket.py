import asyncio

import numpy as np

from forcebench.runner.transport import OpenPIWebSocketPolicyClient
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from websockets.asyncio.server import serve

from pts.forcebench import check_server_metadata
from pts.mock import MockTransport
from pts.openpi_policy import DecoderPolicy
from pts.settings import RobotSettings


def test_original_openpi_server_to_forcebench_client():
    class Policy:
        def infer(self, observation, noise):
            assert observation["image"].shape == (224, 224, 3)
            assert noise.shape == (50, 32) and not noise.any()
            return {"actions": np.tile(observation["state"], (50, 1))}
    metadata = MockTransport().server_metadata
    policy = DecoderPolicy(Policy(), horizon=50, action_dim=32, decoder="zero_noise_one_step", metadata=metadata)
    server = WebsocketPolicyServer(policy, metadata=metadata)
    async def scenario():
        async with serve(server._handler, "127.0.0.1", 0, compression=None, max_size=None) as listener:
            port = listener.sockets[0].getsockname()[1]
            client = OpenPIWebSocketPolicyClient(f"ws://127.0.0.1:{port}")
            try:
                await client.connect()
                check_server_metadata(client.server_metadata, RobotSettings())
                response = await client.infer({"image": np.zeros((224, 224, 3), np.uint8), "state": np.arange(7, dtype=np.float32)})
                np.testing.assert_array_equal(response["actions"], np.tile(np.arange(7), (50, 1)))
                assert "pts_timing" in response and "server_timing" in response
            finally:
                await client.close()
    asyncio.run(scenario())
