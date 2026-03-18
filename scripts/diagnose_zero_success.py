import os
import numpy as np
import torch

from lerobot.envs import libero as libero_module
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_eval import eval_one

MODEL_PATH = "models/smolvla"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def run_case(gripper_value):
    original_step = libero_module.LiberoEnv.step

    def patched_step(self, action):
        a = np.asarray(action)
        if a.ndim == 1 and a.shape[0] == 6:
            a = np.concatenate([a, np.array([gripper_value], dtype=a.dtype)])
        elif a.ndim == 2 and a.shape[1] == 6:
            pad = np.full((a.shape[0], 1), gripper_value, dtype=a.dtype)
            a = np.concatenate([a, pad], axis=1)
        return original_step(self, a)

    libero_module.LiberoEnv.step = patched_step

    model = SmolVLAPolicy.from_pretrained(MODEL_PATH)
    model.to(torch.device(DEVICE))
    model.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=model.config,
        pretrained_path=MODEL_PATH,
        preprocessor_overrides={
            "device_processor": {"device": DEVICE},
            "rename_observations_processor": {
                "rename_map": {
                    "observation.images.image": "observation.images.camera1",
                    "observation.images.image2": "observation.images.camera2",
                }
            },
        },
    )

    env_cfg = LiberoEnvConfig(task="libero_spatial", task_ids=[0], control_mode="relative")
    envs = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False, trust_remote_code=False)
    env = envs["libero_spatial"][0]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=model.config)

    metrics = eval_one(
        env,
        policy=model,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=2,
        max_episodes_rendered=0,
        videos_dir=None,
        return_episode_data=False,
        start_seed=0,
    )

    env.close()
    successes = [int(x) for x in metrics["successes"]]
    print(f"gripper={gripper_value} successes={successes} mean={float(np.mean(successes)):.3f}")


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"device={DEVICE}")
    for g in (-1.0, 0.0, 1.0):
        run_case(g)


if __name__ == "__main__":
    main()
