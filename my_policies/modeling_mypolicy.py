
"""MyPolicy: local SmolVLA augmentation scaffold.

This lives in the project repo (not vendored LeRobot code) and subclasses
SmolVLAPolicy while preserving base behavior for now.

Example:
    from my_policies.modeling_mypolicy import MyPolicy

    policy = MyPolicy.from_libero_pretrained()
"""

from pathlib import Path

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class MyPolicy(SmolVLAPolicy):
    """Augmentation-ready SmolVLA policy initialized from Libero checkpoint."""

    config_class = SmolVLAConfig
    name = "mypolicy"

    @classmethod
    def from_libero_pretrained(
        cls,
        pretrained_name_or_path: str | Path = "HuggingFaceVLA/smolvla_libero",
        *,
        config: SmolVLAConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> "MyPolicy":
        """Load MyPolicy with Libero SmolVLA weights by default."""
        return cls.from_pretrained(
            pretrained_name_or_path=pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )
