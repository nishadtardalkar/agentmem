from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MemoryConfig:
    """Configuration for the dual-path latent memory manager."""

    encoder_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    main_model_id: str = "Qwen/Qwen2.5-32B-Instruct"
    encoder_layer: int = -1
    encoder_device: str = "cuda"
    encoder_dtype: str = "bfloat16"

    tau_break: float = 0.75
    tau_retrieve: float = 0.70
    retrieve_top_k: int = 5

    data_dir: Path = field(default_factory=lambda: Path("data/memory"))
    max_readout_chars: int = 2000

    main_load_in_4bit: bool = True
    main_bnb_4bit_compute_dtype: str = "bfloat16"
    main_bnb_4bit_quant_type: str = "nf4"

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    @property
    def faiss_path(self) -> Path:
        return self.data_dir / "keys.faiss"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "episodes.db"

    @property
    def id_map_path(self) -> Path:
        return self.data_dir / "faiss_ids.json"
