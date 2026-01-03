"""Model switching test with Gaussian random image matrices."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class DummyModel:
    """Lightweight model wrapper for switch testing."""

    name: str
    predict_fn: Callable[[np.ndarray], np.ndarray]

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return self.predict_fn(batch)


def generate_gaussian_images(
    batch_size: int,
    height: int,
    width: int,
    channels: int,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int | None = 42,
) -> np.ndarray:
    """Generate Gaussian-distributed image tensors.

    Returns a float32 array of shape (batch_size, height, width, channels).
    """
    rng = np.random.default_rng(seed)
    images = rng.normal(loc=mean, scale=std, size=(batch_size, height, width, channels))
    return images.astype(np.float32)


def build_models() -> List[DummyModel]:
    """Create dummy models that apply different transformations."""

    def model_a(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def model_b(x: np.ndarray) -> np.ndarray:
        return np.clip(x * 0.5 + 0.1, -1.0, 1.0)

    def model_c(x: np.ndarray) -> np.ndarray:
        return np.sign(x) * np.sqrt(np.abs(x))

    return [
        DummyModel(name="tanh", predict_fn=model_a),
        DummyModel(name="scaled_clip", predict_fn=model_b),
        DummyModel(name="signed_sqrt", predict_fn=model_c),
    ]


def run_switch_test(
    images: np.ndarray,
    models: Iterable[DummyModel],
    switch_order: Iterable[int],
) -> Dict[str, float]:
    """Run prediction while switching models and report latency per model."""
    latency_ms: Dict[str, float] = {model.name: 0.0 for model in models}
    model_list = list(models)

    for idx in switch_order:
        model = model_list[idx]
        start = time.perf_counter()
        _ = model.predict(images)
        elapsed = (time.perf_counter() - start) * 1000.0
        latency_ms[model.name] += elapsed

    return latency_ms


def main() -> None:
    images = generate_gaussian_images(
        batch_size=8,
        height=256,
        width=256,
        channels=3,
        mean=0.0,
        std=1.0,
        seed=123,
    )
    models = build_models()

    switch_order = [0, 1, 2, 1, 0, 2, 0, 1]
    latency_ms = run_switch_test(images, models, switch_order)

    print("Model switch test results (accumulated latency ms):")
    for name, latency in latency_ms.items():
        print(f"- {name}: {latency:.3f} ms")


if __name__ == "__main__":
    main()
