from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Mapping


class RuntimeMetrics:
    """Small dependency-free Prometheus text exporter for runtime timings."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._max_samples = 1024
        self._counters: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._samples: defaultdict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(list)

    @staticmethod
    def _labels(labels: Mapping[str, object] | None) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(key), str(value)) for key, value in (labels or {}).items()))

    def increment(self, name: str, value: float = 1.0, *, labels: Mapping[str, object] | None = None) -> None:
        with self._lock:
            self._counters[(name, self._labels(labels))] += value

    def observe(self, name: str, value: float, *, labels: Mapping[str, object] | None = None) -> None:
        with self._lock:
            samples = self._samples[(name, self._labels(labels))]
            samples.append(float(value))
            if len(samples) > self._max_samples:
                del samples[: len(samples) - self._max_samples]

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            counters = dict(self._counters)
            samples = {key: list(values) for key, values in self._samples.items()}

        def render_labels(labels: tuple[tuple[str, str], ...]) -> str:
            if not labels:
                return ""
            escaped = [
                f'{key}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
                for key, value in labels
            ]
            return "{" + ",".join(escaped) + "}"

        for (name, labels), value in sorted(counters.items()):
            lines.append(f"{name}_total{render_labels(labels)} {value}")
        for (name, labels), values in sorted(samples.items()):
            if not values:
                continue
            label_text = render_labels(labels)
            lines.append(f"{name}_count{label_text} {len(values)}")
            lines.append(f"{name}_sum{label_text} {sum(values)}")
        return "\n".join(lines) + "\n"


runtime_metrics = RuntimeMetrics()


__all__ = ["RuntimeMetrics", "runtime_metrics"]
