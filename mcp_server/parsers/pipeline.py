"""Composition vocabulary for text transformations.

The PGN cleaning pipeline and similar text-transformation chains compose as a
list of ``Step`` instances. Each ``Step`` is a stateless ``apply(text) -> text``
function bound as a class so the pipeline is introspectable (you can name the
step, list it in tests, and reorder confidently).

Why a class over a function:
    - Named log lines: ``logger.info("step %s on %d chars", step.name, len(text))``
    - Reordering is friction-free when the order is the only signal.
    - Tests can construct a single ``Step`` and assert behavior in isolation
      without spinning up the full pipeline.

Example::

    pipeline = Pipeline([
        NormalizeUnicode(),
        MaskComments(),
        StripEscapeLines(),
    ])
    canonical = pipeline.run(input_text)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

__all__ = ["Pipeline", "Step", "chain"]


class Step(ABC):
    """Stateless text-transformation step.

    Subclasses MUST implement :meth:`apply` and may optionally set
    :attr:`name` to override the default ``__class__.__name__``-based label.
    """

    name: str = ""

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.__dict__.get("name"):
            cls.name = cls.__name__

    @abstractmethod
    def apply(self, text: str) -> str:
        """Pure: ``text -> text``. Must NOT mutate inputs."""


class Pipeline:
    """Ordered composition of :class:`Step` instances.

    Construction is eager so an empty pipeline is a unit-of-work no-op::

        Pipeline([]).run(text) == text

    The pipeline is intentionally synchronous; async variants are unnecessary
    for pure-text transforms and would only add overhead.
    """

    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps: tuple[Step, ...] = tuple(steps)

    @property
    def steps(self) -> tuple[Step, ...]:
        return self._steps

    def run(self, text: str) -> str:
        current = text
        for step in self._steps:
            current = step.apply(current)
        return current

    def __len__(self) -> int:
        return len(self._steps)

    def __getitem__(self, idx: int) -> Step:
        return self._steps[idx]


def chain(*steps: Step) -> Pipeline:
    """Functional constructor — ``chain(A, B, C)`` is equivalent to ``Pipeline([A, B, C])``."""
    return Pipeline(list(steps))
