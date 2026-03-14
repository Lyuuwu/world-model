from dataclasses import dataclass

@dataclass
class ObsSpec:
    shape: tuple[int, ...]
    discrete: bool = False
    dtype: str = 'float32'
    classes: int | None = None

    @property
    def is_image(self) -> bool:
        return len(self.shape) == 3

    def __post_init__(self):
        if self.discrete and self.classes is None:
            raise ValueError(f"discrete ObsSpec must have classes set, got shape={self.shape}")
        if not self.discrete and self.classes is not None:
            raise ValueError(f"continuous ObsSpec should not have classes, got classes={self.classes}")