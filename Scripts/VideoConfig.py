from dataclasses import dataclass
@dataclass(frozen=True)
class VideoConfig:
    fps: float
    width: int
    height: int