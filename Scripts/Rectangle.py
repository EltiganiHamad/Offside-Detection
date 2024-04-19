from Point import Point  # Import Point class from Point.py
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def min_x(self) -> float:
        return self.x

    @property
    def min_y(self) -> float:
        return self.y

    @property
    def max_x(self) -> float:
        return self.x + self.width

    @property
    def max_y(self) -> float:
        return self.y + self.height

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)


    def pad(self, padding: float):
            return Rect(
                x=self.x - padding,
                y=self.y - padding,
                width=self.width + 2*padding,
                height=self.height + 2*padding
            )

    def contains_point(self, point: Point) -> bool:
            return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y