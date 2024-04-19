from Color import Color  # Import Color class from Color.py
from Detection import Detection # Import Detection class from Detection.py
import numpy as np
import helpers
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
# dedicated annotator to draw possession markers on video frames
@dataclass
class MarkerAnntator:

    color: Optional[Color]

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = helpers.draw_marker(
                image=image,
                anchor=detection.rect.top_center,
                color=self.color)
        return annotated_image