from Color import Color  # Import Color class from Color.py
from Detection import Detection # Import Detection class from Detection.py
import numpy as np
import helpers
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
@dataclass
class BaseAnnotator:
    colors: List[Optional[Color]]
    thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
              # player_color_selector(detection.offside)
              annotated_image = helpers.draw_ellipse(
                  image=image,
                  rect=detection.rect,
                  color=self.colors[detection.class_id],
                  thickness=self.thickness
              )
        return annotated_image
