from Color import Color  # Import Color class from Color.py
from Point import Point  # Import Point class from Point.py
from Rectangle import Rect
from Detection import Detection
import numpy as np
import cv2
import helpers
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
# text annotator to display tracker_id
@dataclass
class TextAnnotator:
    background_color: Optional[Color]
    text_color: Optional[Color]
    text_thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            # if tracker_id is not assigned skip annotation
            if detection.tracker_id is None:
              if detection.team_id is not None:
                  # continue

                  # calculate text dimensions
                  size, _ = cv2.getTextSize(
                      str(detection.tracker_id),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.7,
                      thickness=self.text_thickness)
                  width, height = size

                  # calculate text background position
                  center_x, center_y = detection.rect.bottom_center.int_xy_tuple
                  x = center_x - width // 2
                  y = center_y - height // 2 + 10

                  # draw background
                  annotated_image = helpers.draw_filled_rect(
                      image=annotated_image,
                      rect=Rect(x=x, y=y, width=(width/2), height=height).pad(padding=3),
                      color=self.background_color)

                  # draw text
                  annotated_image = helpers.draw_text(
                      image=annotated_image,
                      anchor=Point(x=x, y=y + height),
                      text= str(detection.team_id),
                      color=self.text_color,
                      thickness=self.text_thickness)
        return annotated_image