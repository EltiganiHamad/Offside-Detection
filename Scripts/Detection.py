from Rectangle import Rect # Import Rect class from Rectangle.py
import helpers
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Detection:
    rect: Optional[Rect]
    class_id: int
    class_name: str
    confidence: float
    team_id: int
    offside: str = None
    angle: float = None
    tracker_id: Optional[int] = None


    @classmethod
    def from_player_results(self, results) :
          result = []
          cropped_imgs = []
          names = results[0].names
          boxes = results[0].boxes.cpu().numpy()
          img = results[0].orig_img
            
          max_width = 64
          max_height = 64
          # Find maximum width and height
          for i in range(0, len(boxes.cls)):
            x_min, y_min, x_max, y_max = boxes.xyxy[i]
            cropped = helpers.crop_image(img, x_min, y_min, x_max, y_max)
            max_width = max(max_width, cropped.shape[1])
            max_height = max(max_height, cropped.shape[0])

          # Resize all cropped images to the maximum dimensions
          for i in range(0, len(boxes.cls)):
            x_min, y_min, x_max, y_max = boxes.xyxy[i]
            cropped = helpers.crop_image(img, x_min, y_min, x_max, y_max)
            cropped_resized = cv2.resize(cropped, (max_width, max_height))
            cropped_imgs.append(cropped_resized)

          # Flatten and stack cropped images
          cropped_imgs_flat = [cropped.flatten() for cropped in cropped_imgs]
          cropped_imgs_stacked = np.stack(cropped_imgs_flat)
          labels = helpers.agglomerative_clustering(cropped_imgs_stacked)
 
        
          for i in range(0, len(boxes.cls)):
            x_min, y_min, x_max, y_max = boxes.xyxy[i]
            class_id = int(boxes.cls[i])
            class_name = names[class_id]
            if class_name == 'player':
              team = labels[i]
            else:
              team = None
            confidence = boxes.conf[i]
            result.append(Detection(
                rect=Rect(
                          x=x_min,
                          y=y_min,
                          width=(x_max - x_min),
                          height=(y_max - y_min)
                        ),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        team_id = team
                    ))
          return result
    @classmethod
    def from_ball_results(self, results) :
          result = []
          names = results[0].names
          boxes = results[0].boxes.cpu().numpy()
          for i in range(0, len(boxes.cls)):
            x_min, y_min, x_max, y_max = boxes.xyxy[i]
            class_id = int(boxes.cls[i])
            class_name = names[class_id]
            if class_name == 'ball':
                confidence = boxes.conf[i]
                result.append(Detection(
                    rect=Rect(
                              x=x_min,
                              y=y_min,
                              width=(x_max - x_min),
                              height=(y_max - y_min)
                            ),
                            class_id=0,
                            class_name='ball',
                            confidence=confidence,
                            team_id = None
                        ))
          return result



