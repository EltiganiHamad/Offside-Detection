from __future__ import annotations
from roboflow import Roboflow
from ultralytics import YOLO
import cv2
from dataclasses import dataclass
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from typing import List, Optional
from sklearn.cluster import AgglomerativeClustering
import os
from Rectangle import Rect
from Color import Color 
from Detection import Detection
from BaseAnnotator import BaseAnnotator
from TextAnnotator import TextAnnotator
from MarkerAnntator import MarkerAnntator
from VideoConfig import VideoConfig
from Point import Point
from VideoConfig import VideoConfig


MARKER_CONTOUR_THICKNESS = 2
MARKER_WIDTH = 20
MARKER_HEIGHT = 20
MARKER_MARGIN = 10
# black
MARKER_CONTOUR_COLOR_HEX = "000000"
MARKER_CONTOUR_COLOR = Color.from_hex_string(MARKER_CONTOUR_COLOR_HEX)

# red
PLAYER_MARKER_FILL_COLOR_HEX = "FF0000"
PLAYER_MARKER_FILL_COLOR = Color.from_hex_string(PLAYER_MARKER_FILL_COLOR_HEX)

# green
BALL_MERKER_FILL_COLOR_HEX = "00FF00"
BALL_MARKER_FILL_COLOR = Color.from_hex_string(BALL_MERKER_FILL_COLOR_HEX)


def load_dataset():
    """
    Downlaods and extracts dataset from roboflow workspace using api-key with annotations in yolov8 format.

    Args:
        None
    Returns:
        Annotated dataset
    """
    # API authenticiation through API key
    rf = Roboflow(api_key="IuMsFsMjooEROXB64ayh")
    # Retrive project from workspace
    project = rf.workspace("offside-detection-l6jvj").project("football-player-detection-uix6g")
    # Extract version 1 of the dataset
    version = project.version(1)
    # Download and extract dataset with YOLOv8 annotation format
    dataset = version.download("yolov8")
    return dataset


def load_models():
    # Load model
    player_model = YOLO('models/player detector.pt')
    ball_model = YOLO('models/ball detector.pt')

    return player_model, ball_model


def crop_image(image, x_min, y_min, x_max, y_max):
    """
    Crops out a tiny box around the center point of a given bounding box from the image.

    Args:
        image: A NumPy array representing the image.
        x_min: The minimum x-coordinate of the bounding box.
        y_min: The minimum y-coordinate of the bounding box.
        x_max: The maximum x-coordinate of the bounding box.
        y_max: The maximum y-coordinate of the bounding box.

    Returns:
        A cropped image containing the tiny box around the center point of the bounding box.
    """
    # Calculate the center point of the bounding box
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)

    # Define the dimensions of the tiny box
    box_width = 10
    box_height = 10

    # Calculate the coordinates of the top-left and bottom-right corners of the tiny box
    box_x_min = max(center_x - box_width // 2, 0)
    box_y_min = max(center_y - box_height // 2, 0)
    box_x_max = min(center_x + box_width // 2, image.shape[1] - 1)
    box_y_max = min(center_y + box_height // 2, image.shape[0] - 1)

    # Crop out the tiny box from the image
    cropped_image = image[box_y_min:box_y_max+1, box_x_min:box_x_max+1]

    return cropped_image

def extract_features(images):
    """
    Extracts features from images using color histograms.

    Args:
        images: A list of NumPy arrays representing the images.

    Returns:
        A matrix of features (one row per image).
    """
    features = []
    for image in images:
        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Compute color histogram
        hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Append colour histogram to list of features
        features.append(hist)

    return np.array(features)

def cluster_images(images, num_clusters=2):
    """
    Clusters a list of images (NumPy arrays) using k-means clustering and returns labels for each image.

    Args:
        images: A list of NumPy arrays representing the images.
        num_clusters: The number of clusters to form.

    Returns:
        A list of cluster labels for each image in the same order as the input list.
    """
    # Convert images to feature vectors
    features = extract_features(images)

    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(features, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert labels to list format
    labels = labels.flatten().tolist()

    return labels

def agglomerative_clustering(features, n_clusters=2, linkage='ward'):
    """
    Perform Agglomerative Clustering on the given features.

    Args:
        features (array-like): Input features for clustering.
        n_clusters (int): The number of clusters to form.
        linkage (str): The linkage criterion to use.

    Returns:
        array-like: Cluster labels for each sample.
    """
    # Initialize Agglomerative Clustering object
    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    # Perform clustering
    labels = ac.fit_predict(features)

    return labels

# calculates coordinates of possession marker
def calculate_marker(anchor: Point) -> np.ndarray:
    x, y = anchor.int_xy_tuple
    return(np.array([
        [x - MARKER_WIDTH // 2, y - MARKER_HEIGHT - MARKER_MARGIN],
        [x, y - MARKER_MARGIN],
        [x + MARKER_WIDTH // 2, y - MARKER_HEIGHT - MARKER_MARGIN]
    ]))


# draw single possession marker
def draw_marker(image: np.ndarray, anchor: Point, color: Color) -> np.ndarray:
    possession_marker_countour = calculate_marker(anchor=anchor)
    image = draw_filled_polygon(
        image=image,
        countour=possession_marker_countour,
        color=color)
    image = draw_polygon(
        image=image,
        countour=possession_marker_countour,
        color=MARKER_CONTOUR_COLOR,
        thickness=MARKER_CONTOUR_THICKNESS)
    return image


        
def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv2.VideoCapture(video_file)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        yield frame

    video.release()



# create cv2.VideoWriter object that we can use to save output video
def get_video_writer(target_video_path: str, video_config: VideoConfig) -> cv2.VideoWriter:
    video_target_dir = os.path.dirname(os.path.abspath(target_video_path))
    os.makedirs(video_target_dir, exist_ok=True)
    return cv2.VideoWriter(
        target_video_path,
        fourcc=cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G'),
        fps=video_config.fps,
        frameSize=(video_config.width, video_config.height),
        isColor=True
    )


def get_offside_decision(indicators, vertical_vanishing_point, attackingTeamId, defendingTeamId, isKeeperFound):
  # get last defending man
    currMinAngle = 360.0
    last_defending_man_coor = (0,0)
    for pose in indicators:
        if pose.team_id == defendingTeamId:
            if pose.offside not in ['on', 'off', 'def', 'ref']:
                if pose.angle < currMinAngle:
                    currMinAngle = pose.angle
                    last_defending_man_x = int(pose.rect.bottom_center.x)
                    last_defending_man_y = int(pose.rect.bottom_center.y)
                    last_defending_man_coor = (last_defending_man_x,last_defending_man_y)
            elif pose.angle < currMinAngle:
                currMinAngle = pose.angle
                last_defending_man_x = int(pose.rect.bottom_center.x)
                last_defending_man_y = int(pose.rect.bottom_center.y)
                last_defending_man_coor  = (last_defending_man_x,last_defending_man_y)
    exclude_last_man_coor = last_defending_man_coor
    currMinAngle = 360.0
    last_defending_man_coor = (0,0)
    for pose in indicators:
        current_man_x = int(pose.rect.bottom_center.x)
        current_man_y = int(pose.rect.bottom_center.y)
        current_man_coor = (current_man_x,current_man_y)
        if (pose.team_id == defendingTeamId) and current_man_coor != exclude_last_man_coor:
            if pose.offside not in ['on', 'off', 'def', 'ref']:
                if pose.angle < currMinAngle:
                    currMinAngle = pose.angle
                    last_defending_man_coor = current_man_coor
            elif pose.angle < currMinAngle:
                currMinAngle = pose.angle
                last_defending_man_coor = pose.current_man_coor
	# get decision for each detected player
    for pose in indicators:
		# attacking team
        if pose.team_id == attackingTeamId:
            if pose.offside not in ['on', 'off', 'def', 'ref']:
                if pose.angle < currMinAngle:
                    pose.offside = "off"
                else:
                    pose.offside = "on"
            else:
                if pose.angle < currMinAngle:
                    pose.offside = "off"
                else:
                    pose.offside = "on"
	    # defending team, append 'def' to maintain uniformity in data structure
        else:
            if pose.offside not in ['on', 'off', 'def', 'ref']:
                if pose.class_id == 3:
                    pose.offside="ref"
                else:
                    pose.offside="def"
            else:
                pose.offside = "def"

    return indicators, last_defending_man_coor



def get_leftmost_point_angles(vertical_vanishing_point, indicators, image, side):
    for indicator in indicators:
        x = int(indicator.rect.bottom_center.x)
        y = (indicator.rect.bottom_center.y)
        curr_angle = get_angle(vertical_vanishing_point, (x,y), image, side)
        indicator.angle = curr_angle
    return indicators

def get_angle(vanishing_point, test_point, img, goalDirection):
    reference_point = 0.0 , vanishing_point[1]
    a = np.array(reference_point)
    b = np.array(vanishing_point)
    c = np.array(test_point)
    #cv2.line(img , (int(a[0]),int(a[1])) , (int(b[0]),int(b[1])) , (0,0,255) , 2 )
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)
    if goalDirection == 'left':
        if reference_point[0] > vanishing_point[0]:
            angle = -1 * angle
    if goalDirection == 'right':
        if reference_point[0] < vanishing_point[0]:
            angle = -1 * angle

    return angle


def process_frames_vp(frames,side):
    vertical_vanishing_point = get_vertical_vanishing_point(frames, side)
    print("Vertical vanishing point:", vertical_vanishing_point)
    return vertical_vanishing_point


def det (a,b):
  return a[0]*b[1]-a[1]*b[0]

def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return [x, y]

def find_intersections(lines):
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if not line_1 == line_2:
                intersection = line_intersection(line_1, line_2)
                if intersection:
                    intersections.append(intersection)

    return intersections

def get_vertical_lines(img, side):
    linesFound = False
    BlueRedMask = 100
    blur_kernel_size = 3

    while not linesFound:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite("hsv.jpg", hsv)
        mask = cv2.inRange(hsv, (35, 100, 100), (70, 255, 255))
        imask = mask > 0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        green_blurred = cv2.GaussianBlur(green, (blur_kernel_size, blur_kernel_size), 0)
        edges = cv2.Canny(green_blurred, 150, 250, apertureSize=3)
        cv2.imwrite("green_edges_first.jpg", edges)
        green_blurred = cv2.GaussianBlur(green_blurred, (blur_kernel_size, blur_kernel_size), 0)
        green_blurred = cv2.GaussianBlur(green_blurred, (blur_kernel_size, blur_kernel_size), 0)
        green_blurred = cv2.GaussianBlur(green_blurred, (blur_kernel_size, blur_kernel_size), 0)
        green_blurred = cv2.GaussianBlur(green_blurred, (blur_kernel_size, blur_kernel_size), 0)
        green_blurred = cv2.GaussianBlur(green_blurred, (blur_kernel_size, blur_kernel_size), 0)
        green_blurred = cv2.GaussianBlur(green_blurred, (blur_kernel_size, blur_kernel_size), 0)
        green_blurred = cv2.GaussianBlur(green_blurred, (blur_kernel_size, blur_kernel_size), 0)
        green_blurred = cv2.GaussianBlur(green_blurred, (blur_kernel_size, blur_kernel_size), 0)
        green_blurred = cv2.GaussianBlur(green_blurred, (blur_kernel_size, blur_kernel_size), 0)
        green_blurred = cv2.GaussianBlur(green_blurred, (blur_kernel_size, blur_kernel_size), 0)
        green_blurred = cv2.GaussianBlur(green_blurred, (5, 5), 0)
        #edges = cv2.Canny(green_blurred, 150, 250, apertureSize=5)
        #cv2.imwrite("green_edges.jpg", edges)

        gray = cv2.cvtColor(green_blurred, cv2.COLOR_BGR2GRAY)
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        cv2.imwrite("green.jpg", green_blurred)

        edges = cv2.Canny(green_blurred, 150, 250, apertureSize=5)
        minLineLength = 100
        maxLineGap = 1250
        cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("green_edges_final_11x(3x3).jpg", edges)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)
        if lines.any():
          print(lines)
          if len(lines) > 5:
            BlueRedMask +=10
          if len(lines) >= 2 and len(lines < 5) :
            linesFound = True
          else:
            BlueRedMask -=10



    #print(lines)

    linesFound = False
    selectedLines = []
    selectedLinesParams = []




    if side == 'left':
      angleMaxLimit = 20
      angleMinLimit = 70
    else:
      angleMaxLimit = 150
      angleMinLimit = 105

    rLimit = 300
    while linesFound == False:
      for line in lines:
        for r,theta in line:
          isLineValid = True
          a = np.cos(theta)
          b = np.sin(theta)

          if (theta*180 *7/22)> angleMinLimit and (theta*180 *7/22) < angleMaxLimit:
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            #print("pass1")
            if len(selectedLines) > 0:
              for lineParams in selectedLinesParams:
                if abs(lineParams[0] -r) < rLimit:
                  isLineValid = False
              for selectedLine in selectedLines:
                if not line_intersection(selectedLine,[(x1,y1),(x2,y2)]):
                  isLineValid = False
              if [[x1,y1],[x2,y2]] in selectedLines or [[x2,y2],[x1,y1]]in selectedLines:
                isLineValid = False
            if isLineValid:
              selectedLines.append([(x1,y1),(x2,y2)])
              #print("---------------")
              #print(x1,y1,x2,y2)
              selectedLinesParams.append([r,theta])
              #cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 1)
              #cv2.putText(img, str((theta * 180 * 7/22)),(int((x2)),int((y2))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2,cv2.LINE_AA)
              #cv2.putText(img, str((theta * 180 * 7/22)),(int((x1)),int((y1))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2,cv2.LINE_AA)
      if len(selectedLines) < 2:
        if rLimit >=75:
          rLimit -= 10
          #print("pass2")
        else:
          angleMinLimit -=1
          angleMaxLimit +=1
          rLimit = 100
          #print("pass3")
      else:
        linesFound = True
    return selectedLines


def get_horizontal_vanishing_point(img):

    selectedLines = get_horizontal_lines(img)
    intersectionPoints = find_intersections(selectedLines)
    vanishingPointX = 0.0
    vanishingPointY = 0.0
    for point in intersectionPoints:
        vanishingPointX += point[0]
        vanishingPointY += point[1]

    return (vanishingPointX/len(intersectionPoints), vanishingPointY/len(intersectionPoints))

def get_horizontal_lines(image):
    img = image
    selectedLines = []
    selectedLinesParams = []
    linesFound = False
    BlueRedMask = 100

    while linesFound == False:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (36, BlueRedMask, BlueRedMask), (100, 255,255))
        imask = mask>0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,150,250,apertureSize = 3)
        minLineLength = 1
        maxLineGap = 200
        lines = cv2.HoughLines(edges,1,np.pi/180, 150)
        if lines.any():
            if len(lines) > 2:
                linesFound = True
            else:
                BlueRedMask -= 10

    linesFound = False
    angleMaxLimit = 120
    angleMinLimit = 0
    # angleCosLimit = 0.5
    rLimit = 100
    while linesFound == False:
        for line in lines:
            for r,theta in line:
                isLineValid = True
                a = np.cos(theta)
                b = np.sin(theta)
                if (theta * 180 * 7 / 22) > angleMinLimit and (theta * 180 * 7 / 22) < angleMaxLimit:
                    x0 = a*r
                    y0 = b*r
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    if len(selectedLines) > 0:
                        for lineParams in selectedLinesParams:
                            if abs(lineParams[0] - r) < rLimit:
                                isLineValid = False
                        for selectedLine in selectedLines:
                            if not line_intersection(selectedLine, [[x1,y1],[x2,y2]]):
                                isLineValid = False
                        if [[x1,y1],[x2,y2]] in selectedLines or [[x2,y2],[x1,y1]] in selectedLines:
                            isLineValid = False
                    if isLineValid:
                        selectedLines.append([[x1,y1],[x2,y2]])
                        selectedLinesParams.append([r, theta])
                        cv2.line(image,(x1,y1), (x2,y2), (0,0,255),1)
        if len(selectedLines) < 2:
          if rLimit >= 75:
            rLimit -= 10
          else:
            angleMinLimit -= 1
            angleMaxLimit += 1
            # angleCosLimit -= 0.05
        else:
          linesFound = True

    return selectedLines


def get_vertical_vanishing_point(img , side):

    selectedLines = get_vertical_lines(img , side)
    intersectionPoints = find_intersections(selectedLines)
    vanishingPointX = 0.0
    vanishingPointY = 0.0
    for point in intersectionPoints:
        vanishingPointX += point[0]
        vanishingPointY += point[1]
    # print(selectedLines)

    return (vanishingPointX/len(intersectionPoints), vanishingPointY/len(intersectionPoints))

def filter_detections_by_class(detections: List[Detection], class_name: str) -> List[Detection]:
    return [detection for detection in detections if detection.class_name == class_name]

# resolves which player is currently in ball possession based on player-ball proximity
def get_player_in_possession(
    player_detections: List[Detection],
    ball_detections: List[Detection],
    proximity: int
) -> Optional[Detection]:
    if len(ball_detections) != 1:
        return None
    ball_detection = ball_detections[0]
    for player_detection in player_detections:
        if player_detection.rect.pad(proximity).contains_point(point=ball_detection.rect.center):
            return player_detection

def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness)
    return image


def draw_filled_rect(image: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, -1)
    return image


def draw_polygon(image: np.ndarray, countour: np.ndarray, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, thickness)
    return image


def draw_filled_polygon(image: np.ndarray, countour: np.ndarray, color: Color) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, -1)
    return image


def draw_text(image: np.ndarray, anchor: Point, text: str, color: Color, thickness: int = 2) -> np.ndarray:
  cv2.putText(image, text, anchor.int_xy_tuple, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color.bgr_tuple, thickness, 2, False)
  return image


def draw_ellipse(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2) -> np.ndarray:
    cv2.ellipse(
        image,
        center=rect.bottom_center.int_xy_tuple,
        axes=(int(rect.width), int(0.35 * rect.width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color.bgr_tuple,
        thickness=thickness,
        lineType=cv2.LINE_4
    )
    return image

def pad(self, padding: float) -> Optional[Rect]:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2*padding,
            height=self.height + 2*padding
        )

def contains_point(self, point: Point) -> bool:
        return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y

def prediction(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH, side):
    # white
    BALL_COLOR_HEX = "#FFFFFF"
    BALL_COLOR = Color.from_hex_string(BALL_COLOR_HEX)

    # red
    GOALKEEPER_COLOR_HEX = "#FFA500"
    GOALKEEPER_COLOR = Color.from_hex_string(GOALKEEPER_COLOR_HEX)

    # green
    PLAYER_COLOR_HEX = "#00D4BB"
    PLAYER_COLOR = Color.from_hex_string(PLAYER_COLOR_HEX)

    # yellow
    REFEREE_COLOR_HEX = "#FFFF00"
    REFEREE_COLOR = Color.from_hex_string(REFEREE_COLOR_HEX)

    COLORS = [
        BALL_COLOR,
        GOALKEEPER_COLOR,
        PLAYER_COLOR,
        REFEREE_COLOR
    ]
    THICKNESS = 4
    print(TARGET_VIDEO_PATH)
    TARGET_VIDEO_PATH = TARGET_VIDEO_PATH.replace('.mp4','.avi')
    print(TARGET_VIDEO_PATH)

    # distance in pixels from the player's bounding box where we consider the ball is in his possession
    PLAYER_IN_POSSESSION_PROXIMITY = 30

    #video = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    #width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Release the VideoCapture object
    #video.release()
    ##os.makedirs(output_dir, exist_ok=True)

    # initiate video writer
    video_config = VideoConfig(
        fps=30,
        width=1280,
        height=720)
    video_writer = get_video_writer(
        target_video_path=TARGET_VIDEO_PATH,
        video_config=video_config)
    # get fresh video frame generator
    frame_iterator = iter(generate_frames(video_file=SOURCE_VIDEO_PATH))
    # initiate annotators
    base_annotator = BaseAnnotator(
        colors=[
            BALL_COLOR,
            PLAYER_COLOR,
            PLAYER_COLOR,
            REFEREE_COLOR
        ],
        thickness=THICKNESS)
    
    player_goalkeeper_text_annotator = TextAnnotator(
    PLAYER_COLOR, text_color=Color(255, 255, 255), text_thickness=2)
    referee_text_annotator = TextAnnotator(
        REFEREE_COLOR, text_color=Color(0, 0, 0), text_thickness=2)

    ball_marker_annotator = MarkerAnntator(
        color=BALL_MARKER_FILL_COLOR)
    player_in_possession_marker_annotator = MarkerAnntator(
        color=PLAYER_MARKER_FILL_COLOR)
    
    player_model, ball_model = load_models()
    is_offside = False
    # loop over frames
    for frame in frame_iterator:
        if side == "left" :
            frame = frame
        else:
            frame = cv2.flip(frame, 1)
        #if not is_offside:
            #if frame_count % frames_per_function_call == 9:
                #vp = process_frames_vp(cropped_frame , side)
        width = int(frame.shape[1])  # Access width using shape[1]
        height = int(frame.shape[0]) 

        vp =(int(width*3/5), int(height*2))
        # run detector
        player_inference = player_model(frame, device='cpu')
        ball_inference = ball_model(frame, device='cpu')
        detections = Detection.from_player_results(player_inference)
        ball_detection = Detection.from_ball_results(ball_inference)
        if len(ball_detection) > 0:
            detections.append(ball_detection[0])


        ball_detections = filter_detections_by_class(detections=detections, class_name="ball")
        referee_detections = filter_detections_by_class(detections=detections, class_name="referee")
        goalkeeper_detections = filter_detections_by_class(detections=detections, class_name="goalkeeper")
        player_detections = filter_detections_by_class(detections=detections, class_name="player")

        player_goalkeeper_detections = player_detections 
        tracked_detections = player_detections + goalkeeper_detections + referee_detections

        # calculate player in possession
        player_in_possession_detection = get_player_in_possession(
            player_detections=player_goalkeeper_detections,
            ball_detections=ball_detections,
            proximity=PLAYER_IN_POSSESSION_PROXIMITY)
        
        #print(player_detections)
        pose_est = get_leftmost_point_angles(vp,player_detections,frame,"left")

        # Sort pose_est based on the angle attribute
        pose_est = sorted(pose_est, key=lambda x: x.angle)
        #print(pose_est[0].rect.bottom_center.x)
        #print(pose_est[0].rect.bottom_center.y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for pose in pose_est:
            x = int(pose.rect.bottom_center.x)
            y = int(pose.rect.bottom_center.y)
            #cv2.line(frame, (int(vp[0]) , int(vp[1])) , (int(x), int(y)), (0,255,0) , 2 )

        pose_est2, last_defending_man_coordinate = get_offside_decision(pose_est, vp, 1, 0, 0) # attacking and defending

        #print(last_defending_man_coordinate)
        for pose in pose_est2:
        #print(pose.team_id)
            center_x = int(pose.rect.bottom_center.x)
            center_y = int(pose.rect.bottom_center.y)
            top_left_x = int(pose.rect.top_left.x)
            top_left_y = int(pose.rect.top_left.y)



            if pose.team_id == 1:
                if pose.offside == 'off':
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, 'off', (top_left_x, top_left_y), font, 0.5, (0,0,255), 2, cv2.LINE_AA)
                    #cv2.line(frame , (int(vp[0]) , int(vp[1])) , (center_x, center_y), (0,255,0) , 2 )
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    #cv2.putText(cropped_frame, 'on', (top_left_x, top_left_y), font, 0.5, (0,255,0), 2, cv2.LINE_AA)
            if pose.team_id == 0:
                current_defending_man_x = pose.rect.bottom_center.x
                current_defending_man_y = pose.rect.bottom_center.y
                current_defending_man_coor = (current_defending_man_x, current_defending_man_y)
                #if current_defending_man_coor == last_defending_man_coordinate:
                    #cv2.putText(cropped_frame, 'last man', (top_left_x,top_left_y), font, 0.5, (200,255,155), 2, cv2.LINE_AA)
                    #cv2.line(frame , (int(vp[0]) , int(vp[1])) , (center_x, center_y), (0,255,0) , 2 )
                #else:
                    #cv2.putText(cropped_frame, 'def', (top_left_x, top_left_y), font, 0.5, (200,255,155), 2, cv2.LINE_AA)
                    #cv2.line(frame , (int(vp[0]) , int(vp[1])) , (center_x, center_y), (0,255,0) , 2 )
            
                #print(player_in_possession_detection)
            if player_in_possession_detection is not None:
                if pose.offside == 'off':
                    x_p = int(pose.rect.bottom_center.x)
                    y_p = int(pose.rect.bottom_center.y)
                    off_player_coordinate = (x_p,y_p)
                    x = int(player_in_possession_detection.rect.bottom_center.x)
                    y = int(player_in_possession_detection.rect.bottom_center.y)
                    player_in_possession_detection_coordinate = (x,y)
                    if off_player_coordinate == player_in_possession_detection_coordinate:
                        is_offside = True
        if is_offside:
            cv2.putText(frame, 'Offside Detected', (20,60), font, 2, (0,0,255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Offside Not Detected', (20,60), font, 2, (0,255,0), 2, cv2.LINE_AA)


        tracked_referee_detections = filter_detections_by_class(detections=tracked_detections, class_name="referee")
        tracked_goalkeeper_detections = filter_detections_by_class(detections=tracked_detections, class_name="goalkeeper")
        tracked_player_detections = filter_detections_by_class(detections=tracked_detections, class_name="player")

        # initiate annotators
        ball_marker_annotator = MarkerAnntator(color=BALL_MARKER_FILL_COLOR)
        player_marker_annotator = MarkerAnntator(color=PLAYER_MARKER_FILL_COLOR)
        # # annotate video frame
        annotated_image = frame.copy()
        annotated_image = base_annotator.annotate(
            image=annotated_image,
            detections=detections)

        annotated_image = player_goalkeeper_text_annotator.annotate(
            image=annotated_image,
            detections=tracked_goalkeeper_detections + tracked_player_detections)
        annotated_image = referee_text_annotator.annotate(
            image=annotated_image,
            detections=tracked_referee_detections)

        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image,
            detections=ball_detections)
        annotated_image = player_marker_annotator.annotate(
            image=annotated_image,
            detections=[player_in_possession_detection] if player_in_possession_detection else [])

        annotated_image = cv2.resize(annotated_image,(1280,720))
        # save video frame
        video_writer.write(annotated_image)
    print(f"Video written to: {TARGET_VIDEO_PATH}")
    print('Inference is completed')
    # close output video
    video_writer.release()
    return True
