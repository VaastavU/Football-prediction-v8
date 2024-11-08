import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame):
        # Set minimum distance threshold for camera movement detection
        self.minimum_distance = 5

        # Parameters for Lucas-Kanade optical flow calculation
        self.lk_params = dict(
            winSize=(15, 15),  # Window size for optical flow search
            maxLevel=2,        # Max pyramid level for iterative search
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Convert the first frame to grayscale for easier processing
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mask used to select specific regions of interest for feature tracking
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1   # Select a narrow band on the left side of the frame
        mask_features[:, 900:1050] = 1  # Select a narrow band on the right side of the frame

        # Parameters for detecting good features to track (corners)
        self.features = dict(
            maxCorners=100,           # Maximum number of corners to detect
            qualityLevel=0.3,         # Minimum quality level of corners
            minDistance=3,            # Minimum distance between corners
            blockSize=7,              # Size of the block for corner detection
            mask=mask_features        # Mask specifying regions for feature tracking
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        # Adjust positions in object tracks based on detected camera movement
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    # Original position of the tracked object
                    position = track_info['position']
                    # Camera movement offset for the current frame
                    camera_movement = camera_movement_per_frame[frame_num]
                    # Adjust position based on camera movement
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    # Update the track with adjusted position
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Load camera movement data from a stub file if specified
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Initialize a list to store camera movement for each frame
        camera_movement = [[0, 0]] * len(frames)

        # Convert the first frame to grayscale and detect initial features
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Iterate over frames to calculate camera movement
        for frame_num in range(1, len(frames)):
            # Convert current frame to grayscale
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            # Calculate optical flow to get new feature positions
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # Calculate movement vector for each feature point
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # Calculate Euclidean distance between old and new points
                distance = measure_distance(new_features_point, old_features_point)
                if distance > max_distance:
                    max_distance = distance
                    # Calculate x and y components of movement
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            # Update camera movement if it exceeds the minimum distance threshold
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Re-detect features on the current frame
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # Set the current frame as the old frame for the next iteration
            old_gray = frame_gray.copy()

        # Save camera movement data to a stub file if specified
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        # Create an output list for frames with camera movement annotations
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()  # Make a copy of the frame to avoid modifying the original

            # Draw a semi-transparent rectangle at the top for displaying movement information
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Get camera movement values for x and y axes
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            # Display x and y movement values as text on the frame
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            # Append the annotated frame to the output list
            output_frames.append(frame)

        return output_frames
