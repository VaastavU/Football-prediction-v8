import cv2
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance, get_foot_position

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def  add_speed_and_distance_to_tracks(self, tracks):

        total_distance = {}

        for object, object_track in tracks.items():
            if object == "ball" or object == "referees":
                continue
        
            number_of_frames = len(object_track)
            for frame_num in range(0,number_of_frames,self.frame_window):
                last_frame = min(frame_num+self.frame_window, number_of_frames-1)

                for track_id, _ in object_track[frame_num].items():
                    if track_id not in object_track[last_frame]:
                        continue

                    start_position = object_track[frame_num][track_id]['transformed_position']
                    final_position = object_track[last_frame][track_id]['transformed_position']

                    if start_position is None or final_position is None:
                        continue

                    distance_covered = measure_distance(start_position, final_position)
                    time_elapsed = (last_frame - frame_num)/(self.frame_rate)
                    speed_meters_per_sec = distance_covered/time_elapsed
                    speed_kmph = speed_meters_per_sec * (3.6)

                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_kmph
                        tracks[object][frame_num_batch][track_id]['distance_covered'] = total_distance[object][track_id]


    def draw_speed_and_distance_annotations(self, frames, tracks):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == 'ball' or object == 'referees':
                    continue
                for track_id, track_info in object_tracks[frame_num].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed')
                        distance = track_info.get('distance_covered')  # Ensure this key matches the one used in the previous method

                        # Correct condition
                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40
                        position = tuple(map(int, position))

                        cv2.putText(
                            img=frame, 
                            text=f"speed: {speed:.2f} kmph", 
                            org=position, 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.5, 
                            color=(0, 0, 0), 
                            thickness=2
                        )
                        cv2.putText(
                            img=frame, 
                            text=f"distance: {distance:.2f} m", 
                            org=(position[0], position[1] + 20), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=0.5, 
                            color=(0, 0, 0), 
                            thickness=2
                        )
            output_frames.append(frame)

        return output_frames



