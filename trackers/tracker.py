from ultralytics import YOLO
import supervision as sv
import numpy as np
import pickle   # used to create a byte stream of comples DS to reuse later without computing the whole thing again
                # only used by python devs
import os

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        detections = []
        batch_size = 20

        for batch in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[batch:batch+batch_size], conf=0.1)
            detections += detection_batch
            break
        return detections

    def get_object_tracker(self, frames, read_from_stub=False, stub_path=None):
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        # Load from stub if enabled and file exists
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            try:
                # Check if the file is not empty
                if os.path.getsize(stub_path) > 0:
                    with open(stub_path, 'rb') as f:
                        tracks = pickle.load(f)
                    print("Loaded tracks from stub file.")
                    return tracks
                else:
                    print(f"Stub file {stub_path} is empty. Proceeding with fresh tracking.")
            except EOFError:
                print("EOFError: Stub file is corrupted or incomplete. Proceeding with fresh tracking.")
            except Exception as e:
                print(f"An error occurred while loading from stub: {e}. Proceeding with fresh tracking.")

        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {v: k for k, v in class_names.items()}

            if hasattr(detection, 'boxes'):
                detection_supervision = sv.Detections.from_ultralytics(detection)
            else:
                print(f"No 'boxes' attribute found in detection at frame {frame_num}. Skipping.")
                continue

            for obj_ind in range(len(detection_supervision.class_id)):
                class_id = int(detection_supervision.class_id[obj_ind])
                if class_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_ind] = class_names_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            print(detection_with_tracks)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == class_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == class_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist
                cls_id = frame_detection[3]

                if cls_id == class_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                print(f"Tracks saved to stub file: {stub_path}")

        return tracks
# Example usage
# tracker = Tracker('your_model_path')
# tracker.get_object_tracker(your_frames)
