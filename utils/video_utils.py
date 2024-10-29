import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: unable to open the file at path: {video_path}")
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Total frames: {len(frames)}")
    return frames


def save_video(output_video_path, output_video_frames):
    if len(output_video_frames) == 0:
        print("Invalid video, no existing frames\n")
        return
    
    # Use VideoWriter_fourcc to create the fourcc code correctly
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, 
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    for frame in output_video_frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")
