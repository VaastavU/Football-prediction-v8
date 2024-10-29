from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video('input_videos/08fd33_4.mp4')
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracker(video_frames,read_from_stub=True,stub_path="stubs/tracks_stubs.pkl")
    save_video('output_videos/output_video.avi',video_frames)

if __name__ == '__main__':
    main()
