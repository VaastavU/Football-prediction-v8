import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
# import cv2

def main():
   # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # camera movement calculation

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    
    # interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(ball_positions=tracks['ball'])

    # assign team and players with respective colors
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # save cropped image
    # after saving the img no need to run the code again!
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     cropped_img = frame[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]

    #     cv2.imwrite("output_videos/cropped_img.jpg", cropped_img)
    #     break

    # Assign ball Acquisition to player:
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    
    team_ball_control = np.array(team_ball_control)
    

    # Draw output frames

    ## draw player and ball annotations
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## draw camera movement annotations
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)


    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
