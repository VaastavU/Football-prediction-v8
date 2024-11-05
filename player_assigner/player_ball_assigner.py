import sys
sys.path.append("../")
from utils import get_bbox_center, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_ball_player_distance = 70

    def assign_ball_to_player(self, player, ball_bbox):

        ball_position = get_bbox_center(ball_bbox)

        min_distance = sys.maxsize
        player_assigned = -1

        for player_id, player in player.items():
            player_bbox = player["bbox"]

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_ball_player_distance:
                if distance < min_distance:
                    min_distance = distance
                    player_assigned = player_id

        return player_assigned

