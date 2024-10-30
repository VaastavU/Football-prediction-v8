from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        # Initialize dictionaries to store team colors and player-team mappings
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        """
        Reshapes the image to a 2D array and applies k-means clustering to group pixels into two clusters.
        
        Parameters:
            image (ndarray): The input image (part of the frame) where clustering is performed.
        
        Returns:
            kmeans (KMeans): The trained KMeans clustering model.
        """
        # Flatten the image into a 2D array where each row represents a pixel (in RGB format)
        image_2d = image.reshape(-1, 3)

        # Perform K-means clustering with 2 clusters (representing two primary colors/teams)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extracts the primary color of a player within a bounding box region.
        
        Parameters:
            frame (ndarray): The entire video frame.
            bbox (tuple): Bounding box (x_min, y_min, x_max, y_max) around the player.
        
        Returns:
            player_color (ndarray): RGB color representing the player.
        """
        # Extract the region of interest (ROI) around the player in the frame using bounding box coordinates
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Take the top half of the player image as it might contain their primary color (jersey)
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get the clustering model for the top half of the image
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel in the top half
        labels = kmeans.labels_

        # Reshape the labels to match the dimensions of the top_half_image
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Find the cluster that represents the background (non-player cluster)
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        
        # Assume the other cluster represents the player
        player_cluster = 1 - non_player_cluster

        # Extract the RGB color of the player’s cluster from the k-means cluster centers
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Assigns team colors based on clustering the primary colors of all detected players.
        
        Parameters:
            frame (ndarray): The entire video frame.
            player_detections (dict): Dictionary with player detections where each entry has a 'bbox' for a player.
        """
        player_colors = []
        
        # Get color for each detected player
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Perform k-means clustering on the list of player colors to assign two team colors
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        # Store the k-means model for team prediction and assign team colors based on cluster centers
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determines the team of a player based on their color and assigns the team ID.
        
        Parameters:
            frame (ndarray): The entire video frame.
            player_bbox (tuple): Bounding box (x_min, y_min, x_max, y_max) around the player.
            player_id (int): Unique identifier for the player.
        
        Returns:
            team_id (int): ID of the team the player is assigned to (1 or 2).
        """
        # Check if player's team has already been determined
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Get the player's primary color based on the bounding box
        player_color = self.get_player_color(frame, player_bbox)

        # Predict team ID using k-means model
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1  # Adjust index for team ID

        # Optional override for a specific player ID (example logic)
        if player_id == 91:
            team_id = 1  # Manually assign team ID 1 for player with ID 91

        # Store player's team ID in the dictionary for future reference
        self.player_team_dict[player_id] = team_id

        return team_id
