import numpy as np
import cv2
from sklearn.neighbors import KDTree
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot   as plt

# Function to process a single pixel and update its position based on the Mean Shift algorithm
def process_pixel(point, sampled_pixels, tree, max_iter, epsilon, spatial_bandwidth, range_bandwidth):
    shift = np.array([1, 1, 1]) # Initialize the shift vector to a non-zero value for the first iteration, ensuring that we enter the loop
    
    for _ in range(max_iter): # Loop to iteratively adjust the point until convergence or reaching maximum iterations
        
        # Find the indices of neighboring pixels using KDTree for efficient range queries
        # tree.query_radius searches for all the neighbors of the point within the specified radius (spatial_bandwidth)
        indices = tree.query_radius([point], r=spatial_bandwidth)[0]
        
        # If there are no neighbors within the radius, break the loop
        if len(indices) == 0:
            break


        # Compute spatial (location) distance between the current pixel and its neighbors (in the x and y directions)
        # This is calculated by finding the Euclidean distance between the pixel's (x, y) coordinates and its neighbors
        distances_spatial = np.linalg.norm(sampled_pixels[indices, :2] - point[:2], axis=1)
        
        # Apply a Gaussian kernel to the spatial distances to calculate spatial weights
        # The smaller the distance, the larger the weight, making closer pixels more important
        spatial_weights = np.exp(- (distances_spatial**2) / (2 * spatial_bandwidth**2))
        

        # Compute range (color) distance between the current pixel and its neighbors (in RGB space)
        # This is calculated by finding the Euclidean distance between the RGB color values of the current pixel and its neighbors
        distances_range = np.linalg.norm(sampled_pixels[indices, 2:] - point[2:], axis=1)

        # Apply a Gaussian kernel to the range distances to calculate color weights
        # Pixels with similar colors will have a higher weight
        range_weights = np.exp(- (distances_range**2) / (2 * range_bandwidth**2))
        
        # Combine spatial and range weights to get a total weight for each neighbor
        combined_weights = spatial_weights * range_weights

         # Sum of the combined weights (normalization factor)
        weight_sum = np.sum(combined_weights)
        
        # If all weights are zero (meaning no valid neighbors), break the loop
        if weight_sum == 0:
            break
        
        # Update the pixel's position by computing the weighted average of its neighbors' positions
        # The new position is computed by applying the weights to the neighbors' positions (coordinates and color values)
        new_point = np.sum(combined_weights[:, np.newaxis] * sampled_pixels[indices], axis=0) / weight_sum

         # Calculate the shift (movement) in the pixel's position
        shift = np.linalg.norm(new_point - point)

         # Update the point's position
        point = new_point
        
         # If the shift in position is smaller than the threshold epsilon, the algorithm has converged
        if shift < epsilon:
            break
    
    # Return the final mode (cluster center) for the pixel after convergence
    return point

# Mean Shift algorithm implementation
def mean_shift(image, max_iter=100, epsilon=1e-5, spatial_bandwidth=10, range_bandwidth=15, subsample_fraction=0.01, num_threads=4):
    # Get the height, width, and channels of the input image
    height, width, _ = image.shape

    # Flatten the image into a 2D array, where each row represents a pixel with its RGB color values
    pixels = image.reshape((-1, 3))# Now pixels is an array of shape (height * width, 3)
    
    # Create coordinates for each pixel, giving the (y, x) positions in the image
    coords = np.indices((height, width)).reshape(2, -1).T  # This returns an array of shape (height * width, 2), where each row is (y, x)
    
    # Combine the pixel coordinates (y, x) and their corresponding RGB values into one array
    # This allows us to process both spatial and color information together for each pixel
    pixels_with_coords = np.hstack((coords, pixels)) # Shape becomes (height * width, 5) (y, x, R, G, B)
    

    # Randomly sample a fraction of the pixels from the image to speed up the algorithm
    num_pixels = pixels_with_coords.shape[0]

    # Select indices for sampled pixels (subsampling), we use a fraction of the total pixels to reduce computation time
    sampled_indices = np.random.choice(num_pixels, size=int(num_pixels * subsample_fraction), replace=False)

    # Extract the sampled pixels (the selected subset of the image's pixels and their coordinates)
    sampled_pixels = pixels_with_coords[sampled_indices]
    
    # Create a KDTree for fast nearest-neighbor search
    # KDTree is an efficient data structure for both spatial and color distances search
    tree = KDTree(sampled_pixels)
    
    # Use parallel processing (ThreadPoolExecutor) to process each sampled pixel concurrently
    # This reduces the overall runtime by taking advantage of multiple CPU cores
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
          # Use the executor to process all sampled pixels in parallel
        modes = list(executor.map(
            lambda i: process_pixel(sampled_pixels[i], sampled_pixels, tree, max_iter, epsilon, spatial_bandwidth, range_bandwidth),
            range(len(sampled_pixels))
        ))
    
     # Convert the list of modes (the final positions of the pixels after Mean Shift) into a numpy array
    modes = np.array(modes)
    
    # Create a new KDTree for the computed modes
    # This KDTree will be used to find the closest mode for each pixel in the original image
    tree_modes = KDTree(modes)

    # For each pixel in the original image, find the closest mode (cluster center)
    # The result is an array of labels where each label corresponds to the nearest mode for that pixel
    _, labels = tree_modes.query(pixels_with_coords, k=1)  # نزدیک‌ترین mode برای هر پیکسل
    
    
    # Reconstruct the segmented image by assigning the color of the nearest mode to each pixel
    # For each pixel, we replace its original color with the color of its closest mode (based on the labels)
    segmented_image = modes[labels.flatten(), 2:].astype(np.uint8) # Extract the RGB values from the modes
    segmented_image = segmented_image.reshape((height, width, 3)) # Reshape the segmented image to the original dimensions
    
    return segmented_image, modes, labels


image = cv2.imread("image.jpg")

# Run the Mean Shift algorithm on the loaded image
# Here, we specify parameters like spatial bandwidth, range bandwidth, subsample fraction, and number of threads for parallelism
segmented_image, modes, labels = mean_shift(image, spatial_bandwidth=10, range_bandwidth=15, subsample_fraction=0.001, num_threads=8)

# نمایش تصویر
cv2.imshow("Segmented Image", segmented_image)
cv2.imwrite("Final_Segmented_image.jpg" ,segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def plot_clusters(modes, labels, image):
    """
    Visualizes the clusters by plotting the segmented regions over the original image.
    Args:
        modes: Array of cluster centers (modes) calculated by the mean shift algorithm.
        labels: Array of cluster labels assigned to each pixel.
        image: The original input image.
    """
    height, width, _ = image.shape

    # Convert labels to match the height and width of the original image
    labels_reshaped = labels.reshape(height, width)

    # Find unique labels (clusters)
    unique_labels = np.unique(labels)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title("Cluster Visualization")
    plt.imshow(image[:, :, ::-1])  # Convert from BGR to RGB for matplotlib visualization

    for label in unique_labels:
        # Create a binary mask for the current cluster
        cluster_mask = labels_reshaped == label

        # Generate a random color for each cluster
        cluster_color = np.random.rand(3)

        # Overlay the contour of the cluster on the image
        ax.contour(cluster_mask, colors=[cluster_color], linewidths=1)

    plt.show()

plot_clusters(modes, labels, image)
