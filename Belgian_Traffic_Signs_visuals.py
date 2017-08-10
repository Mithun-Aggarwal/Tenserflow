# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 20:19:29 2017

@author: ksxl806
"""

# Import the `pyplot` module
import matplotlib.pyplot as plt 

# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)

# Show the plot
plt.show()

# Determine the (random) indexes of the images that you want to see 
traffic_signs = [300, 2250, 1000, 1500]

# Fill out the subplots with the random images that you defined 
# =============================================================================
# for i in range(len(traffic_signs)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images[traffic_signs[i]])
#     plt.subplots_adjust(wspace=0.5)
# 
# plt.show()
# =============================================================================

# Fill out the subplots with the random images and add shape, min and max values
# =============================================================================
# for i in range(len(traffic_signs)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images[traffic_signs[i]])
#     plt.subplots_adjust(wspace=0.5)
#     plt.show()
#     print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
#                                                   images[traffic_signs[i]].min(), 
#                                                   images[traffic_signs[i]].max()))
# =============================================================================

# Get the unique labels 
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot 
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image 
    plt.imshow(image)
    
# Show the plot
plt.show()
