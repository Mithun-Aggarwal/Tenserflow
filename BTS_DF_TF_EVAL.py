# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 01:25:30 2017

@author: ksxl806
"""

# =============================================================================
# # Pick 10 random images
# sample_indexes = random.sample(range(len(images32)), 10)
# sample_images = [images32[i] for i in sample_indexes]
# sample_labels = [labels[i] for i in sample_indexes]
# 
# # Run the "predicted_labels" op.
# predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
#                         
# # Print the real and predicted labels
# print(sample_labels)
# print(predicted)
# =============================================================================


# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])

plt.show()


# Load the test data
test_images, test_labels = load_data(test_data_dir)

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
from skimage.color import rgb2gray
test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))


sess.close()