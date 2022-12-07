from Data import *
from Functions import *

# Neural Network
nr_correct = 0 # Number of correct classifications
for epoch in range(max_epochs): 
    for img, l in zip(train_images, labels_array):
        img.shape += (1,) # Reshape image array from (1, 64) to (64, 1) column matrix
        l.shape += (1,) # Reshape label array from (1, 64) to (64, 1) column matrix

        # Forward propagation ---------------------------
        # Forward propagation input -> hidden
        hidden_pre = b_ih + w_ih @ img 
        hidden = sigmoid(hidden_pre) # hidden layer array
        # Forward propagation hidden -> output
        out_pre = b_ho + w_ho @ hidden
        output = sigmoid(out_pre) # output layer array
        
        # Cost / Error ----------------------------------
        err = error(output, l) # Error 
        nr_correct += int(np.argmax(output) == np.argmax(l)) # +1 to nr_correct when output label matches real label

        # Backpropagation -------------------------------
        # Backpropagation hidden <- output (Cost function derivative)
        delta_out = output - l # Derivative of the cost function, with some mathematical manipulation
        w_ho += -learning_rate * delta_out @ np.transpose(hidden) # Update weights connecting hidden and output layers
        b_ho += -learning_rate * delta_out  # Update bias added to output

        # Backpropagation input <- hidden  (Activation function derivative)
        delta_hidden = np.transpose(w_ho) @ delta_out * d_sigmoid(hidden) # Gradients of weights hidden to input
        w_ih += -learning_rate * delta_hidden @ np.transpose(img) # Update weights connecting input to hidden
        b_ih += -learning_rate * delta_hidden  # Update bias added to hidden

    print("Accuracy", (nr_correct/train_size) * 100)  # Calculate accuracy
    nr_correct = 0 # Reset the number of label matches 

# Test model
while True:
    index = int(input("Enter a number 0-764: "))
    img = test_images[index] # 765 images
    plt.imshow(img.reshape(8,8), cmap="gray")

    # Forward propagate to get a label from the model
    img.shape += (1,)
    # Forward propagation input -> hidden
    hidden_pre = b_ih + w_ih @ img.reshape(64,1)
    hidden = sigmoid(hidden_pre) # hidden layer array
    # Forward propagation hidden -> output
    out_pre = b_ho + w_ho @ hidden
    output = sigmoid(out_pre) # output layer array

    plt.title(f"Label {np.argmax(output)}")
    plt.show()