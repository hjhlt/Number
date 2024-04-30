import pickle

import mnist
import numpy as np
import scipy.special


def sigmoid_derivative(x):
    return x * (1 - x)


class Conv3x3:
    # A convolution layer using 3x3 filters.

    def __init__(self, num_filters, is_first):
        self.num_filters = num_filters
        self.is_first = is_first

        # filters is a 3d array with dimentions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        # image: matrix of image
        x, h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[:, i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        # input = input[np.newaxis, :, :]
        # 28x28
        self.last_input = input

        # input_im: matrix of image
        x, h, w = input.shape
        output = np.zeros((self.num_filters, h - 2, w - 2))
        for im_region, i, j in self.iterate_regions(input):
            for l in range(self.num_filters):
                for k in range(x):
                    output[l, i, j] += np.sum(im_region[k] * self.filters[l], axis=(0, 1))
        return output

    def backprop(self, d_L_d_out, learn_rate):
        x, h, w = self.last_input.shape
        # d_L_d_out: the loss gradient for this layer's outputs
        # learn_rate: a float
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                # d_L_d_filters[f]: 3x3 matrix
                # d_L_d_out[i, j, f]: num
                # im_region: 3x3 matrix in image
                d_L_d_filters[f] += np.sum(d_L_d_out[f, i, j] * im_region, axis=0)

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        # 计算对输入的梯度
        if self.is_first:

            d_L_d_input = np.zeros(self.last_input.shape)

            # 为了得到输入的梯度，我们需要将每个滤波器的转置与当前层的误差图进行卷积
            # 注意：在numpy中，可以通过切片和反转来实现转置
            filters_transposed = self.filters[:, ::-1, ::-1]  # 转置滤波器
            d_L_d_out = np.pad(d_L_d_out, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)

            for i in range(d_L_d_input.shape[1]):
                for j in range(d_L_d_input.shape[2]):
                    # 计算当前位置与所有滤波器的卷积结果
                    for l in range(x):
                        d_L_d_input_region = 0
                        for f in range(self.num_filters):
                            d_L_d_input_region += np.sum(filters_transposed[f] * d_L_d_out[f, i:i + 3, j:j + 3])
                        d_L_d_input[l, i, j] += d_L_d_input_region

                        # 将卷积结果放到输入梯度图的正确位置

                    # 返回对输入的梯度
            return d_L_d_input
        return None


class MaxPool2:
    # A Max Pooling layer using a pool size of 2.

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        # image: 3d matix of conv layer

        _, h, w = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[:, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        # 26x26x8
        self.last_input = input

        # input: 3d matrix of conv layer
        num_filters, h, w = input.shape
        output = np.zeros((num_filters, h // 2, w // 2))

        for im_region, i, j in self.iterate_regions(input):
            for k in range(num_filters):
                output[k, i, j] = np.amax(im_region[k], axis=(0, 1))

        return output

    def backprop(self, d_L_d_out):
        # d_L_d_out: the loss gradient for the layer's outputs

        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            f, h, w = im_region.shape
            amax = np.amax(im_region, axis=(1, 2))
            for f2 in range(f):
                for i2 in range(h):
                    for j2 in range(w):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[f2, i2, j2] == amax[f2]:
                            d_L_d_input[f2, i * 2 + i2, j * 2 + j2] = d_L_d_out[f2, i, j]

        return d_L_d_input


class Hidden:

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        # input_len: length of input nodes
        # nodes: lenght of ouput nodes

        self.weights = np.random.randn(input_len, nodes) / input_len

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        # 3d
        self.last_input_shape = input.shape

        # 3d to 1d
        input = input.flatten()

        # 1d vector after flatting
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights)

        # output before softmax
        # 1d vector
        self.last_totals = totals

        self.result = scipy.special.expit(totals)
        return self.result

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_t = d_L_d_out * sigmoid_derivative(self.result)

        d_L_d_w = self.last_input[np.newaxis].T @ d_L_d_t[np.newaxis]

        d_L_d_inputs = self.weights @ d_L_d_t

        # Update weights / biases
        self.weights -= learn_rate * d_L_d_w

        # it will be used in previous pooling layer
        # reshape into that matrix
        return d_L_d_inputs.reshape(self.last_input_shape)


class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        # input_len: length of input nodes
        # nodes: lenght of ouput nodes

        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''

        # 1d vector after flatting
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases

        # output before softmax
        # 1d vector
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        # only 1 element of d_L_d_out is nonzero
        for i, gradient in enumerate(d_L_d_out):
            # k != c, gradient = 0
            # k == c, gradient = 1
            # try to find i when k == c
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            # all gradients are given value with k != c
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            # change the value of k == c
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of out[i] against totals
            # gradients to every weight in every node
            # this is not the final results
            d_t_d_w = self.last_input  # vector
            d_t_d_b = 1
            # 1000 x 10
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            # d_L_d_t, d_out_d_t, vector, 10 elements
            d_L_d_t = gradient * d_out_d_t

            # Gradients of loss against weights/biases/input
            # (1000, 1) @ (1, 10) to (1000, 10)
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            # (1000, 10) @ (10, 1)
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # Update weights / biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            # it will be used in previous pooling layer
            # reshape into that matrix
            return d_L_d_inputs


# # We only use the first 1k testing examples (out of 10k total)
# # in the interest of time. Feel free to change this if you want.
# test_images = mnist.test_images()
# test_labels = mnist.test_labels()

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

conv = Conv3x3(2, 0)  # 28x28x1 -> 26x26x4
conv1 = Conv3x3(2, 1)  # 26x26x4 -> 24x24x4
pool1 = MaxPool2()  # 24x24x4 -> 12x12x4
# conv2 = Conv3x31(1)  # 24x24x1 -> 22x22x1
# pool2 = MaxPool2()  # 22x22x2 -> 11x11x2
hidden1 = Hidden(12 * 12 * 2, 20)  # 5x5x2 -> 20
# hidden2 = Hidden(50, 20)  # 50 -> 20
softmax = Softmax(20, 10)  # 20 -> 10


# with open('conv.t.pkl', 'rb') as f:
#     conv = pickle.load(f)
#
# with open('pool.t.pkl', 'rb') as f:
#     pool = pickle.load(f)
#
# with open('hidden.t.pkl', 'rb') as f:
#     hidden = pickle.load(f)
#
# with open('softmax.t.pkl', 'rb') as f:
#     softmax = pickle.load(f)


def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    out = conv.forward((image / 255) - 0.5)
    out = conv1.forward(out)
    out = pool1.forward(out)
    # out = conv2.forward(out)
    # out = pool2.forward(out)
    out = hidden1.forward(out)
    # out = hidden2.forward(out)
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

    # out: vertor of probability
    # loss: num
    # acc: 1 or 0


def train(im, label, lr=0.003):
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate intial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    # gradient = hidden2.backprop(gradient, lr)
    gradient = hidden1.backprop(gradient, lr)
    # gradient = pool2.backprop(gradient)
    # gradient = conv2.backprop(gradient, lr)
    gradient = pool1.backprop(gradient)
    gradient = conv1.backprop(gradient, lr)
    gradient = conv.backprop(gradient, lr)

    return loss, acc


print('MNIST CNN initialized!')

# Test the CNN
# print('\n--- Testing the CNN ---')
# loss = 0
# num_correct = 0
# for im, label in zip(test_images, test_labels):
#     _, l, acc = forward(im, label)
#     loss += l
#     num_correct += acc
#
# num_tests = len(test_images)
# print('Test Loss:', loss / num_tests)
# print('Test Accuracy:', num_correct / num_tests)
#
# lastAccuracy = num_correct / num_tests

for epoch in range(1):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train
    loss = 0
    num_correct = 0
    # i: index
    # im: image
    # label: label
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct)
            )
            loss = 0
            num_correct = 0
        im = im[np.newaxis, :, :]
        l, acc = train(im, label)
        loss += l
        num_correct += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    im = im[np.newaxis, :, :]
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

Accuracy = num_correct / num_tests

# if Accuracy > lastAccuracy:
with open('conv.t.pkl', 'wb') as f:
    pickle.dump(conv, f)

with open('conv1.t.pkl', 'wb') as f:
    pickle.dump(conv1, f)

with open('pool1.t.pkl', 'wb') as f:
    pickle.dump(pool1, f)

# with open('conv2.t.pkl', 'wb') as f:
#     pickle.dump(conv2, f)
#
# with open('pool2.t.pkl', 'wb') as f:
#     pickle.dump(pool2, f)
#
with open('hidden1.t.pkl', 'wb') as f:
    pickle.dump(hidden1, f)

# with open('hidden2.t.pkl', 'wb') as f:
#     pickle.dump(hidden2, f)
#
with open('softmax.t.pkl', 'wb') as f:
    pickle.dump(softmax, f)
