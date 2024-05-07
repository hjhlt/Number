import pickle

import mnist
import numpy as np
import scipy.special


def sigmoid_derivative(x):
    return x * (1 - x)


class Conv3x3:
    def __init__(self, num_filters, is_first):
        self.num_filters = num_filters
        self.is_first = is_first
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        x, h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[:, i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        x, h, w = input.shape
        output = np.zeros((self.num_filters, h - 2, w - 2))
        for im_region, i, j in self.iterate_regions(input):
            for l in range(self.num_filters):
                for k in range(x):
                    output[l, i, j] += np.sum(im_region[k] * self.filters[l], axis=(0, 1))
        return output

    def backprop(self, d_L_d_out, learn_rate):
        x, h, w = self.last_input.shape  # 获取输入尺寸
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += np.sum(d_L_d_out[f, i, j] * im_region, axis=0)  # 累加梯度
        self.filters -= learn_rate * d_L_d_filters  # 更新卷积核权值
        if self.is_first:
            d_L_d_input = np.zeros(self.last_input.shape)
            filters_transposed = self.filters[:, ::-1, ::-1]  # 转置卷积核
            d_L_d_out = np.pad(d_L_d_out, ((0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)  # 零填充
            for i in range(d_L_d_input.shape[1]):
                for j in range(d_L_d_input.shape[2]):
                    for l in range(x):
                        d_L_d_input_region = 0
                        for f in range(self.num_filters):
                            d_L_d_input_region += np.sum(filters_transposed[f] * d_L_d_out[f, i:i + 3, j:j + 3])
                        d_L_d_input[l, i, j] += d_L_d_input_region  # 计算输入梯度
            return d_L_d_input
        return None


class MaxPool2:

    def iterate_regions(self, image):
        _, h, w = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[:, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        num_filters, h, w = input.shape
        output = np.zeros((num_filters, h // 2, w // 2))
        for im_region, i, j in self.iterate_regions(input):
            for k in range(num_filters):
                output[k, i, j] = np.amax(im_region[k], axis=(0, 1))
        return output

    def backprop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            f, h, w = im_region.shape
            amax = np.amax(im_region, axis=(1, 2))
            for f2 in range(f):
                for i2 in range(h):
                    for j2 in range(w):
                        if im_region[f2, i2, j2] == amax[f2]:
                            d_L_d_input[f2, i * 2 + i2, j * 2 + j2] = d_L_d_out[f2, i, j]
        return d_L_d_input


class Hidden:

    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        input_len, nodes = self.weights.shape
        totals = np.dot(input, self.weights)
        self.last_totals = totals
        self.result = scipy.special.expit(totals)
        return self.result

    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_t = d_L_d_out * sigmoid_derivative(self.result)
        d_L_d_w = self.last_input[np.newaxis].T @ d_L_d_t[np.newaxis]
        d_L_d_inputs = self.weights @ d_L_d_t
        self.weights -= learn_rate * d_L_d_w
        return d_L_d_inputs.reshape(self.last_input_shape)


class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input = input
        input_len, nodes = self.weights.shape
        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights
            d_L_d_t = gradient * d_out_d_t
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

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

# conv = Conv3x3(4, 0)  # 28x28x1 -> 26x26x4
# conv1 = Conv3x3(2, 1)  # 26x26x4 -> 24x24x4
# pool1 = MaxPool2()  # 26x26x2 -> 13x13x2
# conv2 = Conv3x3(4, 1)  # 13x13x2 -> 11x11x2
# pool2 = MaxPool2()  # 22x22x2 -> 11x11x2
# hidden1 = Hidden(11 * 11 * 4, 20)  # 5x5x2 -> 20
# hidden2 = Hidden(50, 20)  # 50 -> 20
# softmax = Softmax(20, 10)  # 20 -> 10


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


with open('conv.t.pkl', 'rb') as f:
    conv = pickle.load(f)

with open('pool1.t.pkl', 'rb') as f:
    pool1 = pickle.load(f)

with open('conv2.t.pkl', 'rb') as f:
    conv2 = pickle.load(f)

with open('hidden1.t.pkl', 'rb') as f:
    hidden1 = pickle.load(f)

with open('softmax.t.pkl', 'rb') as f:
    softmax = pickle.load(f)


def forward(image, label):
    out = conv.forward((image / 255) - 0.5)
    out = pool1.forward(out)
    out = conv2.forward(out)
    out = hidden1.forward(out)
    out = softmax.forward(out)
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc


def train(im, label, lr=0.0001):
    out, loss, acc = forward(im, label)
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    gradient = softmax.backprop(gradient, lr)
    gradient = hidden1.backprop(gradient, lr)
    gradient = conv2.backprop(gradient, lr)
    gradient = pool1.backprop(gradient)
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

epoch_num = 1

for epoch in range(epoch_num):
    print('--- Epoch %d ---' % (epoch + 1))
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]
    loss = 0
    num_correct = 0
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

# with open('conv1.t.pkl', 'wb') as f:
#     pickle.dump(conv1, f)

with open('pool1.t.pkl', 'wb') as f:
    pickle.dump(pool1, f)

with open('conv2.t.pkl', 'wb') as f:
    pickle.dump(conv2, f)
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
