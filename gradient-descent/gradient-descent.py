epochs = 100
learning_rate = 0.01


def mean_squared_loss(hypothesis, xs, ys):
    sum_delta_squared = 0

    i = 0
    while i < len(xs):
        sum_delta_squared = sum_delta_squared + \
            ((hypothesis(xs[i]) - ys[i]) ** 2)
        i = i + 1

    return (1 / (2 * len(xs))) * sum_delta_squared


def gradient_descent(t1, t2, loss):
    # derivative_of_loss_t1 = (1 / len(xs)) * (sum((hypothesis(xs[i]) - ys[i])))
    # derivative_of_loss_t2 = (1 / len(xs)) * \
    #     (sum((hypothesis(xs[i]) - ys[i]) * xs[i]))

    derivative_of_loss_t1 = loss
    derivative_of_loss_t2 = loss * xs[i]

    new_t1 = t1 - learning_rate * derivative_of_loss_t1
    new_t2 = t2 - learning_rate * derivative_of_loss_t2

    return new_t1, new_t2


xs = [3, 1, 0, 4]
ys = [3, 2, 1, 3]

theta1 = 0
theta2 = 1
def hypothesis(x): return theta1 + theta2 * x


i = 0
while i < epochs:
    i = i + 1
    loss = mean_squared_loss(hypothesis, xs, ys)
    # theta1, theta2 = gradient_descent(theta1, theta2, loss)
    print("loss: {}".format(loss))
