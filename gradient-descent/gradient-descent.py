epochs = 200
learning_rate = 0.1


def mean_squared_loss(hypothesis, xs, ys, t1, t2):
    sum_delta_squared = 0

    for i, _ in enumerate(xs):
        sum_delta_squared += (hypothesis(t1, t2, xs[i]) - ys[i]) ** 2

    return (1 / (2 * len(xs))) * sum_delta_squared


def gradient_descent(hypothesis, xs, ys):
    theta1, theta2 = 0, 1

    e = 0
    while e < epochs:
        e += 1
        sum_differences_t1 = 0
        sum_differences_t2 = 0

        for i, _ in enumerate(xs):
            sum_differences_t1 += hypothesis(theta1, theta2, xs[i]) - ys[i]

        for i, _ in enumerate(xs):
            sum_differences_t2 += (hypothesis(theta1, theta2, xs[i]) - ys[i]) * xs[i]

        derivative_of_loss_t1 = (1 / len(xs)) * sum_differences_t1
        derivative_of_loss_t2 = (1 / len(xs)) * sum_differences_t2

        loss = mean_squared_loss(hypothesis, xs, ys, theta1, theta2)
        print("loss: {}".format(loss))

        theta1 = theta1 - learning_rate * derivative_of_loss_t1
        theta2 = theta2 - learning_rate * derivative_of_loss_t2

    return theta1, theta2


xs = [1, 2, 3, 4]
ys = [2, 4, 6, 8]

theta1 = 0
theta2 = 1
def hypothesis(t1, t2, x): return t1 + t2 * x

print(gradient_descent(hypothesis, xs, ys))
