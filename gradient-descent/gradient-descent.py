epochs = 200
learning_rate = 0.1


def mean_squared_loss(hypothesis, xs, ys, t0, t1):
    sum_delta_squared = 0

    for i, _ in enumerate(xs):
        sum_delta_squared += (hypothesis(t0, t1, xs[i]) - ys[i]) ** 2

    return (1 / (2 * len(xs))) * sum_delta_squared


def gradient_descent(hypothesis, xs, ys):
    theta0, theta1, size = 0, 1, len(xs)

    e = 0
    while e < epochs:
        e += 1
        sum_differences_t0 = 0
        sum_differences_t1 = 0

        for i, _ in enumerate(xs):
            sum_differences_t0 += hypothesis(theta0, theta1, xs[i]) - ys[i]

        for i, _ in enumerate(xs):
            sum_differences_t1 += (hypothesis(theta0, theta1, xs[i]) - ys[i]) * xs[i]

        derivative_of_loss_t0 = (1 / size) * sum_differences_t0
        derivative_of_loss_t1 = (1 / size) * sum_differences_t1

        if e % 10 == 0:
            print("loss: {}".format(mean_squared_loss(hypothesis, xs, ys, theta0, theta1)))

        theta0 = theta0 - learning_rate * derivative_of_loss_t0
        theta1 = theta1 - learning_rate * derivative_of_loss_t1

    return theta0, theta1


xs = [1, 2, 3, 4]
ys = [2, 4, 6, 8]

def hypothesis(t0, t1, x): return t0 + t1 * x

t1, t2 = gradient_descent(hypothesis, xs, ys)

print("t1: {}, t2: {}".format(t1, t2))

print("predict: 100 -> {}".format(hypothesis(t1, t2, 100)))
