epochs = 100
learning_rate = 0.01


def mean_squared_loss(hypothesis, xs, ys):
    sum_delta_squared = 0

    for i, _ in enumerate(xs):
        sum_delta_squared += (hypothesis(xs[i]) - ys[i]) ** 2

    return (1 / (2 * len(xs))) * sum_delta_squared


def gradient_descent(xs, ys):
    theta1 = 0
    theta2 = 1
    def hypothesis(x): return theta1 + theta2 * x

    e = 0
    while e < epochs:
        e += 1
        sum_differences = 0

        for i, _ in enumerate(xs):
            sum_differences += hypothesis(xs[i]) - ys[i]

        for i, _ in enumerate(xs):
            derivative_of_loss_t1 = (1 / len(xs)) * sum_differences
            derivative_of_loss_t2 = (1 / len(xs)) * sum_differences * xs[i]

            # print(sum_differences)

            theta1 = theta1 - learning_rate * derivative_of_loss_t1
            theta2 = theta2 - learning_rate * derivative_of_loss_t2

    return theta1, theta2


xs = [3, 1, 0, 4]
ys = [3, 2, 1, 3]

theta1 = 0
theta2 = 1
def hypothesis(x): return theta1 + theta2 * x


# i = 0
# while i < epochs:
#     i = i + 1
#     loss = mean_squared_loss(hypothesis, xs, ys)
#     print("loss: {}".format(loss))

print(gradient_descent(xs, ys))
