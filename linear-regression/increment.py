import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from utils import show_output, get_input_data

x = np.arange(1000).reshape(-1, 1)
y = np.arange(1, 1001)

x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression()

model.fit(x, y)

score = model.score(x_test, y_test)

input_data = get_input_data()

prediction = model.predict(input_data)

show_output(input_data, score, prediction)
