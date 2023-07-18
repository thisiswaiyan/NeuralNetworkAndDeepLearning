import numpy as np
import copy
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimag



# Loading the data (cat/ non-cat)
#train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# training examples m
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
#shape of training examples ([209, 64, 64, 3])
#numpx = number of height / width (64)
#209 = training examples
#3 = number of images

# flatten training set n test set
# A trick to flatten the shape (a,b,c,d) to shape of (b*c*d, a)
# x_flatten = x.reshape(x.shape[0], -1).T
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#Check the first 10 pixels are in the right place
assert np.alltrue(train_set_x_flatten[0:10, 1] == [196, 192, 190, 193, 186, 182, 188, 179, 174, 213]), "Wrong solution. Use (X.shape[0], -1).T"
assert np.alltrue(test_set_x_flatten[0:10, 1] == [115, 110, 111, 137, 129, 129, 155, 146, 145, 159]), "Wrong solution. Use (X.shape[0], -1).T"

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


# sigmoid function (sigmoid(z) = 1 / (1 + e^(-z))) where z = w.Tx + b
def sigmoid(z):
    # z -- A scalar of numpy array of any size
    s = 1 / (1 + np.exp(-z))
    return s


# testing sigmoid function
print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))

x = np.arrary([0.5, 0, 2.0])
output = sigmoid(x)
print(output)


#initialize the parameters(w, b)
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros and shape (dim, 1) for w and initialize b to 0.

    dim -- size of the w vector we want
    """
    w = np.zeros((dim, 1))
    b = float(0)
    return w, b


dim = 2
w, b = initialize_with_zeros(dim)

assert type(b) == float
print("w: " + str(w))
print("b: " + str(b))


# Forward and backward propagation
def propagate(w, b, X, Y):
    # Forward prop
    # X.shape[1] means counting in columns of total of x
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    cost = (-1/m) * np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1 - A)))

    # Backward prop
    dw = (1 / m) * np.dot(X,(A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


w = np.array([[1.], [2]])
b = 1.5
X = np.array([[1., -2, -1], [3., 0.5, -3.2]])
Y = np.array([[1, 1, 0]])
grads, cost = propagate(w, b, X, Y)

assert type(grads["dw"]) == np.ndarray
assert grads["dw"].shape == (2, 1)
assert type(grads["db"]) == np.float64

print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))

# optimizing the parameters w and b
# the goal is to minimize the cost function J. So, we have to update the w and b
def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    # num_iterations -- number of iteration to the global mininum
    # learning rate -- alpha value
    # print_cost -- True to print after every 100 times iterations
    # deepcopy can edit the value inside the array doesn't effect each other
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration is {i}: {cost}")
        
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


# predict the Y value
# convert the value if A_value <= 0.5 convert Y_prediction to 0 elif A_value > 0.5 convert to 1
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range (A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    return Y_prediction


w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
print("prediction = " + str(predict(w, b, X)))


# make a model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


#logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# predict the cat = 1 or not 0
#index = 2
#plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
#print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(logistic_regression_model['Y_prediction_test'][0,index])].decode("utf-8") +  "\" picture.")

#learning_rates = [0.01, 0.001, 0.0001]
#models = {}

#for lr in learning_rates:
    #print ("Training a model with learning rate: " + str(lr))
    #models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
    #print ('\n' + "-------------------------------------------------------" + '\n')

#for lr in learning_rates:
    #plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

#plt.ylabel('cost')
#plt.xlabel('iterations (hundreds)')

#legend = plt.legend(loc='upper center', shadow=True)
#frame = legend.get_frame()
#frame.set_facecolor('0.90')
#plt.show()
