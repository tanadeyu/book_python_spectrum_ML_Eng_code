# Define and apply ReLU function
def relu(x):
     return max(0, x)

# Example: Apply ReLU to input value x
x1 = 5
x2 = -5
result1 = relu(x1) # result: 5
result2 = relu(x2) # result: 0
print('result1 = ',result1)
print('result2 = ',result2)
