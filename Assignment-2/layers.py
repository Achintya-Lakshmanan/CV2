import numpy as np

# Starter code for layers.py 
# Put all your forward and backprop function implementations in here.
# Refer to the assignment description for shape of each input/output.


### relu

def forw_relu(x):

    # your code goes here
    # x is an array of size m x n
    y = np.maximum(0, x)

    return y

def back_relu(x, y, dzdy):

    # your code goes here
    # Pass gradients only when output > 0
    dzdx = dzdy * (y > 0)   

    return dzdx


### maxpool

def forw_maxpool(x):

    # your code goes here
    # Find max of each 2x2 square in x
    stride = 2
    m, n = x.shape
    m //= stride
    n //= stride
    y = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            y[i, j] = np.max(x[i*stride:(i+1)*stride, j*stride:(j+1)*stride])
    

    return y

def back_maxpool(x, y, dzdy):

    # your code goes here
    # Similar to before, find max value that passed through and that gets the gradients
    stride = 2
    m, n = x.shape
    m //= stride
    n //= stride
    dzdx = np.zeros_like(x)
    for i in range(m):
        for j in range(n):
            max_val = y[i, j]
            for ii in range(stride):
                for jj in range(stride):
                    if x[i*stride + ii, j*stride + jj] == max_val:
                        dzdx[i*stride + ii, j*stride + jj] = dzdy[i, j]

    return dzdx


### meanpool

def forw_meanpool(x):

    # your code goes here
    # Max pool but take mean
    stride = 2
    m, n = x.shape
    m //= stride
    n //= stride
    y = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            y[i, j] = np.mean(x[i*stride:(i+1)*stride, j*stride:(j+1)*stride])

    return y

def back_meanpool(x, y, dzdy):

    # your code goes here
    # Same as before, each value gets 1/4th the gradient
    stride = 2
    m, n = x.shape
    m //= stride
    n //= stride
    dzdx = np.zeros_like(x)
    for i in range(m):
        for j in range(n):
            dzdx[i*stride:(i+1)*stride, j*stride:(j+1)*stride] = dzdy[i, j] / (stride * stride)

    return dzdx



### FC (fully connected)

def forw_fc(x, w, b):
    y = np.sum(x * w) + b
    y = float(y)

    return y


def back_fc(x, w, b, y, dzdy):

    # your code goes here
    # As deriveed in class
    dzdx = dzdy * w
    dzdw = dzdy * x
    dzdb = dzdy
    dzdb = float(dzdb)

    return dzdx, dzdw, dzdb


### softmax

def forw_softmax(x):

    # your code goes here
    # Subtract max value for stability
    x = x - np.max(x)
    y = np.exp(x) / np.sum(np.exp(x))
    y = np.squeeze(y)

    return y

def back_softmax(x, y, dzdy):

    # your code goes here
    # As derived in class

    dzdx = np.zeros_like(x)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                dzdx[i] += y[i] * (1 - y[j]) * dzdy[j]
            else:
                dzdx[i] -= y[i] * y[j] * dzdy[j]

    dzdx = np.squeeze(dzdx)

    return dzdx

################# Self test code ###################


data = {}

with open('selftestfile.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

i = 0
while i < len(lines):
    func_name = lines[i]
    i += 1

    num_inputs = int(lines[i])
    i += 1
    
    inputs = []
    for _ in range(num_inputs):
        parts = lines[i].split()
        num_rows = int(parts[0])
        num_cols = int(parts[1])
        values = [float(x) for x in parts[2:]]
        arr = np.array(values).reshape((num_rows, num_cols))
        inputs.append(arr)
        i += 1
    
    num_outputs = int(lines[i])
    i += 1
    
    outputs = []
    for _ in range(num_outputs):
        parts = lines[i].split()
        num_rows = int(parts[0])
        num_cols = int(parts[1])
        values = [float(x) for x in parts[2:]]
        arr = np.array(values).reshape((num_rows, num_cols))
        outputs.append(arr)
        i += 1

    if func_name not in data:
        data[func_name] = []
    
    data[func_name].append({
        'inputs': inputs,
        'outputs': outputs
    })


if __name__ == "__main__":

    print("First self test")

    forw_relu_output_1 = forw_relu(data['forw_relu'][0]['inputs'][0])
    print("Forw relu output:")
    print(forw_relu_output_1)
    print("Expected forw relu output:")
    print(data['forw_relu'][0]['outputs'][0])

    print("-" * 50)

    back_relu_output_1 = back_relu(data['back_relu'][0]['inputs'][0], data['back_relu'][0]['inputs'][1], data['back_relu'][0]['inputs'][2])
    print("Back relu output:")
    print(back_relu_output_1)
    print("Expected back relu output:")
    print(data['back_relu'][0]['outputs'][0])

    print("-" * 50)

    forw_maxpool_output_1 = forw_maxpool(data['forw_maxpool'][0]['inputs'][0])
    print("Forw maxpool output:")
    print(forw_maxpool_output_1)
    print("Expected forw maxpool output:")
    print(data['forw_maxpool'][0]['outputs'][0])

    print("-" * 50)

    back_maxpool_output_1 = back_maxpool(data['back_maxpool'][0]['inputs'][0], data['back_maxpool'][0]['inputs'][1], data['back_maxpool'][0]['inputs'][2])
    print("Back maxpool output:")
    print(back_maxpool_output_1)
    print("Expected back maxpool output:")
    print(data['back_maxpool'][0]['outputs'][0])

    print("-" * 50)

    forw_meanpool_output_1 = forw_meanpool(data['forw_meanpool'][0]['inputs'][0])
    print("Forw meanpool output:")
    print(forw_meanpool_output_1)
    print("Expected forw meanpool output:")
    print(data['forw_meanpool'][0]['outputs'][0])

    print("-" * 50)

    back_meanpool_output_1 = back_meanpool(data['back_meanpool'][0]['inputs'][0], data['back_meanpool'][0]['inputs'][1], data['back_meanpool'][0]['inputs'][2])
    print("Back meanpool output:")
    print(back_meanpool_output_1)
    print("Expected back meanpool output:")
    print(data['back_meanpool'][0]['outputs'][0])

    print("-" * 50) 

    forw_fc_output_1 = forw_fc(data['forw_fc'][0]['inputs'][0], data['forw_fc'][0]['inputs'][1], data['forw_fc'][0]['inputs'][2])
    print("Forw fc output:")
    print(forw_fc_output_1)
    print("Expected forw fc output:")
    print(data['forw_fc'][0]['outputs'][0])

    print("-" * 50) 

    back_fc_output_1 = back_fc(data['back_fc'][0]['inputs'][0], data['back_fc'][0]['inputs'][1], data['back_fc'][0]['inputs'][2], data['back_fc'][0]['inputs'][3], data['back_fc'][0]['inputs'][4])
    print("Back fc output:")
    print(back_fc_output_1)
    print("Expected back fc output:")
    print(data['back_fc'][0]['outputs'][0])
    print(data['back_fc'][0]['outputs'][1])
    print(data['back_fc'][0]['outputs'][2])

    print("-" * 50)

    forw_softmax_output_1 = forw_softmax(data['forw_softmax'][0]['inputs'][0])
    print("Forw softmax output:")
    print(forw_softmax_output_1)
    print("Expected forw softmax output:")
    print(data['forw_softmax'][0]['outputs'][0])   

    print("-" * 50)

    back_softmax_output_1 = back_softmax(data['back_softmax'][0]['inputs'][0], data['back_softmax'][0]['inputs'][1], data['back_softmax'][0]['inputs'][2])
    print("Back softmax output:")
    print(back_softmax_output_1)
    print("Expected back softmax output:")
    print(data['back_softmax'][0]['outputs'][0])

    print("-" * 50)

    print("Second self test")
    forw_relu_output_2 = forw_relu(data['forw_relu'][1]['inputs'][0])
    print("Forw relu output:")
    print(forw_relu_output_2)
    print("Expected forw relu output:")
    print(data['forw_relu'][1]['outputs'][0])

    print("-" * 50)

    back_relu_output_2 = back_relu(data['back_relu'][1]['inputs'][0], data['back_relu'][1]['inputs'][1], data['back_relu'][1]['inputs'][2])
    print("Back relu output:")
    print(back_relu_output_2)
    print("Expected back relu output:")
    print(data['back_relu'][1]['outputs'][0])

    print("-" * 50)

    forw_maxpool_output_2 = forw_maxpool(data['forw_maxpool'][1]['inputs'][0])
    print("Forw maxpool output:")
    print(forw_maxpool_output_2)
    print("Expected forw maxpool output:")
    print(data['forw_maxpool'][1]['outputs'][0])

    print("-" * 50)

    back_maxpool_output_2 = back_maxpool(data['back_maxpool'][1]['inputs'][0], data['back_maxpool'][1]['inputs'][1], data['back_maxpool'][1]['inputs'][2])
    print("Back maxpool output:")
    print(back_maxpool_output_2)
    print("Expected back maxpool output:")
    print(data['back_maxpool'][1]['outputs'][0])

    print("-" * 50)

    forw_meanpool_output_2 = forw_meanpool(data['forw_meanpool'][1]['inputs'][0])
    print("Forw meanpool output:")
    print(forw_meanpool_output_2)
    print("Expected forw meanpool output:")
    print(data['forw_meanpool'][1]['outputs'][0])

    print("-" * 50)

    back_meanpool_output_2 = back_meanpool(data['back_meanpool'][1]['inputs'][0], data['back_meanpool'][1]['inputs'][1], data['back_meanpool'][1]['inputs'][2])
    print("Back meanpool output:")
    print(back_meanpool_output_2)
    print("Expected back meanpool output:")
    print(data['back_meanpool'][1]['outputs'][0])

    print("-" * 50) 

    forw_fc_output_2 = forw_fc(data['forw_fc'][1]['inputs'][0], data['forw_fc'][1]['inputs'][1], data['forw_fc'][1]['inputs'][2])
    print("Forw fc output:")
    print(forw_fc_output_2)
    print("Expected forw fc output:")
    print(data['forw_fc'][1]['outputs'][0])

    print("-" * 50) 

    back_fc_output_2 = back_fc(data['back_fc'][1]['inputs'][0], data['back_fc'][1]['inputs'][1], data['back_fc'][1]['inputs'][2], data['back_fc'][1]['inputs'][3], data['back_fc'][1]['inputs'][4])
    print("Back fc output:")
    print(back_fc_output_2)
    print("Expected back fc output:")
    print(data['back_fc'][1]['outputs'][0])
    print(data['back_fc'][1]['outputs'][1])
    print(data['back_fc'][1]['outputs'][2])

    print("-" * 50)

    forw_softmax_output_2 = forw_softmax(data['forw_softmax'][1]['inputs'][0])
    print("Forw softmax output:")
    print(forw_softmax_output_2)
    print("Expected forw softmax output:")
    print(data['forw_softmax'][1]['outputs'][0])   

    print("-" * 50)

    back_softmax_output_2 = back_softmax(data['back_softmax'][1]['inputs'][0], data['back_softmax'][1]['inputs'][1], data['back_softmax'][1]['inputs'][2])
    print("Back softmax output:")
    print(back_softmax_output_2)
    print("Expected back softmax output:")
    print(data['back_softmax'][1]['outputs'][0])
