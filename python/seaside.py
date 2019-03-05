from typing import List, Callable, Tuple, Union, Dict
from math import log10, ceil, floor, exp, log
import random

class Vec:
    """ An implementation of mathematical Vectors.
    
    === Public Attributes ===
    data: The values stored in the vector.
    precision: The display precision.
    """

    data: List[float]
    precision: int

    def __init__(self, data: List[float]) -> None:
        """Initialize the Vector."""

        self.data = data
        self.precision = 1

    def __len__(self) -> int:
        """Return the dimension of this Vector."""
        
        return len(self.data)

    def __getitem__(self, i: int) -> float:
        """Return the i-th value of this Vector."""
        
        return self.data[i]

    def __setitem__(self, i: int, x: float) -> None:
        """Set the i-th element of this Vector to be x."""

        self.data[i] = x

    def __calculate_num_digits(self, num: float) -> int:
        """Return the number of digits a number will take up."""
    
        num_digits = 0
        
        # Negative Number
        if num < 0:
            num_digits += 1

        # -1 < Number < 1
        if floor(abs(num)) == 0:
            num_digits += 1
        # abs(Number) >= 1
        else:
            num_digits += 1 + floor(log10(abs(num)))

        num_digits += 1 + self.precision

        return num_digits

    def __str__(self) -> str:
        """Return the string representation of this Vector."""
        
        max_digits = 0
        for i in range(len(self)):
            cur = self.__calculate_num_digits(self[i])

            if cur > max_digits:
                max_digits = cur

        out = '┌' + (' ' * (2 + max_digits)) + '┐\n'
        for i in range(len(self)):
            digits = self.__calculate_num_digits(self[i])
            out += '| ' + (' ' * (max_digits - digits)) 

            if self[i] < 0:
                out += '-'

            abs_num = int(abs(self[i]))
            out += str(abs_num) + '.'
            out += str(int(abs(self[i]) * (10**self.precision)))[-self.precision:]
            

            out += ' |\n'

        out += '└' + (' ' * (2 + max_digits)) + '┘\n'

        return out

    def __add__(self, other: 'Vec') -> 'Vec':
        """Return the addition of this vector and the other vector."""
        
        if len(other) != len(self):
            raise VecError("When adding two Vectors, the dimensions should match!")
        else:
            values = []
            for i in range(len(self)):
                values.append(self[i] + other[i])

            return Vec(values) 
    
    def __sub__(self, other: 'Vec') -> 'Vec':
        """Return the difference of this Vector and other."""

        if len(other) != len(self):
            raise VecError("When subtracting two Vectors, the dimensions should match!")
        else:
            values = []
            for i in range(len(self)):
                values.append(self[i] - other[i])

            return Vec(values) 

    def __mul__(self, c: float) -> 'Vec':
        """Scalar multiply this Vector by constant c."""

        if not isinstance(c, float) and not isinstance(c, int):
            raise VecError('Vectors should only be multiplied by constants!')
        else:
            values = []
            
            for i in range(len(self)):
                values.append(self[i] * c)

            return Vec(values)

    def __truediv__(self, c: float) -> 'Vec':
        """Scalar divide this Vector by constant c."""

        if not isinstance(c, float) and not isinstance(c, int):
            raise VecError('Vectors should only be divided by constants!')
        else:
            values = []
            
            for i in range(len(self)):
                values.append(self[i] / c)

            return Vec(values)

    def dot(self, other: 'Vec') -> float:
        """Return the dot product of this Vector and other."""
        if not isinstance(other, Vec) or len(other) != len(self):
            raise VecError('The other vector is of the wrong length or may not be a vector!')
        else:
            dot_total = 0
            for i in range(len(self)):
                dot_total += self[i] * other[i]

            return dot_total

    def norm(self) -> float:
        """Return the norm of this Vector."""  

        return (self.dot(self))**0.5

    def clone(self) -> 'Vec':
        """Return a clone of this Vector."""
        vec_data = []

        for i in range(len(self)):
            vec_data.append(self[i])
        
        return Vec(vec_data)

    def map(self, f: Callable) -> 'Vec':
        """Map this Vector through the given function."""

        clone = self.clone()
        return f(clone)

    def fill(self, x: float, n: int) -> 'Vec':
        """Create a Vector of n elements each holding x."""

        out = [x] * n
        return Vec(out)

    def onehot(self, i: int, size: int) -> 'Vec':
        """Return a onehot Vector, hot at i, and of length size."""

        if i >= size:
            raise VecError('The hot element must be within bounds!')
        else:
            values = [0.0] * size
            values[i] = 1.0

            return Vec(values)

    def randomize(self, min: float, max: float) -> 'Vec':
        """Randomize this Vector between min and max."""

        data = []

        for i in range(len(self)):
            data.append((random.random() * (max - min)) + min)

        return Vec(data)

    def to_mat(self) -> 'Mat':
        """Turn this column Vector into a Matrix."""
        clone = self.clone()
        out = Mat(0, 0)
        out.data = [clone]

        return out

class VecError(Exception):
    """A Vector Class Error."""

    def __init__(self, error: str) -> None:
        """Print the error that occured."""
        print(str)
        quit()    

class Mat:
    """ An implemention of Matricies in math.
    
    === Public Attributes ===
    data: An array of column vectors.
    precision: The display precision of this matrix
    size: The dimensions of this Matrix when filling.
    """

    data: List['Vec']
    precision: int
    size: List[int]

    def __init__(self, m: int, n: int) -> None:
        """Initialization of a m by n Matrix."""

        self.data = []
        for i in range(n):
            self.data.append(Vec.fill(None, 0, m))

        self.precision = 1

    def dim(self) -> Tuple[int, int]:
        """Return the dimensions of this Matrix."""
        if len(self.data) != 0:
            return (len(self.data[0]), len(self.data))
        else:
            return (0, 0)

    def __getitem__(self, i: int) -> 'Vec':
        """Return the i-th column Vector."""

        return self.data[i]

    def __calculate_num_digits(self, x: float) -> int:
        """Calculate the number of characters needed to display the number."""

        num_digits = 0
        
        # Negative Number
        if x < 0:
            num_digits += 1

        # -1 < Number < 1
        if floor(abs(x)) == 0:
            num_digits += 1
        # abs(Number) >= 1
        else:
            num_digits += 1 + floor(log10(abs(x)))

        num_digits += 1 + self.precision

        return num_digits

    def __str__(self) -> str:
        """Return the string representation of this Matrix."""

        dim = self.dim()

        # Calculate the column widths
        max_digits = [-1] * dim[1]
        for x in range(dim[1]):
            for y in range(dim[0]):
                cur_num_digits = self.__calculate_num_digits(self[x][y])

                if cur_num_digits > max_digits[x]:
                    max_digits[x] = cur_num_digits

        digits_width = 0
        for i in range(len(max_digits)):
            digits_width += max_digits[i]

        out = '┌' + (' ' * (dim[1] + 1 + digits_width)) + '┐\n'
        for y in range(dim[0]):
            out += '|'

            for x in range(dim[1]):
                digits = self.__calculate_num_digits(self[x][y])
            
                out += (' ' * (max_digits[x] - digits + 1)) 

                if self[x][y] < 0:
                    out += '-'

                abs_num = int(abs(self[x][y]))
                out += str(abs_num) + '.'
                
                abs_str = str(abs_num)
                try:
                    out += '0' * (self.precision - ceil(log10(int(abs(self[x][y]) * (10**self.precision)))))
                except ValueError:
                    out += '0'
                out += str(int(abs(self[x][y]) * (10**self.precision)))[-self.precision:]

            out += ' |\n'
        
        out += '└' + (' ' * (dim[1] + 1 + digits_width)) + '┘\n'

        return out

    def clone(self) -> 'Mat':
        """Return a clone of this Matrix."""

        cols = []  
        for i in range(self.dim()[1]):
            cols.append(self[i].clone())

        result = Mat(0, 0)
        result.data = cols
        
        return result

    def append_column(self, col: 'Vec') -> 'Mat':
        """Append the col Vector to this Matrix."""

        if len(col) != self.dim()[0]:
            raise MatrixError('The Vector being appended must have the correct dimension!')
        else:
            clone = self.clone()
            clone.data.append(col)

            return clone
    
    def append_row(self, row: 'Vec') -> 'Mat':
        """Append the row Vector to this Matrix."""

        if len(row) != self.dim()[1]:
            raise MatrixError('The Vector being appended must have the correct dimension!')
        else:
            clone = self.clone()
            for i in range(self.dim()[1]):
                clone[i].data.append(row[i])

            return clone

        """Return the i-th column Vector."""

        if i < 0 or i >= self.dim()[1]:
            raise MatrixError('Index out of bounds: ' + str(i))
        else:
            return self[i].clone()

    def get_row(self, i: int) -> 'Vec':
        """Return the i-th row Vector."""

        if i < 0 or i >= self.dim()[0]:
            raise MatrixError('Index out of bounds: ' + str(i))
        else:    
            values = []
            for j in range(self.dim()[1]):
                values.append(self[j][i])

            return Vec(values)

    def map(self, f: Callable) -> 'Mat':
        """Map the Matrix through f."""

        cols = []
        for i in range(self.dim()[1]):
            cols.append(self.data[i].map(f))
        
        result = Mat(0, 0)
        result.data = cols

        return result

    def __mul__(self, other) -> 'Mat':
        """Matrix or constant multiply this Matrix by other."""

        if isinstance(other, int) or isinstance(other, float):
            cols = []
            for i in range(self.dim()[1]):
                cols.append(self[i] * other)

            result = Mat(0, 0)
            result.data = cols
            
            return result
        else:
            if self.dim()[1] != other.dim()[0]:
                raise MatrixError('When Matrix multiplying the two inner dimensions must match!')
            else:
                mat = Mat(self.dim()[0], other.dim()[1])

                for y in range(mat.dim()[0]):
                    for x in range(mat.dim()[1]):
                        mat[x][y] = self.get_row(y).dot(other[x])

                return mat

    def fill(self, val: float) -> 'Mat':
        """Fill the Matrix with val."""

        clone = self.clone()
        
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                clone[x][y] = val
        
        return clone

    def hadamard(self, other: 'Mat') -> 'Mat':
        """Element wise multiplication."""
        
        if self.dim()[0] != other.dim()[0] or self.dim()[1] != other.dim()[1]:
            raise MatrixError('The dimensions of the two Matricies must be the same!')
        else:
            dim = self.dim()

            result = Mat(dim[0], dim[1])
            
            for y in range(dim[0]):
                for x in range(dim[1]):
                    result[x][y] = self[x][y] * other[x][y]
            
            return result

    def __add__(self, other: 'Mat') -> 'Mat':
        """Add other to self."""

        if self.dim()[0] != other.dim()[0] or self.dim()[1] != other.dim()[1]:
            raise MatrixError('The dimensions of the two Matricies must be the same!')
        else:
            dim = self.dim()

            result = Mat(dim[0], dim[1])
            
            for y in range(dim[0]):
                for x in range(dim[1]):
                    result[x][y] = self[x][y] + other[x][y]
            
            return result

    def __sub__(self, other: 'Mat') -> 'Mat':
        """Subtract other from self."""

        if self.dim()[0] != other.dim()[0] or self.dim()[1] != other.dim()[1]:
            raise MatrixError('The dimensions of the two Matricies must be the same!')
        else:
            dim = self.dim()

            result = Mat(dim[0], dim[1])
            
            for y in range(dim[0]):
                for x in range(dim[1]):
                    result[x][y] = self[x][y] - other[x][y]
            
            return result

    def randomize(self, min: float, max: float) -> 'Mat':
        """Randomize this Matrix between min and max."""

        data = []

        dim = self.dim()
        for i in range(dim[1]):
            data.append(self[i].randomize(min, max))

        out = Mat(dim[0], dim[1])
        out.data = data

        return out

    def transpose(self) -> 'Mat':
        """Transpose this Matrix."""
        
        data = []
        for y in range(self.dim()[0]):
            data.append(self.get_row(y))

        result = Mat(0, 0)
        result.data = data

        return result

    def remove_row(self, i: int) -> 'Mat':
        """Remove the i-th row of this Matrix."""

        clone = self.clone()
        for x in range(self.dim()[1]):
            clone[x].data.pop(i)

        return clone

    def normalize(self) -> 'Mat':
        """Normalize the Matrix over its highest value."""
        highest = -99999999
        dim = self.dim()

        for y in range(dim[0]):
            for x in range(dim[1]):
                if self[x][y] > highest:
                    highest = self[x][y]
        
        clone = self.clone()
        for y in range(dim[0]):
            for x in range(dim[1]):
                clone[x][y] /= highest

        return clone

    def identity(self, n: int) -> 'Mat':
        """Return the identity Matrix of size n x n."""

        out = Mat(0, 0)
        for i in range(n):
            out.data.append(Vec.onehot(None, i, n))

        return out

class MatrixError(Exception):
    """This occurs when there is a Matrix Error."""

    def __init__(self, msg: str) -> None:
        """A Matrix Error."""

        print(msg)
        quit()

class Net:
    """ A Neural Network Base. 
    
    === Public Attributes ===
    layers: The weight layers for this neural network.
    transfuncs: The list of transfer functions this neural net uses. 
    schematic: The list of the number of neurons in each layer.
    maps: The set of transfer functions and their derivatives.
    """

    layers: List['Mat']
    transfuncs: List[str]
    schematic: List[int]
    maps: Dict[str, Callable]

    def __init__(self, schematic: List[int], transfuncs: List[str]) -> None:
        """Initialize this neural network."""
        self.transfuncs = transfuncs
        self.schematic = schematic
        self.layers = []
        self.maps = {
            'sigmoid': self.__sigmoid,
            'sigmoid_prime': self.__sigmoid_prime,
            'softmax': self.__softmax,
            'softmax_prime': self.__softmax_prime,
            'relu': self.__relu,
            'relu_prime': self.__relu_prime
        }

        for i in range(len(schematic) - 1):
            layer = Mat(schematic[i + 1], schematic[i] + 1)
            layer = layer.randomize(-1.0, 1.0)

            self.layers.append(layer)

    def query(self, input_data: 'Mat') -> 'Mat':
        """Query the Neural Network for the classification of input."""

        cur_layer = input_data.clone()

        for i in range(len(self.layers)):
            bias_vector = Vec([])
            bias_vector = bias_vector.fill(1, cur_layer.dim()[1])

            cur_layer = cur_layer.append_row(bias_vector)
            cur_layer = self.layers[i] * cur_layer
            cur_layer = cur_layer.map(self.maps[self.transfuncs[i]])

        return cur_layer

    def __feed_forward(self, input_data: 'Mat') -> List['Mat']:
        """Feed the input data through the network."""
        layer_data = []

        cur_layer = input_data.clone()
        layer_data.append(cur_layer)

        for i in range(len(self.layers)):
            bias_vector = Vec([])
            bias_vector = bias_vector.fill(1, cur_layer.dim()[1])

            cur_layer = cur_layer.append_row(bias_vector)
            cur_layer = self.layers[i] * cur_layer

            # Save Layer Data Before Mapping
            layer_data.append(cur_layer)

            cur_layer = cur_layer.map(self.maps[self.transfuncs[i]])

        return layer_data

    def train_mse(self, input_data: 'Mat', target_data: 'Mat', eta: float) -> None:
        """Train the neural net with the given input data, target data, and eta."""
        
        layer_data = self.__feed_forward(input_data)
        layer_error = []

        layer_input = layer_data[-1:][0]
        layer_output = layer_input.map(self.maps[self.transfuncs[-1:][0]])
        layer_primed = layer_input.map(self.maps[self.transfuncs[-1:][0] + '_prime'])

        layer_error.append(layer_primed.hadamard(layer_output - target_data))
        for i in range(len(self.layers) - 1, -1, -1):
            # Update input, primed
            layer_input = layer_data[i]
            layer_primed = layer_input.map(self.maps[self.transfuncs[i] + '_prime'])

            # Calculate Errors
            current_error = self.layers[i].transpose()
            current_error = current_error.remove_row(current_error.dim()[0] - 1)
            current_error = current_error * layer_error[-1:][0]
            current_error = current_error.hadamard(layer_primed)

            layer_error.append(current_error)

        layer_weight_updates = []
        for i in range(len(self.layers) - 1, -1, -1):
            # Update Layer Input
            layer_input = layer_data[i]

            bias_vector = Vec([])
            bias_vector = bias_vector.fill(1, layer_input.dim()[1])

            current_weigth_update = layer_error[len(self.layers) - 1 - i]
            
            input_values = layer_input
            input_values = input_values.transpose()
            input_values = input_values.map(self.maps[self.transfuncs[i]])
            input_values = input_values.append_column(bias_vector)
            input_values = input_values * -eta

            current_weigth_update = current_weigth_update * input_values

            layer_weight_updates.append(current_weigth_update)

        for i in range(len(self.layers)):
            self.layers[i] = self.layers[i] + layer_weight_updates[len(self.layers) - 1 - i]

    def train_xent(self, input_data: 'Mat', target_data: 'Mat', eta: float) -> None:
            """Train this Neural Net with the Cross Entropy Loss Function."""

            layer_data = self.__feed_forward(input_data)
            last_input_data = layer_data[len(self.layers)]

            for example_number in range(last_input_data.dim()[1]):
                # Generate Softmax Jacobian
                example = last_input_data[example_number]
                example_target = target_data[example_number]
                
                softmax_out = self.__softmax(example)
                softmax_len = len(softmax_out)

                vert_softmax_n_cols = Mat(0, 0)
                cols = []
                for i in range(softmax_len):
                    cols.append(softmax_out.clone())

                vert_softmax_n_cols.data = cols

                jac_softmax = vert_softmax_n_cols * -1
                jac_softmax = Mat.identity(None, softmax_len) + jac_softmax
                jac_softmax = vert_softmax_n_cols.transpose().hadamard(jac_softmax)

                # Generate Cost Function Gradient
                cost_function_grad = example_target * -1
                for j in range(len(softmax_out)):
                    cost_function_grad[j] /= softmax_out[j]

                cost_function_grad = cost_function_grad.to_mat().transpose()

                # Generate First Error Deltas
                deltas = []
                deltas.insert(0, (cost_function_grad * jac_softmax).transpose())

                # Generate Other Error Deltas
                i = len(self.layers) - 1
                while i >= 1:
                    layer_primed = layer_data[i][example_number].map(self.maps[self.transfuncs[i - 1] + '_prime'])
                    layer_primed = layer_primed.to_mat()

                    delta = self.layers[i].transpose() * deltas[0]
                    delta = delta.remove_row(delta.dim()[0] - 1)
                    delta = delta.hadamard(layer_primed)

                    deltas.insert(0, delta)

                    i -= 1             

                # Generate All Weight Deltas
                weight_deltas = []
                # Generate First Weight Delta w/o running it through the logit.
                layer_input = layer_data[0][example_number]
                layer_input = layer_input.to_mat()
                layer_input = layer_input.append_row(Vec([1.0]))
                layer_input = layer_input.transpose()
                weight_deltas.append(deltas[0] * layer_input)

                i = 1
                while i < len(self.layers):
                    layer_out = layer_data[i][example_number]
                    layer_out = layer_out.map(self.maps[self.transfuncs[i - 1]])
                    layer_out = layer_out.to_mat()
                    layer_out = layer_out.append_row(Vec([1.0]))
                    layer_out = layer_out.transpose()
                    weight_deltas.append(deltas[i] * layer_out)

                    i += 1

                for j in range(len(self.layers)):
                    self.layers[j] = self.layers[j] + (weight_deltas[j] * -eta)                 

    def learn(self, optimizer_type: str, input_data: 'Mat', target_data: 'Mat', eta: float, epochs: int) -> None:
        """Make the Neural Net learn the model."""
        print('Starting Learning...')

        optimizer = None
        if optimizer_type == 'mse':
            optimizer = self.train_mse
        elif optimizer_type == 'xent':
            optimizer = self.train_xent

        if optimizer == None:
            print(optimizer_type + " is not a valid optimizer!")
            exit(1)

        for i in range(epochs):
            optimizer(input_data, target_data, 0.1)
            if (i % int(epochs / 10) == 0 and i != 0):
                print(str(i / epochs * 100) + '% Complete!')

        print('Learning Complete!')

    def save(self, file: str) -> None:
        """Save the properties of this Neural Network."""
        
        with open(file, 'w') as out:
            # Write Schematic
            for i in range(len(self.schematic)):
                out.write(str(self.schematic[i]) + ' ')
            out.write('\n')

            # Write Transfuncs
            for i in range(len(self.transfuncs)):
                out.write(self.transfuncs[i] + ' ')
            out.write('\n')

            # Write Weights
            for i in range(len(self.layers)):
                layer = self.layers[i]
                dim = layer.dim()

                for x in range(dim[1]):
                    for y in range(dim[0]):
                        out.write(str(layer[x][y]) + ' ')
                    out.write('\n')

        print('FILE: ' + file + ' Saved!')

    def load(self, file: str) -> None:
        """Load the Neural Network from the given file."""
        
        with open(file, 'r') as in_file:
            # Load Schematic
            self.schematic = [int(s) for s in in_file.readline().strip().split(' ')]

            # Load Transfuncs
            self.transfuncs = in_file.readline().strip().split(' ')
            
            # Load Weights
            layers = []
            for i in range(len(self.schematic) - 1):
                cols = []
                for x in range(self.schematic[i] + 1):
                    rows = [float(s) for s in in_file.readline().strip().split(' ')]
                    cols.append(Vec(rows))
                
                mat = Mat(0, 0)
                mat.data = cols
                layers.append(mat)

            self.layers = layers

        print('FILE: ' + file + ' Loaded!')

    # Transfer functions and their derivatives.

    def __sigmoid(self, x: 'Vec') -> 'Vec':
        """Map the Vector through the sigmoid function."""

        out = x.clone()
        for i in range(len(out)):
            out.data[i] = 1.0 / (1.0 + exp(-out.data[i]))

        return out

    def __sigmoid_prime(self, x: 'Vec') -> 'Vec':
        """Map the Vector through the sigmoid prime function."""

        out = x.clone()
        for i in range(len(out)):
            out.data[i] = exp(out.data[i]) / ((1.0 + exp(out.data[i]))**2)

        return out

    def __softmax(self, x: 'Vec') -> 'Vec':
        """Map the Vector through the softmax function."""
        
        out = x.clone()
        max_in = -9999
        for i in range(len(x)):
            if out[i] > max_in:
                max_in = out[i]

        exponential_sum = 0
        for i in range(len(x)):
            exponential_sum += exp(out[i] - max_in)

        for i in range(len(x)):
            out[i] = exp(out[i] - max_in) / exponential_sum

        return out

    def __softmax_prime(self, x: 'Vec') -> 'Vec':
        """Map the Vector through the softmax prime function."""

        out = x.clone()
        softmax = self.__softmax(x)

        for i in range(len(out)):
            out[i] = softmax[i] * (1 - softmax[i])

        return out

    def __relu(self, x: 'Vec') -> 'Vec':
        """Map the Vector through the relu function."""

        out = x.clone()
        for i in range(len(out)):
            out.data[i] = log(1 + exp(out.data[i]))

        return out

    def __relu_prime(self, x: 'Vec') -> 'Vec':
        """Map the Vector through the relu prime function."""

        out = x.clone()
        for i in range(len(out)):
            out.data[i] = exp(out.data[i]) / (1 + exp(out.data[i]))

        return out