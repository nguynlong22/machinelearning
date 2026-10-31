from torch import stack, tril
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
# import torch
from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros 
from torch.nn.functional import relu, softmax
from torch import movedim

"""
##################
### QUESTION 1 ###
##################
"""


class PerceptronModel(Module):
    def __init__(self, dimensions):
        super(PerceptronModel, self).__init__()
        # Khởi tạo trọng số là một tensor gồm các giá trị 1, có kích thước (1, dimensions)
        # Bọc trong Parameter để PyTorch và autograder nhận diện đây là biến cần huấn luyện
        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        return self.w

    def forward(self, x):
        # Tính tích vô hướng giữa ma trận đầu vào x và vector trọng số self.w
        # x có kích thước (batch_size, dimensions) và w có kích thước (1, dimensions)
        return tensordot(x, self.w, dims=([1], [1]))

    def get_prediction(self, x):
        score = self(x)
        # Nếu tích vô hướng >= 0 trả về 1, ngược lại trả về -1
        if score.item() >= 0:
            return 1
        else:
            return -1



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        super().__init__()
   

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"


class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"


    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        """ YOUR CODE HERE """



class LanguageIDModel(Module):

    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()

        hidden_size = 128

        # Xử lý ký tự đầu vào:
        self.input_layer = Linear(self.num_chars, hidden_size)

        # Xử lý trạng thái ẩn trước đó:
        self.hidden_layer = Linear(hidden_size, hidden_size)

        # Chuyển trạng thái ẩn thành dự đoán
        self.output_layer = Linear(hidden_size, len(self.languages))


    def forward(self, xs):
        # Xử lý ký tự đầu
        h = relu(self.input_layer(xs[0]))

        # Xử lý các ký tự tiếp theo
        for x in xs[1:]:
            h = relu(self.input_layer(x) + self.hidden_layer(h))

        return self.output_layer(h)



def Convolve(input: tensor, weight: tensor):
    input_h, input_w = input.shape
    weight_h, weight_w = weight.shape
    
    # Tính toán kích thước của ma trận đầu ra (do không dùng padding)
    out_h = input_h - weight_h + 1
    out_w = input_w - weight_w + 1
    
    Output_Tensor = zeros((out_h, out_w))
    
    # Trượt bộ lọc (kernel) qua từng vị trí trên ảnh đầu vào
    for y in range(out_h):
        for x in range(out_w):
            # Trích xuất vùng không gian tương ứng trên input
            sub_tensor = input[y:y+weight_h, x:x+weight_w]
            # Nhân element-wise và tính tổng
            Output_Tensor[y, x] = (sub_tensor * weight).sum()

    return Output_Tensor


class DigitConvolutionalModel(Module):
    def __init__(self):
        super().__init__()
        self.convolution_weights = Parameter(ones((3, 3)))
        
        # Ảnh 28x28 qua filter 3x3 sẽ thành 26x26. 
        # Khi flatten ra mảng 1 chiều: 26 * 26 = 676
        self.layer1 = Linear(676, 128)
        self.layer2 = Linear(128, 10) # 10 class tương ứng với các số 0-9

    def forward(self, x):
        x = x.reshape(len(x), 28, 28)
        x = stack(
            list(map(lambda sample: Convolve(sample, self.convolution_weights), x))
        )
        x = x.flatten(start_dim=1)
        
        # Đưa qua các lớp tuyến tính với hàm kích hoạt ReLU
        x = relu(self.layer1(x))
        x = self.layer2(x)
        return x


class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        """
        All the layers you should use are defined here.

        In order to pass the autograder, make sure each linear layer matches up with their corresponding matrix,
        ie: use self.k_layer to generate the K matrix.
        """
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)

        # Masking part of attention layer
        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        """
        Applies the attention mechanism to input. All necessary layers have
        been defined in __init__()

        In order to apply the causal mask to a given matrix M, you should update
        it as such:

        M = M.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))[0]

        For the softmax activation, it should be applied to the last dimension of the input,
        Take a look at the "dim" argument of torch.nn.functional.softmax to figure out how to do this.
        """
        B, T, C = input.size()

        """YOUR CODE HERE"""

