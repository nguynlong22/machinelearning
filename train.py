from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    with no_grad():
        # Tạo DataLoader với batch_size=1 để duyệt qua từng điểm dữ liệu
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        while True:
            mistakes = False
            
            for batch in dataloader:
                x = batch['x']
                y = batch['label']
                
                # Lấy dự đoán từ mô hình
                prediction = model.get_prediction(x)
                
                # Nếu dự đoán sai, tiến hành cập nhật trọng số
                if prediction != y.item():
                    model.w += y * x
                    mistakes = True
            
            # Nếu chạy hết 1 epoch (toàn bộ data) mà không có lỗi, dừng huấn luyện
            if not mistakes:
                break


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """


def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"



def Train_DigitConvolution(model, dataset):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    # Khởi tạo DataLoader. Dùng batch_size nhỏ (ví dụ 16 hoặc 32) do vòng lặp convolve khá chậm
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(10): # Huấn luyện khoảng 10 epoch là đủ để qua ngưỡng 80% với bộ dữ liệu thu gọn
        for batch in dataloader:
            x = batch['x']
            y = batch['label']
            
            optimizer.zero_grad()
            y_pred = model(x)
            loss = digitconvolution_Loss(y_pred, y)
            loss.backward()
            optimizer.step()
