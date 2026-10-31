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
    model.train()

    learning_rate = 0.002
    batch_size = 64
    epochs = 20

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(epochs):
        for batch in dataloader:
            x = batch['x']
            y = batch['label']

            # Chuyển (batch, length, 47)
            # thành (length, batch, 47)
            x = movedim(x, 0, 1)

            optimizer.zero_grad()

            # Dự đoán
            y_pred = model(x)

            # Tính Loss
            loss = languageid_loss(y_pred, y)

            # Backpropagation
            loss.backward()

            # Cập nhật trọng số
            optimizer.step()




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
