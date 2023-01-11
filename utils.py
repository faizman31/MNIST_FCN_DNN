import torch

def load_mnist(is_train=True , flatten=True):
    from torchvision import datasets,transforms
    
    datasets = datasets.MNIST(
        '../data',train=is_train,download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = datasets.data.float() / 255. # min/max scaling
    y = datasets.targets

    if flatten:
        x = x.reshape(x.shape[0],-1)

    return x,y

def split_data(x,y,train_ratio=.8):
    train_cnt = int(x.shape[0] * train_ratio)
    valid_cnt = x.shape[0] - train_cnt

    indices =torch.randperm(x.shape[0])
    x = torch.index_select(x,dim=0,index=indices).split([train_cnt,valid_cnt],dim=0)
    y = torch.index_select(y,dim=0,index=indices).split([train_cnt,valid_cnt],dim=0)

    return x,y


def get_hidden_size(input_size,output_size,n_layers):
    step_size = int((input_size - output_size) / n_layers)

    hidden_sizes=[]
    current_size=input_size

    for i in range(n_layers-1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]

    return hidden_sizes