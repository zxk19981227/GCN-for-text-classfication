
import torch
def train(matrix,feature,model,optim,epoch,device,train_index,train_label,loss_function):
    model.train()

    predict=model(matrix,feature)
    optim.zero_grad()
    # predict=torch.index_select(predict,0,train_index)
    # print(predict.shape)
    # exit(0)
    loss=loss_function(predict[train_index],train_label)
    loss.backward()
    optim.step()
    accuracy=(torch.argmax(predict[train_index],-1)==train_label).sum().item()/train_index.shape[0]
    # print("training set ------------")
    # print(train_index)
    # print(torch.argmax(predict[train_index],-1))
    # print(train_label)
    print("train epoch{} train accuracy {} train loss {}".format(epoch,accuracy,loss))
    return accuracy,loss
def test(matrix,feature,model,optim,epoch,device,train_index,train_label,loss_function,info):
    model.eval()
    predict=model(matrix,feature)
    # predict=torch.index_select(predict,0,train_index)
    loss=loss_function(predict[train_index],train_label)
    accuracy=(torch.argmax(predict[train_index],-1)==train_label).sum().item()/train_index.shape[0]
    # print("test_set ----------------")
    # print(train_index)
    # print(torch.argmax(predict[train_index],-1))
    # print(train_label)
    print(info+" epoch{} accuracy {}  loss {}".format(epoch,accuracy,loss))
    return accuracy,loss

