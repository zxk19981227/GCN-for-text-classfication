
import torch
def train(matrix,feature,model,optim,epoch,device,train_mask,train_label,loss_function):
    model.train()
    optim.zero_grad()
    predict=model(matrix,feature)
    # predict=torch.index_select(predict,0,train_index)
    # print(predict.shape)
    # exit(0)
    loss=loss_function(predict,train_label,train_mask)
    loss.backward()
    optim.step()
    train_label=torch.argmax(train_label,-1)
    accuracy=((torch.argmax(predict,-1)==train_label) & (train_mask.float()>0)).sum().item()/(train_mask>0).sum()
    # print("training set ------------")
    # print(train_index)
    # print(torch.argmax(predict[train_index],-1))
    # print(train_label)
    print("train epoch{} train accuracy {} train loss {}".format(epoch,accuracy,loss))
    return accuracy,loss
def test(matrix,feature,model,optim,epoch,device,train_mask,train_label,loss_function,info):
    model.eval()
    predict=model(matrix,feature)
    # predict=torch.index_select(predict,0,train_index)

    loss=loss_function(predict,train_label,train_mask)
    train_label=torch.argmax(train_label,-1)
    accuracy = ((torch.argmax(predict, -1) == train_label) & (train_mask > 0)).sum().item() / (train_mask > 0).sum()
    # print("test_set ----------------")
    # print(train_index)
    # print(torch.argmax(predict[train_index],-1))
    # print(train_label)
    print(info+" epoch{} accuracy {}  loss {}".format(epoch,accuracy,loss))
    return accuracy,loss

