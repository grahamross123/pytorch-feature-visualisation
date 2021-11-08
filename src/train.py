from tqdm import tqdm
import torch

def train(net, optimizer, loss_function, train_X, train_y, batch_size=100):
    for i in tqdm(range(0, len(train_X), batch_size)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
        #print(f"{i}:{i+BATCH_SIZE}")
        batch_X = train_X[i:i+batch_size].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+batch_size]

        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()    # Does the update
    return loss

def validate(net, test_X, test_y):
        correct = 0
        total = 0
        with torch.no_grad():
            for i in tqdm(range(len(test_X))):
                real_class = torch.argmax(test_y[i])
                net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # returns a list, 
                predicted_class = torch.argmax(net_out)

                if predicted_class == real_class:
                    correct += 1
                total += 1
        return round(correct/total, 3)