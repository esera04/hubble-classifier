import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np

def train_model(model: torch.nn.Module, train_data, test_data, epochs, device):
    loss_func = nn.CrossEntropyLoss()
    optimizer = opt.Adam(model.parameters(), lr=0.01)

    model.train()

    for epoch in range(epochs):
        print(f"running epoch {epoch + 1} out of {epochs}")
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_data):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'running loss: {running_loss}')

        if (epoch + 1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                for i in range(20):
                    _, (imgs, labels) = next(enumerate(test_data))
                    imgs = imgs.to(device)
                    output = model(imgs)
                    print(f'Predicted is {np.argmax(output.cpu().detach().numpy(), axis=1)} and actual is {labels.cpu().detach().numpy()}')
                    torch.save(model.state_dict(), f"models/model-epoch-{epoch + 1}.pt")
            model.train()