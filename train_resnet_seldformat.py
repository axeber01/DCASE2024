import torch
import torchvision
import torch.nn as nn
import sys
import parameters
import cls_data_generator
import torch.optim as optim
import seldnet_model
import torch.nn.functional as F


def main(argv):
    task_id = '6'
    params = parameters.get_params(task_id)

    # Load the ResNet model
    resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')

    model = resnet
    model = torch.nn.Sequential(*(list(model.children())[:-2]))
    conv_layer = nn.Conv2d(2048, 156, kernel_size=1)

    # Concatenate the ResNet model with the convolutional layer
    model = torch.nn.Sequential(model, conv_layer)

    # Ensure parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    torch.cuda.empty_cache()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    # Ensure parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    test_splits = [[4]]
    val_splits = [[4]]
    train_splits = [[3]]

    data_gen_train = cls_data_generator.DataGenerator(
        params=params, split=train_splits[0]
    )

    nb_epoch = 2 if params['quick_test'] else params['nb_epochs']

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = seldnet_model.MSELoss_ADPIT()

    for epoch_cnt in range(nb_epoch):
        nb_train_batches, train_loss = 0, 0.
        model.train()
        for values in data_gen_train.generate():
            _, _, frame, target = values
            frame, target = torch.tensor(frame).to(device).float(), torch.tensor(target).to(device).float()
            # print("frame target: ", frame.shape, target.shape)
            x = frame
            x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
            x = x.permute(0, 3, 1, 2)
            # print("X: ", x.shape)

            optimizer.zero_grad()

            outputs = model(x)
            # Perform global average pooling
            outputs = F.adaptive_avg_pool2d(outputs, (1, 1))
            outputs = outputs.view(outputs.size(0), -1)
            outputs = outputs.view(params['batch_size'], params['label_sequence_length'], -1)
            # print("Outputs: ", outputs.shape)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            nb_train_batches += 1

            if nb_train_batches % 500 == 0:
                print("Iteration ", nb_train_batches, "Loss: ", train_loss/nb_train_batches)

        train_loss /= nb_train_batches
        print("Training loss: ", train_loss)

    torch.save(model.state_dict(), 'final_model.pth')


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
