import torch
import torchvision
import torch.nn as nn
import sys
import os
import parameters
import cls_data_generator
import torch.optim as optim
import seldnet_model
import torch.nn.functional as F
import cls_feature_class
from train_seldnet import test_epoch
from time import gmtime, strftime
from cls_compute_seld_results import ComputeSELDResults


def main(argv):
    task_id = '6'
    params = parameters.get_params(task_id)

    score_obj = ComputeSELDResults(params)

    # Load the ResNet model
    resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')

    model = resnet
    model = torch.nn.Sequential(*(list(model.children())[:-2]))
    # Define the new convolutional layers
    conv_layer_1 = nn.Conv2d(2048, 1024, kernel_size=1)
    conv_layer_2 = nn.Conv2d(1024, 156, kernel_size=1)
    conv_layer_3 = nn.Conv2d(128, 1, kernel_size=1)
    conv_layer_up = nn.Conv2d(1, 156, kernel_size=1)

    # Concatenate the ResNet model with the new convolutional layers
    model = torch.nn.Sequential(model, conv_layer_1, conv_layer_2)

    conv_layer = nn.Conv2d(2048, 1024, kernel_size=1)
    conv_layer_2 = nn.Conv2d(1024, 156, kernel_size=1)
    #model = torch.nn.Sequential(model, conv_layer, conv_layer_2)

    # Freeze all layers except the last two
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the parameters of the last two layers
    for param in conv_layer_1.parameters():
        param.requires_grad = True
    for param in conv_layer_2.parameters():
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

    print('Loading validation dataset:')
    data_gen_val = cls_data_generator.DataGenerator(
        params=params, split=val_splits[0], shuffle=False, per_file=True
    )

    nb_epoch = 2 if params['quick_test'] else params['nb_epochs']

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    criterion = seldnet_model.MSELoss_ADPIT(relative_dist=True, visual_loss=True)

    unique_name = 'ResNet_train_fronzen'
    # Dump results in DCASE output format for calculating final scores
    dcase_output_val_folder = os.path.join(params['dcase_output_dir'],
                                            '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
    cls_feature_class.delete_and_create_folder(dcase_output_val_folder)

    model_name = unique_name + strftime("%Y%m%d%H%M%S", gmtime())

    best_val_epoch = -1
    best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err, best_rel_dist_err = 1., 0., 180., 0., 9999, 999999., 999999.

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
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            nb_train_batches += 1

            #if nb_train_batches % 500 == 0:
            #    print("Iteration ", nb_train_batches, "Loss: ", train_loss/nb_train_batches)

        if epoch_cnt % 10 == 0:
            val_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)
            # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores

            val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(
                dcase_output_val_folder)

            # Save model if F-score is good
            if val_F >= best_F:
                best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr, val_dist_err
                best_rel_dist_err = val_rel_dist_err
                torch.save(model.state_dict(), model_name)

            print('best_val_epoch: ', best_val_epoch, 'best_F, best_LE, best_dist_err, best_rel_dist_err, best_seld_scr' , '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format( best_F, best_LE, best_dist_err, best_rel_dist_err, best_seld_scr), flush=True)

        train_loss /= nb_train_batches
        print("Epoch: ", epoch_cnt, "Training loss: ", train_loss, flush=True)

    torch.save(model.state_dict(), 'final_model_down_to_1_larger.pth')


if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
