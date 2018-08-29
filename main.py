#============================================
__author__ = "Sachin Mehta"
__license__ = "MIT"
__maintainer__ = "Sachin Mehta"
# File Description: This file contains the code for training and validation
# ==============================================================================
import loadData as ld
import os
import torch
import pickle
import Model as net
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import Transforms as myTransforms
import DataSet as myDataLoader
import time
from argparse import ArgumentParser
from IOUEval import iouEval
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def val(args, val_loader, model, criterion):
    # switch to evaluation mode
    model.eval()

    iouEvalVal = iouEval(args.classes)

    epoch_loss = []

    total_batches = len(val_loader)
    for i, (inp, inputA, inputB, inputC, target) in enumerate(val_loader):

        start_time = time.time()
        input = torch.cat([inp, inputA, inputB, inputC], 1) # dim-0 is batch

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()

        # If you are using PyTorch > 0.3, then you don't need variable.
        # Instead you can use torch.no_grad(). See Pytorch documentation for more details
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)
        loss = criterion(output, target_var)

        # If you are using PyTorch > 0.3, then you loss.item() instead of loss.data[0]
        epoch_loss.append(loss.data[0])

        time_taken = time.time() - start_time

        # compute the confusion matrix
        iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)

        print('[%d/%d] loss: %.3f time: %.2f' % (i, total_batches, loss.data[0], time_taken))

    average_epoch_loss_val = np.mean(epoch_loss)

    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalVal.getMetric()

    return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU


def train(args, train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    iouEvalTrain = iouEval(args.classes)

    epoch_loss = []

    total_batches = len(train_loader)
    for i, (inp, inputA, inputB, inputC, target) in enumerate(train_loader):
        #continue
        start_time = time.time()
        input = torch.cat([inp, inputA, inputB, inputC], 1) # dim-0 is batch

        if args.onGPU == True:
            input = input.cuda()
            target = target.cuda()

        # If you are using PyTorch > 0.3, then you don't need variable.
        # Instead you can use torch.enable_grad(). See Pytorch documentation for more details
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var) #, output_down, dec_out
        # set the grad to zero
        optimizer.zero_grad()
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # If you are using PyTorch > 0.3, then you loss.item() instead of loss.data[0]
        epoch_loss.append(loss.data[0])
        time_taken = time.time() - start_time

        # compute the confusion matrix
        iouEvalTrain.addBatch(output.max(1)[1].data, target_var.data)
        print('[%d/%d] loss: %.3f time:%.2f' % (
            i, total_batches, loss.data[0], time_taken))

    average_epoch_loss_train = np.mean(epoch_loss)
    overall_acc, per_class_acc, per_class_iu, mIOU = iouEvalTrain.getMetric()
    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def trainValidateSegmentation(args):
    
    print('Data file: ' + str(args.cached_data_file))
    print(args)

    # check if processed data file exists or not
    if not os.path.isfile(args.cached_data_file):
        dataLoader = ld.LoadData(args.data_dir, args.data_dir_val, args.classes, args.cached_data_file)
        data = dataLoader.processData()
        if data is None:
            print('Error while pickling data. Please check.')
            exit(-1)
    else:
        data = pickle.load(open(args.cached_data_file, "rb"))
    print('=> Loading the model')
    model = net.ESPNet(classes=args.classes, channels=args.channels)
    args.savedir = args.savedir + os.sep

    if args.onGPU:
        model = model.cuda()

    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)

    if args.onGPU:
        model = model.cuda()

    if args.visualizeNet:
        import VisualizeGraph as viz
        x = Variable(torch.randn(1, args.channels, args.inDepth, args.inWidth, args.inHeight))

        if args.onGPU:
            x = x.cuda()

        y = model(x, (128, 128, 128)) #, _, _
        g = viz.make_dot(y)
        g.render(args.savedir + os.sep + 'model', view=False)

    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    print('Parameters: ' + str(total_paramters))

    # define optimization criteria
    weight = torch.from_numpy(data['classWeights'])  # convert the numpy array to torch <- Sachin
    print('Class Imbalance Weights')
    print(weight)
    criteria = torch.nn.CrossEntropyLoss(weight)
    if args.onGPU:
        criteria = criteria.cuda()

    # We train at three different resolutions (144x144x144, 96x96x96 and 128x128x128)
    # and validate at one resolution (128x128x128)
    trainDatasetA = myTransforms.Compose([
        myTransforms.MinMaxNormalize(),
        myTransforms.ScaleToFixed(dimA=144, dimB=144, dimC=144),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
    ])

    trainDatasetB = myTransforms.Compose([
        myTransforms.MinMaxNormalize(),
        myTransforms.ScaleToFixed(dimA=96, dimB=96, dimC=96),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
    ])

    trainDatasetC = myTransforms.Compose([
        myTransforms.MinMaxNormalize(),
        myTransforms.ScaleToFixed(dimA=args.inWidth, dimB=args.inHeight, dimC=args.inDepth),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(args.scaleIn),
    ])


    valDataset = myTransforms.Compose([
        myTransforms.MinMaxNormalize(),
        myTransforms.ScaleToFixed(dimA=args.inWidth, dimB=args.inHeight, dimC=args.inDepth),
        myTransforms.ToTensor(args.scaleIn),
        #
    ])

    trainLoaderA = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDatasetA),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False) #disabling pin memory because swap usage is high
    trainLoaderB = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDatasetB),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    trainLoaderC = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDatasetC),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # define the optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr, (0.9, 0.999),
                                     eps=1e-08, weight_decay=2e-4)

    if args.onGPU == True:
        cudnn.benchmark = True

    start_epoch = 0
    stored_loss = 100000000.0
    if args.resume:
        if os.path.isfile(args.resumeLoc):
            print("=> loading checkpoint '{}'".format(args.resumeLoc))
            checkpoint = torch.load(args.resumeLoc)
            start_epoch = checkpoint['epoch']
            stored_loss = checkpoint['stored_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
        logger.flush()
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Arguments: %s" % (str(args)))
        logger.write("\n Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val'))
        logger.flush()

    # reduce the learning rate by 0.5 after every 100 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_loss, gamma=0.5) #40
    best_val_acc = 0

    loader_idxs = [0, 1, 2] # Three loaders at different resolutions are mapped to three indexes
    for epoch in range(start_epoch, args.max_epochs):
        # step the learning rate
        scheduler.step(epoch)
        lr = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print('Running epoch {} with learning rate {:.5f}'.format(epoch, lr))

        if epoch > 0:
            # shuffle the loaders
            np.random.shuffle(loader_idxs)

        for l_id in loader_idxs:
            if l_id == 0:
                train(args, trainLoaderA, model, criteria, optimizer, epoch)
            elif l_id == 1:
                train(args, trainLoaderB, model, criteria, optimizer, epoch)
            else:
                lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr = \
                    train(args, trainLoaderC, model, criteria, optimizer, epoch)

        # evaluate on validation set
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(args, valLoader, model, criteria)

        print('saving checkpoint') ## added
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'iouTr': mIOU_tr,
            'iouVal': mIOU_val,
            'stored_loss' : stored_loss,
        }, args.savedir + '/checkpoint.pth.tar')

        # save the model also
        if mIOU_val >= best_val_acc:
            best_val_acc = mIOU_val 
            torch.save(model.state_dict(), args.savedir + '/best_model.pth')

        with open(args.savedir + 'acc_' + str(epoch) + '.txt', 'w') as log:
            log.write(
                "\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (
                    epoch, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
            log.write('\n')
            log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
            log.write('\n')
            log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
            log.write('\n')
            log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
            log.write('\n')
            log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.6f" % (epoch, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print("Epoch : " + str(epoch) + ' Details')
        print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f" % (
            epoch, lossTr, lossVal, mIOU_tr, mIOU_val))

    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="ESPNet-3D")
    parser.add_argument('--data_dir', default="./data/original_brats18_preprocess/", help='data directory for training set')
    parser.add_argument('--data_dir_val', default="./data/original_brats17_preprocess/", help='data directory for validation set')
    parser.add_argument('--inWidth', type=int, default=128, help='Volume width')
    parser.add_argument('--inHeight', type=int, default=128, help='Volume height')
    parser.add_argument('--inDepth', type=int, default=128, help='Volume depth or channels')
    parser.add_argument('--scaleIn', type=int, default=1, help='Scale the segmentation mask. Not supported')
    parser.add_argument('--max_epochs', type=int, default=500, help='Max. epochs')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to load the data')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='reduce the learning rate by these many epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--savedir', default='./results', help='Location to save the logs/models/etc.')
    parser.add_argument('--visualizeNet', type=bool, default=False, help='Visualize the network')
    parser.add_argument('--resume', type=bool, default=False, help='Resume the training from saved checkpoint')  # Use this flag to load the last checkpoint for training
    parser.add_argument('--resumeLoc', default='./results/checkpoint.pth.tar', help='Location to resume from')
    parser.add_argument('--classes', type=int, default=4, help='Number of segmentation classes')
    parser.add_argument('--cached_data_file', default='brats.p', help='This file caches the file names and other statistics')
    parser.add_argument('--logFile', default='trainValLog.txt')
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.onGPU = True
    else:
        args.onGPU = False

    args.channels = 4 # because 4 modalities. You can think each modality as a single channel (R or G or B) of an RGB image


    #set the seed to 0
    torch.cuda.manual_seed_all(0)

    trainValidateSegmentation(args)
