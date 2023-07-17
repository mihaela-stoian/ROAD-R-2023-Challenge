import argparse
import datetime
import random
import sys

import numpy as np
import torch
from torchvision import transforms

import data.transforms as vtf
from data import VideoDataset
from gen_dets import gen_dets
from models.retinanet import build_retinanet
from modules import utils
from train import train
# from tubes import build_eval_tubes
from utils_ssl import save_ulb_indices
from val import val


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def seed_everything(args):
    random.seed(args.MAN_SEED)
    np.random.seed(args.MAN_SEED)
    torch.manual_seed(args.MAN_SEED)
    torch.cuda.manual_seed_all(args.MAN_SEED)


def main():
    parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
    parser.add_argument('TASK', type=int, choices=[1,2], help="if task 1, using only 3 out of 15 videos from the training parition; if task 2, using all 15 videos from the training partition train_1")
    parser.add_argument('DATA_ROOT', help='Location to root directory for dataset reading') # /mnt/mars-fast/datasets/
    parser.add_argument('SAVE_ROOT', help='Location to root directory for saving checkpoint models') # /mnt/mars-alpha/
    parser.add_argument('MODEL_PATH',help='Location to root directory where kinetics pretrained models are stored')
    
    parser.add_argument('--MODE', default='train', choices=['train', 'gen_dets', 'eval_frames', 'val'],
                        help='MODE can be train, gen_dets, eval_frames, val; define SUBSETS accordingly, build tubes')
    # Name of backbone network, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported
    parser.add_argument('--ARCH', default='resnet50', 
                        type=str, help=' base arch')
    parser.add_argument('--MODEL_TYPE', default='RCGRU',
                        type=str, help=' base model')
    parser.add_argument('--ANCHOR_TYPE', default='RETINA',
                        type=str, help='type of anchors to be used in model')
    
    parser.add_argument('--SEQ_LEN', default=8,
                        type=int, help='NUmber of input frames')
    parser.add_argument('--TEST_SEQ_LEN', default=8,
                        type=int, help='NUmber of input frames')
    parser.add_argument('--MIN_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    parser.add_argument('--MAX_SEQ_STEP', default=1,
                        type=int, help='DIFFERENCE of gap between the frames of sequence')
    # if output heads are have shared features or not: 0 is no-shareing else sharining enabled
    # parser.add_argument('--MULIT_SCALE', default=False, type=str2bool,help='perfrom multiscale training')
    parser.add_argument('--HEAD_LAYERS', default=3, 
                        type=int,help='0 mean no shareding more than 0 means shareing')
    parser.add_argument('--NUM_FEATURE_MAPS', default=5, 
                        type=int,help='0 mean no shareding more than 0 means shareing')
    parser.add_argument('--CLS_HEAD_TIME_SIZE', default=3, 
                        type=int, help='Temporal kernel size of classification head')
    parser.add_argument('--REG_HEAD_TIME_SIZE', default=3,
                    type=int, help='Temporal kernel size of regression head')
    
    #  Name of the dataset only voc or coco are supported
    parser.add_argument('--DATASET', default='road', 
                        type=str,help='dataset being used')
    parser.add_argument('--TRAIN_SUBSETS', default='train_1,',
                        type=str,help='Training SUBSETS seprated by ,')
    parser.add_argument('--VAL_SUBSETS', default='val_1',
                        type=str,help='Validation SUBSETS seprated by ,')
    parser.add_argument('--TEST_SUBSETS', default='', 
                        type=str,help='Testing SUBSETS seprated by ,')
    # Input size of image only 600 is supprted at the moment 
    parser.add_argument('--MIN_SIZE', default=512, 
                        type=int, help='Input Size for FPN')
    
    #  data loading argumnets
    parser.add_argument('-b','--BATCH_SIZE', default=4, 
                        type=int, help='Batch size for training')
    parser.add_argument('--TEST_BATCH_SIZE', default=1, 
                        type=int, help='Batch size for testing')
    # Number of worker to load data in parllel
    parser.add_argument('--NUM_WORKERS', '-j', default=8, 
                        type=int, help='Number of workers used in dataloading')
    # optimiser hyperparameters
    parser.add_argument('--OPTIM', default='SGD', 
                        type=str, help='Optimiser type')
    parser.add_argument('--RESUME', default=0, 
                        type=int, help='Resume from given epoch')
    parser.add_argument('--MAX_EPOCHS', default=150,
                        type=int, help='Number of training epoc')
    parser.add_argument('-l','--LR', '--learning-rate', 
                        default=0.0041, type=float, help='initial learning rate')
    parser.add_argument('--MOMENTUM', default=0.9, 
                        type=float, help='momentum')
    parser.add_argument('--MILESTONES', default='130,145',
                        type=str, help='Chnage the lr @')
    parser.add_argument('--GAMMA', default=0.1, 
                        type=float, help='Gamma update for SGD')
    parser.add_argument('--WEIGHT_DECAY', default=1e-4, 
                        type=float, help='Weight decay for SGD')
    
    # Freeze layers or not 
    parser.add_argument('--FBN','--FREEZE_BN', default=True, 
                        type=str2bool, help='freeze bn layers if true or else keep updating bn layers')
    parser.add_argument('--FREEZE_UPTO', default=1, 
                        type=int, help='layer group number in ResNet up to which needs to be frozen')
    
    # Loss function matching threshold
    parser.add_argument('--POSTIVE_THRESHOLD', default=0.5, 
                        type=float, help='Min threshold for Jaccard index for matching')
    parser.add_argument('--NEGTIVE_THRESHOLD', default=0.4,
                        type=float, help='Max threshold Jaccard index for matching')
    # Evaluation hyperparameters
    parser.add_argument('--EVAL_EPOCHS', default='150',
                        type=str, help='eval epochs to test network on these epoch checkpoints usually the last epoch is used')
    parser.add_argument('--VAL_STEP', default=10,
                        type=int, help='Number of training epoch before evaluation')
    parser.add_argument('--IOU_THRESH', default=0.5, 
                        type=float, help='Evaluation threshold for validation and for frame-wise mAP')
    parser.add_argument('--CONF_THRESH', default=0.025, 
                        type=float, help='Confidence threshold for to remove detection below given number')
    parser.add_argument('--NMS_THRESH', default=0.5, 
                        type=float, help='NMS threshold to apply nms at the time of validation')
    parser.add_argument('--TOPK', default=10, 
                        type=int, help='topk detection to keep for evaluation')
    parser.add_argument('--GEN_CONF_THRESH', default=0.025, 
                        type=float, help='Confidence threshold at the time of generation and dumping')
    parser.add_argument('--GEN_TOPK', default=100, 
                        type=int, help='topk at the time of generation')
    parser.add_argument('--GEN_NMS', default=0.5, 
                        type=float, help='NMS at the time of generation')
    parser.add_argument('--CLASSWISE_NMS', default=False, 
                        type=str2bool, help='apply classwise NMS/no tested properly')
    parser.add_argument('--JOINT_4M_MARGINALS', default=False, 
                        type=str2bool, help='generate score of joints i.e. duplexes or triplet by marginals like agents and actions scores')
    
    ## paths hyper parameters
    parser.add_argument('--COMPUTE_PATHS', default=False, 
                        type=str2bool, help=' COMPUTE_PATHS if set true then it overwrite existing ones')
    parser.add_argument('--PATHS_IOUTH', default=0.5,
                        type=float, help='Iou threshold for building paths to limit neighborhood search')
    parser.add_argument('--PATHS_COST_TYPE', default='score',
                        type=str, help='cost function type to use for matching, other options are scoreiou, iou')
    parser.add_argument('--PATHS_JUMP_GAP', default=4,
                        type=int, help='GAP allowed for a tube to be kept alive after no matching detection found')
    parser.add_argument('--PATHS_MIN_LEN', default=6,
                        type=int, help='minimum length of generated path')
    parser.add_argument('--PATHS_MINSCORE', default=0.1,
                        type=float, help='minimum score a path should have over its length')
    
    ## paths hyper parameters
    parser.add_argument('--COMPUTE_TUBES', default=False, type=str2bool, help='if set true then it overwrite existing tubes')
    parser.add_argument('--TUBES_ALPHA', default=0,
                        type=float, help='alpha cost for changeing the label')
    parser.add_argument('--TRIM_METHOD', default='none',
                        type=str, help='other one is indiv which works for UCF24')
    parser.add_argument('--TUBES_TOPK', default=10,
                        type=int, help='Number of labels to assign for a tube')
    parser.add_argument('--TUBES_MINLEN', default=5,
                        type=int, help='minimum length of a tube')
    parser.add_argument('--TUBES_EVAL_THRESHS', default='0.2,0.5',
                        type=str, help='evaluation threshold for checking tube overlap at evaluation time, one can provide as many as one wants')
    # parser.add_argument('--TRAIL_ID', default=0,
    #                     type=int, help='eval TUBES_Thtrshold at evaluation time')
    
    ###
    parser.add_argument('--LOG_START', default=10, 
                        type=int, help='start loging after k steps for text/tensorboard') 
    parser.add_argument('--LOG_STEP', default=10, 
                        type=int, help='Log every k steps for text/tensorboard')
    parser.add_argument('--TENSORBOARD', default=1,
                        type=str2bool, help='Use tensorboard for loss/evalaution visualization')

    # Program arguments
    parser.add_argument('--MAN_SEED', default=123, 
                        type=int, help='manualseed for reproduction')
    parser.add_argument('--MULTI_GPUS', default=True, type=str2bool, help='If  more than 0 then use all visible GPUs by default only one GPU used ') 

    parser.add_argument('--LOGIC', default='None', type=str, help='T-norm to be used in the loss')
    parser.add_argument('--req_loss_weight', default=0.0, type=float, help='weight for the logic-based loss')
    parser.add_argument('--DEBUG_num_iter', default=0, type=int, help='num iterations to run; fast debugging')
    parser.add_argument('--EXP_NAME', default="", type=str, help="Custom experiment name, for resuming and eval, provide the full path to the experiment directory")

    parser.add_argument('--pretrained_model_path', default=None, type=str, help='path to pretrained model; e.g. use this to add warm-up to neurosymbolic training with tnorm based loss')
    parser.add_argument('--unlabelled_proportion', default=0.0, type=float)
    parser.add_argument('--agentness_th', default=0.125, type=float, help='threshold to distinguish foreground vs background boxes when computing t-norm')

    train_dataset = None
    ulb_train_dataset = None

    ## Parse arguments
    args = parser.parse_args()
    args.DATETIME_NOW = datetime.datetime.now()

    if args.TASK == 1:
        args.labelled_videos = "2014-07-14-14-49-50_stereo_centre_01,2015-02-03-19-43-11_stereo_centre_04,2015-02-24-12-32-19_stereo_centre_04"
    elif args.TASK == 2:
        args.labelled_videos = "2014-06-25-16-45-34_stereo_centre_02,2014-07-14-14-49-50_stereo_centre_01," \
            "2014-07-14-15-42-55_stereo_centre_03,2014-08-08-13-15-11_stereo_centre_01,2014-08-11-10-59-18_stereo_centre_02," \
            "2014-11-14-16-34-33_stereo_centre_06,2014-11-18-13-20-12_stereo_centre_05,2014-11-21-16-07-03_stereo_centre_01," \
            "2014-12-09-13-21-02_stereo_centre_01,2015-02-03-08-45-10_stereo_centre_02,2015-02-03-19-43-11_stereo_centre_04," \
            "2015-02-06-13-57-16_stereo_centre_02,2015-02-13-09-16-26_stereo_centre_05,2015-02-24-12-32-19_stereo_centre_04," \
            "2015-03-03-11-31-36_stereo_centre_01"
    args.labelled_videos = args.labelled_videos.split(',')

    if args.LOGIC.lower() == 'none':
        args.LOGIC = None

    args = utils.set_args(args) # set directories and SUBSETS of datasets
    args.MULTI_GPUS = False if args.BATCH_SIZE == 1 else args.MULTI_GPUS
    args.log_ulb_gt_separately = args.MULTI_GPUS and (torch.cuda.device_count() == args.BATCH_SIZE) # if number of gpus and batch size are equal, the losses for the unlabelled (ulb) and labelled (gt) samples are logged separately in tensorboard
    ## set random seeds and global settings
    seed_everything(args)
    torch.set_default_tensor_type('torch.FloatTensor')

    args = utils.create_exp_name(args)

    utils.setup_logger(args)
    logger = utils.get_logger(__name__)
    logger.info(sys.version)

    if args.MODE == 'train':
        args.TEST_SEQ_LEN = args.SEQ_LEN
    else:
        args.SEQ_LEN = args.TEST_SEQ_LEN

    if args.MODE in ['train','val']:
        # args.CONF_THRESH = 0.05
        args.SUBSETS = args.TRAIN_SUBSETS
        train_transform = transforms.Compose([
                            vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
                            vtf.ToTensorStack(),
                            vtf.Normalize(mean=args.MEANS, std=args.STDS)])
        
        # train_skip_step = args.SEQ_LEN
        # if args.SEQ_LEN>4 and args.SEQ_LEN<=10:
        #     train_skip_step = args.SEQ_LEN-2
        if args.SEQ_LEN>10:
            train_skip_step = args.SEQ_LEN + (args.MAX_SEQ_STEP - 1) * 2 - 2
        else:
            train_skip_step = args.SEQ_LEN 

        full_train_dataset = VideoDataset(args, train=True, skip_step=train_skip_step, transform=train_transform)
        logger.info('Done Loading Train Dataset')

        train_dataset, ulb_train_dataset = save_ulb_indices(args, full_train_dataset)

        ## For validation set
        full_test = False
        args.SUBSETS = args.VAL_SUBSETS
        skip_step = args.SEQ_LEN*8
    else:
        args.SEQ_LEN = args.TEST_SEQ_LEN
        args.MAX_SEQ_STEP = 1
        args.SUBSETS = args.TEST_SUBSETS
        full_test = True #args.MODE != 'train'
        args.skip_beggning = 0
        args.skip_ending = 0
        if args.MODEL_TYPE == 'I3D':
            args.skip_beggning = 2
            args.skip_ending = 2
        elif args.MODEL_TYPE != 'C2D':
            args.skip_beggning = 2

        skip_step = args.SEQ_LEN - args.skip_beggning


    val_transform = transforms.Compose([ 
                        vtf.ResizeClip(args.MIN_SIZE, args.MAX_SIZE),
                        vtf.ToTensorStack(),
                        vtf.Normalize(mean=args.MEANS,std=args.STDS)])
    

    val_dataset = VideoDataset(args, train=False, transform=val_transform, skip_step=skip_step, full_test=full_test)
    logger.info('Done Loading Dataset Validation Dataset')

    args.num_classes =  val_dataset.num_classes
    # one for objectness
    args.label_types = val_dataset.label_types
    args.num_label_types = val_dataset.num_label_types
    args.all_classes = val_dataset.all_classes
    args.num_classes_list = val_dataset.num_classes_list
    # args.num_ego_classes = val_dataset.num_ego_classes
    # args.ego_classes = val_dataset.ego_classes
    args.head_size = 256

    if args.MODE in ['train', 'val','gen_dets']:
        net = build_retinanet(args).cuda()
        if args.MULTI_GPUS:
            logger.info('\nLets do dataparallel\n')
            net = torch.nn.DataParallel(net)

    for arg in sorted(vars(args)):
        logger.info(str(arg)+': '+str(getattr(args, arg)))
    
    if args.MODE == 'train':
        if args.FBN:
            if args.MULTI_GPUS:
                net.module.backbone.apply(utils.set_bn_eval)
            else:
                net.backbone.apply(utils.set_bn_eval)
        mixed_train_dataset = train_dataset.__add__(ulb_train_dataset)
        train(args, net, mixed_train_dataset, val_dataset)
        print('\n', args.SAVE_ROOT)
    elif args.MODE == 'val':
        val(args, net, val_dataset)
    elif args.MODE == 'gen_dets':
        preds_pkl_filename = gen_dets(args, net, val_dataset)
        # for task 2, please follow the jupyter instructions to postprocess your predictions found at preds_pkl_filename

if __name__ == "__main__":
    main()
