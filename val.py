
"""

This script contain valiudation code at the time of training

"""
import time

import torch
import torch.utils.data as data_utils

import modules.evaluation as evaluate
from data import custom_collate
from modules import utils
from modules.utils import get_individual_labels

logger = utils.get_logger(__name__)


def val(args, net, val_dataset):
    val_data_loader = data_utils.DataLoader(val_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                            shuffle=False, pin_memory=True, collate_fn=custom_collate)
    args.MODEL_PATH = args.SAVE_ROOT + 'model_{:06d}.pth'.format(args.EVAL_EPOCHS[0])
    logger.info('Loaded model from :: '+args.MODEL_PATH)
    net.load_state_dict(torch.load(args.MODEL_PATH))
    mAP, ap_all, ap_strs = validate(args, net,  val_data_loader, val_dataset, args.EVAL_EPOCHS[0])
    label_types = args.label_types  #+ ['ego_action']
    all_classes = args.all_classes  #+ [args.ego_classes]
    for nlt in range(args.num_label_types+1):
        for ap_str in ap_strs[nlt]:
            logger.info(ap_str)
        ptr_str = '\n{:s} MEANAP:::=> {:0.5f}'.format(label_types[nlt], mAP[nlt])
        logger.info(ptr_str)


def validate(args, net,  val_data_loader, val_dataset, iteration_num):
    """Test a FPN network on an image database."""
    
    iou_thresh = args.IOU_THRESH
    num_samples = len(val_dataset)
    logger.info('Validating at ' + str(iteration_num) + ' number of samples:: '+ str(num_samples))
    
    
    print_time = True
    val_step = 20
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    activation = torch.nn.Sigmoid().cuda()

    # ego_pds = []
    # ego_gts = []

    det_boxes = []
    gt_boxes_all = []

    for nlt in range(args.num_label_types):
        numc = args.num_classes_list[nlt]
        det_boxes.append([[] for _ in range(numc)])
        gt_boxes_all.append([])

    net.eval()
    with torch.no_grad():
        # for val_itr, (images, gt_boxes, gt_targets, ego_labels, batch_counts, img_indexs, wh) in enumerate(val_data_loader):
        for val_itr, (images, gt_boxes, gt_targets, batch_counts, img_indexs, wh, _, _, is_pseudo_labelled) in enumerate(val_data_loader):

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_size = images.size(0)
            
            images = images.cuda(0, non_blocking=True)
            # decoded_boxes, confidence, ego_preds = net(images)
            decoded_boxes, confidence = net(images)
            # ego_preds = activation(ego_preds).cpu().numpy()
            # ego_labels = ego_labels.numpy()
            confidence = activation(confidence)

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                logger.info('Forward Time {:0.3f}'.format(tf-t1))
            
            seq_len = gt_targets.size(1)
            for b in range(batch_size):
                for s in range(seq_len):
                    # if ego_labels[b,s]>-1:
                    #     ego_pds.append(ego_preds[b,s,:])
                    #     ego_gts.append(ego_labels[b,s])
                    
                    width, height = wh[b][0], wh[b][1]
                    gt_boxes_batch = gt_boxes[b, s, :batch_counts[b, s],:].numpy()
                    gt_labels_batch =  gt_targets[b, s, :batch_counts[b, s]].numpy()

                    decoded_boxes_frame = decoded_boxes[b, s].clone()
                    
                    cc = 0 
                    for nlt in range(args.num_label_types):
                        num_c = args.num_classes_list[nlt]
                        tgt_labels = gt_labels_batch[:,cc:cc+num_c]
                        # print(gt_boxes_batch.shape, tgt_labels.shape)
                        frame_gt = get_individual_labels(gt_boxes_batch, tgt_labels)
                        gt_boxes_all[nlt].append(frame_gt)
                        
                        for cl_ind in range(num_c):
                            scores = confidence[b, s, :, cc].clone().squeeze()
                            cc += 1
                            cls_dets = utils.filter_detections(args, scores, decoded_boxes_frame)
                            det_boxes[nlt][cl_ind].append(cls_dets)
                count += 1 

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                logger.info('detections done: {:d}/{:d} time taken {:0.3f}'.format(count, num_samples, te-ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                logger.info('NMS stuff Time {:0.3f}'.format(te - tf))

    logger.info('Evaluating detections for epoch number ' + str(iteration_num))
    mAP, ap_all, ap_strs = evaluate.evaluate(gt_boxes_all, det_boxes, args.all_classes, iou_thresh=iou_thresh)
    # mAP_ego, ap_all_ego, ap_strs_ego = evaluate.evaluate_ego(np.asarray(ego_gts), np.asarray(ego_pds),  args.ego_classes)
    # return mAP + [mAP_ego], ap_all + [ap_all_ego], ap_strs + [ap_strs_ego]
    return mAP, ap_all, ap_strs