
"""
 Testing 
"""

import os
import pickle
import time
import zipfile

import numpy as np
import torch
import torch.utils.data as data_utils

from data import custom_collate
from modules import utils

logger = utils.get_logger(__name__)

def gen_dets(args, net, val_dataset):
    
    net.eval()
    val_data_loader = data_utils.DataLoader(val_dataset, int(args.TEST_BATCH_SIZE), num_workers=args.NUM_WORKERS,
                                            shuffle=False, pin_memory=True, collate_fn=custom_collate)
    for epoch in args.EVAL_EPOCHS:
        args.det_itr = epoch
        logger.info('Testing at ' + str(epoch))
        
        args.det_save_dir = os.path.join(args.SAVE_ROOT, "detections-{it:02d}-{sq:02d}-{n:d}_{subsets:s}/".format(it=epoch, sq=args.TEST_SEQ_LEN, n=int(100*args.GEN_NMS), subsets='-'.join(val_dataset.SUBSETS)))
        logger.info('detection saving dir is :: '+args.det_save_dir)

        args.predictions_file_pkl = os.path.join(args.SAVE_ROOT,
                                                        "pred_detections-{it:02d}-{sq:02d}-{n:d}_{subsets:s}.pkl".format(
                                                            it=epoch,
                                                            sq=args.TEST_SEQ_LEN,
                                                            n=int(100 * args.GEN_NMS), subsets='-'.join(val_dataset.SUBSETS)))
        args.predictions_file_zip = os.path.join(args.SAVE_ROOT,
                                                        "pred_detections-{it:02d}-{sq:02d}-{n:d}_{subsets:s}.zip".format(
                                                            it=epoch,
                                                            sq=args.TEST_SEQ_LEN,
                                                            n=int(100 * args.GEN_NMS), subsets='-'.join(val_dataset.SUBSETS)))
        logger.info('Detection saving pkl file path  :: ' + args.predictions_file_pkl)
        logger.info('Detection saving zip file path  :: ' + args.predictions_file_zip)


        is_all_done = True
        if os.path.isdir(args.det_save_dir):
            for vid, videoname in enumerate(val_dataset.video_list):
                save_dir = '{:s}/{}'.format(args.det_save_dir, videoname)
                if os.path.isdir(save_dir):
                    numf = val_dataset.numf_list[vid]
                    dets_list = [d for d in os.listdir(save_dir) if d.endswith('.pkl')]
                    if numf != len(dets_list):
                        is_all_done = False
                        print('Not done', save_dir, numf, len(dets_list))
                        break 
                else:
                    is_all_done = False
                    break
        else:
            is_all_done = False
            os.makedirs(args.det_save_dir)
        
        if is_all_done:
            print('All done! skipping detection')
            continue
        
        args.MODEL_PATH = args.SAVE_ROOT + 'model_{:06d}.pth'.format(epoch)
        net.load_state_dict(torch.load(args.MODEL_PATH))
        
        logger.info('Finished loading model %d !' % epoch )
        
        torch.cuda.synchronize()
        tt0 = time.perf_counter()

        net.eval()  # switch net to evaluation mode
        txt_saved_detections_file = perform_detection(args, net, val_data_loader, val_dataset, epoch)
        # label_types = [args.label_types[0]]
        # for nlt in range(len(label_types)):
        #     for ap_str in ap_strs[nlt]:
        #         logger.info(ap_str)
        #     ptr_str = '\n{:s} MEANAP:::=> {:0.5f}'.format(label_types[nlt], mAP[nlt])
        # logger.info(ptr_str)

        torch.cuda.synchronize()
        logger.info('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        print('\n Pickle dets file', args.predictions_file_pkl)
        print('\n Zip dets file', args.predictions_file_zip)
    return args.predictions_file_pkl

def convert_pred_bbox_to_gt_size(preds):
    # gt size is 960 x 1280
    preds[:,0] = preds[:,0] / 682.0 * 1280.0
    preds[:,1] = preds[:,1] / 512.0 * 960.0
    preds[:,2] = preds[:,2] / 682.0 * 1280.0
    preds[:,3] = preds[:,3] / 512.0 * 960.0
    return preds


def update_preds_dict(videoname, frame_num, pred, preds_dict):
    # pred is: first four numbers are bbox coordinates, next number is agentness score, then the rest of the 41 numbers are labels scores
    pred = pred['main']
    frame_num = '{:05d}.jpg'.format(frame_num)

    if videoname not in preds_dict:
        preds_dict[videoname] = {}
    if frame_num not in preds_dict[videoname]:
        preds_dict[videoname][frame_num] = []

    pred = convert_pred_bbox_to_gt_size(pred)

    for bbox_pred in pred:
        # bbox_dict = {'bbox': bbox_pred[:4], 'agentness': bbox_pred[4], 'labels': bbox_pred[5:46]}
        bbox_dict = {'bbox': bbox_pred[:4], 'labels': bbox_pred[5:46]}
        preds_dict[videoname][frame_num].append(bbox_dict)

    return preds_dict


def perform_detection(args, net,  val_data_loader, val_dataset, iteration):

    """Test a network on a video database."""

    num_images = len(val_data_loader.dataset) #len(val_dataset)
    print_time = True
    val_step = 50
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    activation = torch.nn.Sigmoid().cuda()
    preds_dict = {}

    # ego_pds = []
    # ego_gts = []

    det_boxes = []
    # gt_boxes_all = []

    for nlt in range(1):
        numc = args.num_classes_list[nlt]
        det_boxes.append([[] for _ in range(numc)])
        # gt_boxes_all.append([])

    nlt = 0
    processed_videos = []
    txt_saved_detections_file = args.det_save_dir + "/log-lo_" + args.det_save_dir.split('/')[-6] + "_ROAD_R_predictions_"
    txt_saved_detections_file += args.MODEL_TYPE+"_logic-"+str(args.LOGIC)+"-"+str(args.req_loss_weight)+"_ag-"+str(args.agentness_th)+".txt"
    f = open(txt_saved_detections_file, 'w')

    with torch.no_grad():
        # for val_itr, (images, gt_boxes, gt_targets, ego_labels, batch_counts, img_indexs, wh) in enumerate(val_data_loader):
        # for val_itr, (images, gt_boxes, gt_targets, batch_counts, img_indexs, wh) in enumerate(val_data_loader):
        for val_itr, (images, gt_boxes, gt_targets, batch_counts, img_indexs, wh, videonames, start_frames, is_pseudo_labelled) in enumerate(val_data_loader):

            if args.DEBUG_num_iter:
                if val_itr > args.DEBUG_num_iter:
                    break

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_size = images.size(0)
            
            images = images.cuda(0, non_blocking=True)
            # decoded_boxes, confidence, ego_preds = net(images)
            decoded_boxes, confidence = net(images)
            # ego_preds = activation(ego_preds).cpu().numpy()
            # ego_labels = ego_labels.numpy()
            confidence = activation(confidence)
            # seq_len = ego_preds.shape[1]
            # seq_len = args.SEQ_LEN
            seq_len = confidence.shape[1]


            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                logger.info('Forward Time {:0.3f}'.format(tf-t1))
            
            for b in range(batch_size):
                index = img_indexs[b]
                annot_info = val_dataset.ids[index]
                video_id, frame_num, step_size = annot_info
                videoname = val_dataset.video_list[video_id]
                save_dir = '{:s}/{}'.format(args.det_save_dir, videoname)
                store_last = False
                if videoname not in processed_videos:
                    processed_videos.append(videoname)
                    store_last = True

                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                count += 1
                for s in range(seq_len):
                    # if ego_labels[b,s]>-1:
                    #     ego_pds.append(ego_preds[b,s,:])
                    #     ego_gts.append(ego_labels[b,s])
                    
                    # gt_boxes_batch = gt_boxes[b, s, :batch_counts[b, s],:].numpy()
                    # gt_labels_batch =  gt_targets[b, s, :batch_counts[b, s]].numpy()
                    decoded_boxes_batch = decoded_boxes[b,s]
                    # frame_gt = utils.get_individual_labels(gt_boxes_batch, gt_labels_batch[:,:1])
                    # gt_boxes_all[0].append(frame_gt)
                    confidence_batch = confidence[b,s]
                    scores = confidence_batch[:, 0].squeeze().clone()
                    cls_dets, save_data = utils.filter_detections_for_dumping(args, scores, decoded_boxes_batch, confidence_batch)
                    det_boxes[0][0].append(cls_dets)

                    #
                    save_data_clean = save_data[:, 0:46]
                    for detts in range(len(save_data_clean)):
                        single_item = save_data_clean[detts]
                        f.write(videonames[b] + ',' + '{:05d}.jpg'.format(start_frames[b] + 1 + s) + ',')
                        for itt in range(single_item.shape[0]):
                            f.write(str(single_item[itt]) + ',')
                        f.write('\n')
                    
                    save_name = '{:s}/{:05d}.pkl'.format(save_dir, frame_num+1)
                    frame_num += step_size
                    save_data = {'main':save_data}

                    if s<seq_len-args.skip_ending or store_last:
                        preds_dict = update_preds_dict(videoname, frame_num, save_data, preds_dict)
                        with open(save_name,'wb') as ff:
                            pickle.dump(save_data, ff)

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                logger.info('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                logger.info('NMS stuff Time {:0.3f}'.format(te - tf))

    # write dict to pkl for EVAL AI submission
    with open(args.predictions_file_pkl, 'wb') as preds_f:
        pickle.dump(preds_dict, preds_f)

    zf = zipfile.ZipFile(args.predictions_file_zip, 'w', zipfile.ZIP_DEFLATED)
    zf.writestr(args.predictions_file_pkl.split('/')[-1], pickle.dumps(preds_dict))

    # mAP, ap_all, ap_strs = evaluate.evaluate(gt_boxes_all, det_boxes, args.all_classes, iou_thresh=args.IOU_THRESH)
    # mAP_ego, ap_all_ego, ap_strs_ego = evaluate.evaluate_ego(np.asarray(ego_gts), np.asarray(ego_pds),  args.ego_classes)
    # return mAP + [mAP_ego], ap_all + [ap_all_ego], ap_strs + [ap_strs_ego]
    # print('mAP:', mAP)
    # print('ap_all:', ap_all)
    # print('ap_strs:', ap_strs)
    # f.close()
    # return mAP , ap_all , ap_strs, txt_saved_detections_file
    return txt_saved_detections_file


