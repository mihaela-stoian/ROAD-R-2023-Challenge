
import math
import time

import torch
import torch.utils.data as data_utils

from data import custom_collate
from modules import AverageMeter
from modules import utils
from modules.solver import get_optim
from utils_ssl import compute_req_matrices
from val import validate

logger = utils.get_logger(__name__)


def setup_training(args, net):
    optimizer, scheduler, solver_print_str = get_optim(args, net)
    if args.TENSORBOARD:
        from tensorboardX import SummaryWriter
    source_dir = args.SAVE_ROOT + '/source/'  # where to save the source
    utils.copy_source(source_dir)
    args.START_EPOCH = 1
    if args.RESUME > 0:
        # raise Exception('Not implemented')
        args.START_EPOCH = args.RESUME + 1
        # args.iteration = args.START_EPOCH
        for _ in range(args.RESUME):
            scheduler.step()
        model_file_name = '{:s}/model_{:06d}.pth'.format(args.SAVE_ROOT, args.RESUME)
        optimizer_file_name = '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, args.RESUME)
        # sechdular_file_name = '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, args.START_EPOCH)
        net.load_state_dict(torch.load(model_file_name))
        optimizer.load_state_dict(torch.load(optimizer_file_name))
        logger.info('After loading checkpoint from epoch {:}, the learning rate is {:}'.format(args.RESUME, args.LR))
    if args.TENSORBOARD:
        log_dir = '{:s}/log-lo_tboard-{}-{date:%m-%d-%H-%M-%S}_logic-{logic:s}_req-weight-{weight}'.format(args.log_dir,
                                                                                                           args.MODE,
                                                                                                           date=args.DATETIME_NOW,
                                                                                                           logic=str(
                                                                                                               args.LOGIC),
                                                                                                           weight=args.req_loss_weight)
        # log_dir = '{:s}/log-lo_tboard-{}_logic-{logic:s}_req-weight-{weight}'.format(args.log_dir, args.MODE, logic=str(args.LOGIC), weight=args.req_loss_weight)
        args.sw = SummaryWriter(log_dir)
        logger.info('Created tensorboard log dir ' + log_dir)

    if args.pretrained_model_path is not None and args.RESUME == 0:
        net.load_state_dict(torch.load(args.pretrained_model_path))
        logger.info("Load pretrained model {:}".format(args.pretrained_model_path))

    logger.info(str(net))
    logger.info(solver_print_str)

    logger.info('EXPERIMENT NAME:: ' + args.exp_name)
    logger.info('Training FPN with {} + {} as backbone '.format(args.ARCH, args.MODEL_TYPE))
    return args, optimizer, scheduler


def train(args, net, train_dataset, val_dataset):
    epoch_size = len(train_dataset) // args.BATCH_SIZE
    args.MAX_ITERS = epoch_size #args.MAX_EPOCHS * epoch_size
    args, optimizer, scheduler = setup_training(args, net)

    train_data_loader = data_utils.DataLoader(train_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                              shuffle=True, pin_memory=True, collate_fn=custom_collate, drop_last=True)

    val_data_loader = data_utils.DataLoader(val_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                            shuffle=False, pin_memory=True, collate_fn=custom_collate)

    # TO REMOVE
    # if not args.DEBUG_num_iter:
    #     net.eval()
    #     run_val(args, val_data_loader, val_dataset, net, 0, 0)

    iteration = 0
    for epoch in range(args.START_EPOCH, args.MAX_EPOCHS + 1):
        print('LR at epoch {:} is {:}'.format(epoch, scheduler.get_last_lr()[0]))
        net.train()

        if args.FBN:
            if args.MULTI_GPUS:
                net.module.backbone.apply(utils.set_bn_eval)
            else:
                net.backbone.apply(utils.set_bn_eval)
        iteration = run_train(args, train_data_loader, net, optimizer, epoch, iteration)

        if epoch % args.VAL_STEP == 0 or epoch == args.MAX_EPOCHS:
            net.eval()
            run_val(args, val_data_loader, val_dataset, net, epoch, iteration)

        scheduler.step()

def run_train(args, train_data_loader, net, optimizer, epoch, iteration):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}
    loc_losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}
    cls_losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}
    req_losses = {'gt': AverageMeter(), 'ulb': AverageMeter(), 'all': AverageMeter()}

    torch.cuda.synchronize()
    start = time.perf_counter()

    if args.LOGIC is not None:
        Cplus, Cminus = compute_req_matrices(args)

    # for internel_iter, (images, gt_boxes, gt_labels, ego_labels, counts, img_indexs, wh) in enumerate(train_data_loader):
    for internel_iter, (mix_images, mix_gt_boxes, mix_gt_labels, mix_counts, mix_img_indexs, mix_wh, _, _, mix_is_ulb) in enumerate(train_data_loader):
        if args.DEBUG_num_iter and internel_iter > 22:
            logger.info('DID 5 ITERATIONS IN TRAIN, break.... for debugging only')
            break

        images = mix_images.cuda(0, non_blocking=True)
        gt_boxes = mix_gt_boxes.cuda(0, non_blocking=True)
        gt_labels = mix_gt_labels.cuda(0, non_blocking=True)
        counts = mix_counts.cuda(0, non_blocking=True)
        img_indexs = mix_img_indexs

        loc_loss_dict = {'gt': [], 'ulb': [], 'all': []}
        conf_loss_dict = {'gt': [], 'ulb': [], 'all': []}
        req_loss_dict = {'gt': [], 'ulb': [], 'all': []}

        iteration += 1
        if args.LOGIC is not None:
            Cplus = Cplus.cuda(0, non_blocking=True)
            Cminus = Cminus.cuda(0, non_blocking=True)
        data_time.update(time.perf_counter() - start)


        mix_mask_is_ulb = torch.tensor([all(mix_is_ulb[b]) for b in range(mix_images.shape[0])])

        # forward
        torch.cuda.synchronize()

        # print(images.size(), anchors.size())
        optimizer.zero_grad()
        # pdb.set_trace()

        if args.LOGIC is None:
            (mix_loc_loss, mix_conf_loss), selected_is_ulb = net(images, gt_boxes, gt_labels, counts, img_indexs, is_ulb=mix_mask_is_ulb)
            # Mean over the losses computed on the different GPUs
            loc_loss, conf_loss = mix_loc_loss.mean(), mix_conf_loss.mean()
            loss = loc_loss + conf_loss
        else:
            (mix_loc_loss, mix_conf_loss, mix_req_loss), selected_is_ulb = net(images, gt_boxes, gt_labels, counts, img_indexs, logic=args.LOGIC, Cplus=Cplus, Cminus=Cminus, is_ulb=mix_mask_is_ulb)
            # Mean over the losses computed on the different GPUs
            loc_loss, conf_loss, req_loss = mix_loc_loss.mean(), mix_conf_loss.mean(), mix_req_loss.mean()
            loss = loc_loss + conf_loss + args.req_loss_weight * req_loss

        if args.log_ulb_gt_separately:  # for DataParallel only
            for i, elem in enumerate(selected_is_ulb):
                type_elem = 'ulb' if elem else 'gt'
                loc_loss_dict[type_elem].append(mix_loc_loss[i])  # for DataParallel only
                conf_loss_dict[type_elem].append(mix_conf_loss[i])  # for DataParallel only
                if args.LOGIC is not None:
                    req_loss_dict[type_elem].append(mix_req_loss[i])
        else:
            type_elem = 'all'
            loc_loss_dict[type_elem].append(mix_loc_loss.mean())
            conf_loss_dict[type_elem].append(mix_conf_loss.mean())
            if args.LOGIC is not None:
                req_loss_dict[type_elem].append(mix_req_loss.mean())

        loss.backward()
        optimizer.step()

        for elem in mix_mask_is_ulb.unique():
            if args.log_ulb_gt_separately:  # for DataParallel only
                type_elem = 'ulb' if elem else 'gt'
            else:
                type_elem = 'all'

            loc_loss_dict[type_elem] = torch.tensor(loc_loss_dict[type_elem]).mean()
            conf_loss_dict[type_elem] = torch.tensor(conf_loss_dict[type_elem]).mean()

            loc_loss = loc_loss_dict[type_elem].item()
            conf_loss = conf_loss_dict[type_elem].item()
            if math.isnan(loc_loss) or loc_loss > 300:
                lline = '\n\n\n We got faulty LOCATION loss {} {} {}\n\n\n'.format(loc_loss, conf_loss, type_elem)
                logger.info(lline)
                loc_loss = 20.0
            if math.isnan(conf_loss) or conf_loss > 300:
                lline = '\n\n\n We got faulty CLASSIFICATION loss {} {} {}\n\n\n'.format(loc_loss, conf_loss, type_elem)
                logger.info(lline)
                conf_loss = 20.0

            loc_losses[type_elem].update(loc_loss)
            cls_losses[type_elem].update(conf_loss)
            if args.LOGIC is None:
                losses[type_elem].update(loc_loss + conf_loss)
            else:
                req_loss_dict[type_elem] = torch.tensor(req_loss_dict[type_elem]).mean()
                req_loss = req_loss_dict[type_elem]
                req_losses[type_elem].update(req_loss)
                losses[type_elem].update(loc_loss + conf_loss + req_loss)  # do not multiply by req weight, so exp are comparable

            if internel_iter % args.LOG_STEP == 0 and iteration > args.LOG_START and internel_iter > 0:
                if args.TENSORBOARD:
                    loss_group = dict()
                    loss_group['Classification-'+type_elem] = cls_losses[type_elem].val
                    loss_group['Localisation-'+type_elem] = loc_losses[type_elem].val
                    loss_group['Requirements-'+type_elem] = req_losses[type_elem].val
                    loss_group['Overall-'+type_elem] = losses[type_elem].val
                    args.sw.add_scalars('Losses', loss_group, iteration)
                    args.sw.add_scalars('Losses_ep', loss_group, epoch)

                print_line = 'Iteration [{:d}/{:d}]{:06d}/{:06d} losses for {:} loc-loss {:.2f}({:.2f}) cls-loss {:.2f}({:.2f}) req-loss {:.8f}({:.8f}) ' \
                             'overall-loss {:.2f}({:.2f})'.format(epoch,
                                                                  args.MAX_EPOCHS, internel_iter, args.MAX_ITERS, type_elem,
                                                                  loc_losses[type_elem].val, loc_losses[type_elem].avg,
                                                                  cls_losses[type_elem].val,
                                                                  cls_losses[type_elem].avg, req_losses[type_elem].val,
                                                                  req_losses[type_elem].avg, losses[type_elem].val,
                                                                  losses[type_elem].avg)
                logger.info(print_line)
            if type_elem == 'all':
                break

        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - start)
        start = time.perf_counter()

        if internel_iter % args.LOG_STEP == 0 and iteration > args.LOG_START and internel_iter>0:
            print_line = 'DataTime {:0.2f}({:0.2f}) Timer {:0.2f}({:0.2f})'.format(10 * data_time.val, 10 * data_time.avg, 10 * batch_time.val, 10 * batch_time.avg)
            logger.info(print_line)

            if internel_iter % (args.LOG_STEP*20) == 0:
                logger.info(args.exp_name)
    logger.info('Saving state, epoch:' + str(epoch))
    torch.save(net.state_dict(), '{:s}/model_{:06d}.pth'.format(args.SAVE_ROOT, epoch))
    torch.save(optimizer.state_dict(), '{:s}/optimizer_{:06d}.pth'.format(args.SAVE_ROOT, epoch))

    return iteration


def run_val(args, val_data_loader, val_dataset, net, epoch, iteration):
        torch.cuda.synchronize()
        tvs = time.perf_counter()

        mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, epoch)
        label_types = args.label_types #+ ['ego_action']
        all_classes = args.all_classes #+ [args.ego_classes]
        mAP_group = dict()

        # for nlt in range(args.num_label_types+1):
        for nlt in range(args.num_label_types):
            for ap_str in ap_strs[nlt]:
                logger.info(ap_str)
            ptr_str = '\n{:s} MEANAP:::=> {:0.5f}'.format(label_types[nlt], mAP[nlt])
            logger.info(ptr_str)

            if args.TENSORBOARD:
                mAP_group[label_types[nlt]] = mAP[nlt]
                # args.sw.add_scalar('{:s}mAP'.format(label_types[nlt]), mAP[nlt], iteration)
                class_AP_group = dict()
                for c, ap in enumerate(ap_all[nlt]):
                    class_AP_group[all_classes[nlt][c]] = ap
                args.sw.add_scalars('ClassAP-{:s}'.format(label_types[nlt]), class_AP_group, epoch)

        if args.TENSORBOARD:
            args.sw.add_scalars('mAPs', mAP_group, epoch)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
        logger.info(prt_str)




