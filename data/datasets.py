
"""

Target is in xmin, ymin, xmax, ymax, label
coordinates are in range of [0, 1] normlised height and width

"""

import json
import os
import pdb
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils as tutils
from PIL import ImageFile
from torch.utils.data import random_split

from .transforms import get_clip_list_resized

ImageFile.LOAD_TRUNCATED_IMAGES =   True
from PIL import Image
from modules.tube_helper import make_gt_tube
from modules import utils
from random import shuffle

logger = utils.get_logger(__name__)


def split_train_dataset(dataset, unlabelled_proportion):
    # this function partitions the dataset into labelled and unlabelled subsets,
    # where the proportion of the unlabelled examples can be changed using subset2_proportion
    # (e.g. if only 10% unlabelled samples (from available data) are needed for training, use subset2_proportion=0.1)
    indices1 = dataset.labelled_ids
    indices2 = dataset.unlabelled_ids

    length_subset2 = int(len(dataset) * unlabelled_proportion)
    # select a subset of the unlabelled indices
    indices2 = indices2[:length_subset2]

    subset1 = torch.utils.data.Subset(dataset, indices1)
    subset2 = torch.utils.data.Subset(dataset, indices2)

    dataset.lbl_indices = subset1.indices
    dataset.ulb_indices = subset2.indices
    return subset1, subset2



def get_box(box, counts):
    box = box.astype(np.float32) - 1
    box[2] += box[0]  #convert width to xmax
    box[3] += box[1]  #converst height to ymax
    for bi in range(4):
        scale = 320 if bi % 2 == 0 else 240
        box[bi] /= scale
        assert 0<=box[bi]<=1.01, box
        # if add_one ==0:
        box[bi] = min(1.0, max(0, box[bi]))
        if counts is None:
            box[bi] = box[bi]*682 if bi % 2 == 0 else box[bi]*512

    return box, counts

def get_frame_level_annos_ucf24(annotations, numf, num_classes, counts=None):
    frame_level_annos = [ {'labeled':True,'ego_label':0,'boxes':[],'labels':[]} for _ in range(numf)]
    add_one = 1
    # if num_classes == 24:
    # add_one = 0
    for tubeid, tube in enumerate(annotations):
    # print('numf00', numf, tube['sf'], tube['ef'])
        for frame_index, frame_num in enumerate(np.arange(tube['sf'], tube['ef'], 1)): # start of the tube to end frame of the tube
            label = tube['label']
            # assert action_id == label, 'Tube label and video label should be same'
            box, counts = get_box(tube['boxes'][frame_index, :].copy(), counts)  # get the box as an array
            frame_level_annos[frame_num]['boxes'].append(box)
            box_labels = np.zeros(num_classes)
            # if add_one == 1:
            box_labels[0] = 1 
            box_labels[label+add_one] = 1
            frame_level_annos[frame_num]['labels'].append(box_labels)
            # frame_level_annos[frame_num]['ego_label'] = label+1
            # frame_level_annos[frame_index]['ego_label'][] = 1
            if counts is not None:
                counts[0,0] += 1
                counts[label,1] += 1
        
    return frame_level_annos, counts


def get_filtered_tubes_ucf24(annotations):
    filtered_tubes = []
    for tubeid, tube in enumerate(annotations):
        frames = []
        boxes = []
        label = tube['label']
        count = 0
        for frame_index, frame_num in enumerate(np.arange(tube['sf'], tube['ef'], 1)):
            frames.append(frame_num+1)
            box, _ = get_box(tube['boxes'][frame_index, :].copy(), None)
            boxes.append(box)
            count += 1
        assert count == tube['boxes'].shape[0], 'numb: {} count ={}'.format(tube['boxes'].shape[0], count)
        temp_tube = make_gt_tube(frames, boxes, label)
        filtered_tubes.append(temp_tube)
    return filtered_tubes


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def filter_labels(ids, all_labels, used_labels):
    """Filter the used ids"""
    used_ids = []
    for id in ids:
        label = all_labels[id]
        if label in used_labels:
            used_ids.append(used_labels.index(label))
    
    return used_ids


def get_gt_video_list(anno_file, SUBSETS):
    """Get video list form ground truth videos used in subset 
    and their ground truth tubes """

    with open(anno_file, 'r') as fff:
        final_annots = json.load(fff)

    video_list = []
    for videoname in final_annots['db']:
        if is_part_of_subsets(final_annots['db'][videoname]['split_ids'], SUBSETS):
            video_list.append(videoname)

    return video_list


def get_filtered_tubes(label_key, final_annots, videoname):
    
    key_tubes = final_annots['db'][videoname][label_key]
    all_labels = final_annots['all_'+label_key.replace('tubes','labels')]
    labels = final_annots[label_key.replace('tubes','labels')]
    filtered_tubes = []
    for _ , tube in key_tubes.items():
        label_id = tube['label_id']
        label = all_labels[label_id]
        if label in labels:
            new_label_id = labels.index(label)
            # temp_tube = GtTube(new_label_id)
            frames = []
            boxes = []
            if 'annos' in tube.keys():
                for fn, anno_id in tube['annos'].items():
                    frames.append(int(fn))
                    anno = final_annots['db'][videoname]['frames'][fn]['annos'][anno_id]
                    box = anno['box'].copy()
                    for bi in range(4):
                        assert 0<=box[bi]<=1.01, box
                        box[bi] = min(1.0, max(0, box[bi]))
                        box[bi] = box[bi]*682 if bi % 2 == 0 else box[bi]*512
                    boxes.append(box)
            else:
                for fn in tube['frames']:
                    frames.append(int(fn))

            temp_tube = make_gt_tube(frames, boxes, new_label_id)
            filtered_tubes.append(temp_tube)
            
    return filtered_tubes


def get_filtered_frames(label_key, final_annots, videoname, filtered_gts):
    
    frames = final_annots['db'][videoname]['frames']
    if label_key == 'agent_ness':
        all_labels = []
        labels = []
    else:
        all_labels = final_annots['all_'+label_key+'_labels']
        labels = final_annots[label_key+'_labels']
    
    for frame_id , frame in frames.items():
        frame_name = '{:05d}'.format(int(frame_id))
        if frame['annotated']>0:
            all_boxes = []
            if 'annos' in frame:
                frame_annos = frame['annos']
                for key in frame_annos:
                    anno = frame_annos[key]
                    box = np.asarray(anno['box'].copy())
                    for bi in range(4):
                        assert 0<=box[bi]<=1.01, box
                        box[bi] = min(1.0, max(0, box[bi]))
                        # box[bi] = box[bi]*682 if bi % 2 == 0 else box[bi]*512
                        box[bi] = box[bi]*1280.0 if bi % 2 == 0 else box[bi]*960.0
                    if label_key == 'agent_ness':
                        filtered_ids = [0]
                    else:
                        filtered_ids = filter_labels(anno[label_key+'_ids'], all_labels, labels)

                    if len(filtered_ids)>0:
                        all_boxes.append([box, filtered_ids])
                
            filtered_gts[videoname+frame_name] = all_boxes
            
    return filtered_gts

def get_av_actions(final_annots, videoname):
    label_key = 'av_action'
    frames = final_annots['db'][videoname]['frames']
    all_labels = final_annots['all_'+label_key+'_labels']
    labels = final_annots[label_key+'_labels']
    
    filtered_gts = {}
    for frame_id , frame in frames.items():
        frame_name = '{:05d}'.format(int(frame_id))
        if frame['annotated']>0:
            gts = filter_labels(frame[label_key+'_ids'], all_labels, labels)
            filtered_gts[videoname+frame_name] = gts
            
    return filtered_gts

def get_video_tubes(final_annots, videoname):
    
    tubes = {}
    for key in final_annots['db'][videoname].keys():
        if key.endswith('tubes'):
            filtered_tubes = get_filtered_tubes(key, final_annots, videoname)
            tubes[key] = filtered_tubes
    
    return tubes


def is_part_of_subsets(split_ids, SUBSETS):
    
    is_it = False
    for subset in SUBSETS:
        if subset in split_ids:
            is_it = True
    
    return is_it


class VideoDataset(tutils.data.Dataset):
    """
    ROAD Detection dataset class for pytorch dataloader
    """

    def __init__(self, args, train=True, input_type='rgb', transform=None, 
                skip_step=1, full_test=False):

        self.ANCHOR_TYPE =  args.ANCHOR_TYPE 
        self.DATASET = args.DATASET
        self.SUBSETS = args.SUBSETS
        self.SEQ_LEN = args.SEQ_LEN
        self.BATCH_SIZE = args.BATCH_SIZE
        self.MIN_SEQ_STEP = args.MIN_SEQ_STEP
        self.MAX_SEQ_STEP = args.MAX_SEQ_STEP
        self.labelled_videos = args.labelled_videos

        # self.MULIT_SCALE = args.MULIT_SCALE
        self.full_test = full_test
        self.skip_step = skip_step #max(skip_step, self.SEQ_LEN*self.MIN_SEQ_STEP/2)
        self.num_steps = max(1, int(self.MAX_SEQ_STEP - self.MIN_SEQ_STEP + 1 )//2)
        # self.input_type = input_type
        self.input_type = input_type+'-images'
        self.train = train
        self.root = args.DATA_ROOT + args.DATASET + '/'
        self.MODE = args.MODE
        
        self._imgpath = os.path.join(self.root, self.input_type)
        # self.image_sets = image_sets
        self.transform = transform
        self.ids = list()
        if self.DATASET == 'road':
            self._make_lists_road()  
        elif self.DATASET == 'ucf24':
            self._make_lists_ucf24() 
        self.num_label_types = len(self.label_types)


    def _make_lists_ucf24(self):

        self.anno_file  = os.path.join(self.root, 'pyannot_with_class_names.pkl')
        

        with open(self.anno_file,'rb') as fff:
            final_annots = pickle.load(fff)
        
        database = final_annots['db']
        self.trainvideos = final_annots['trainvideos']
        ucf_classes = final_annots['classes']
        self.label_types =  ['action_ness', 'action'] #
        
        self.num_classes_list = [1, 24]
        self.num_classes = 25 # one for action_ness
        
        # self.ego_classes = ['Non_action']  +  ucf_classes
        # self.num_ego_classes = len(self.ego_classes)
        
        counts = np.zeros((24, 2), dtype=np.int32)
    
        ratios = [1.0, 1.1, 1.1, 0.9, 1.1, 0.8, 0.7, 0.8, 1.1, 1.4, 1.0, 0.8, 0.7, 1.2, 1.0, 0.8, 0.7, 1.2, 1.2, 1.0, 0.9]
    
        self.video_list = []
        self.numf_list = []
        
        frame_level_list = []

        # default_ego_label = np.zeros(self.num_ego_classes)
        # default_ego_label[0] = 1
        total_labeled_frame = 0
        total_num_frames = 0
        for videoname in sorted(database.keys()):
            
            is_part = 1
            if 'train' in self.SUBSETS and videoname not in self.trainvideos:
                continue
            elif 'test' in self.SUBSETS and videoname in self.trainvideos:
                continue
            # print(database[videoname].keys())
            action_id = database[videoname]['label']
            annotations = database[videoname]['annotations']
            
            numf = database[videoname]['numf']
            self.numf_list.append(numf)
            self.video_list.append(videoname)
            
            # frames = database[videoname]['frames']
            
            frame_level_annos, counts = get_frame_level_annos_ucf24(annotations, numf, self.num_classes, counts)

            frames_with_boxes = 0
            for frame_index in range(numf): #frame_level_annos:
                if len(frame_level_annos[frame_index]['labels'])>0:
                    frames_with_boxes += 1
                frame_level_annos[frame_index]['labels'] = np.asarray(frame_level_annos[frame_index]['labels'], dtype=np.float32)
                frame_level_annos[frame_index]['boxes'] = np.asarray(frame_level_annos[frame_index]['boxes'], dtype=np.float32)

            total_labeled_frame += frames_with_boxes
            total_num_frames += numf

            # logger.info('Frames with Boxes are {:d} out of {:d} in {:s}'.format(frames_with_boxes, numf, videoname))
            frame_level_list.append(frame_level_annos)  
            ## make ids
            start_frames = [ f for f in range(numf-self.MIN_SEQ_STEP*self.SEQ_LEN, -1,  -self.skip_step)]
            
            if self.full_test and 0 not in start_frames:
                start_frames.append(0)
            # logger.info('number of start frames: '+ str(len(start_frames)))
            for frame_num in start_frames:
                step_list = [s for s in range(self.MIN_SEQ_STEP, self.MAX_SEQ_STEP+1) if numf-s*self.SEQ_LEN>=frame_num]
                shuffle(step_list)
                # print(len(step_list), self.num_steps)
                for s in range(min(self.num_steps, len(step_list))):
                    video_id = self.video_list.index(videoname)
                    self.ids.append([video_id, frame_num ,step_list[s]])

        logger.info('Labeled frames {:d}/{:d}'.format(total_labeled_frame, total_num_frames))
        # pdb.set_trace()
        ptrstr = '\n'
        self.frame_level_list = frame_level_list
        self.all_classes = [['action_ness'], ucf_classes.copy()]
        for k, name in enumerate(self.label_types):
            labels = self.all_classes[k]
            # self.num_classes_list.append(len(labels))
            for c, cls_ in enumerate(labels): # just to see the distribution of train and test sets
                ptrstr += '-'.join(self.SUBSETS) + ' {:05d} label: ind={:02d} name:{:s}\n'.format(
                                                counts[c,k] , c, cls_)
        
        ptrstr += 'Number of ids are {:d}\n'.format(len(self.ids))
        ptrstr += 'Labeled frames {:d}/{:d}'.format(total_labeled_frame, total_num_frames)
        self.childs = {}
        self.num_videos = len(self.video_list)
        self.print_str = ptrstr
        
        
    def _make_lists_road(self):
        if ('train' in self.MODE) or ('test' not in self.SUBSETS):
            self.anno_file = os.path.join(self.root, 'road_trainval_v1.0.json')
        else:
            self.anno_file = os.path.join(self.root, 'blank_road_test_v1.0.json')

        with open(self.anno_file,'r') as fff:
            final_annots = json.load(fff)
        
        database = final_annots['db']
        
        self.label_types = ['agent', 'action', 'loc']  # final_annots['label_types'] #['agent', 'action', 'loc', 'duplex', 'triplet'] #

        num_label_type = len(self.label_types)  #5
        self.num_classes = 1  ## one for presence
        self.num_classes_list = [1]
        for name in self.label_types: 
            logger.info('Number of {:s}: all :: {:d} to use: {:d}'.format(name, 
                len(final_annots['all_'+name+'_labels']),len(final_annots[name+'_labels'])))
            numc = len(final_annots[name+'_labels'])
            self.num_classes_list.append(numc)
            self.num_classes += numc
        
        # self.ego_classes = final_annots['av_action_labels']
        # self.num_ego_classes = len(self.ego_classes)
        
        # counts = np.zeros((len(final_annots[self.label_types[-1] + '_labels']), num_label_type), dtype=np.int32)
        counts = np.zeros((19, num_label_type), dtype=np.int32)

        self.video_list = []
        self.numf_list = []
        self.labelled_ids = []
        self.unlabelled_ids = []
        ids_counter = 0
        frame_level_list = []

        for videoname in sorted(database.keys()):
            
            if not is_part_of_subsets(final_annots['db'][videoname]['split_ids'], self.SUBSETS):
                continue
            
            numf = database[videoname]['numf']
            self.numf_list.append(numf)
            self.video_list.append(videoname)
            
            frames = database[videoname]['frames']
            frame_level_annos = [ {'labeled':False,'ego_label':-1,'boxes':np.asarray([]),'labels':np.asarray([])} for _ in range(numf)]

            frame_nums = [int(f) for f in frames.keys()]
            frames_with_boxes = 0
            for frame_num in sorted(frame_nums): #loop from start to last possible frame which can make a legit sequence
                frame_id = str(frame_num)
                if frame_id in frames.keys() and frames[frame_id]['annotated']>0:
                    
                    frame_index = frame_num-1  
                    frame_level_annos[frame_index]['labeled'] = True 
                    # frame_level_annos[frame_index]['ego_label'] = frames[frame_id]['av_action_ids'][0]
                    
                    frame = frames[frame_id]
                    if 'annos' not in frame.keys():
                        frame = {'annos':{}}
                    
                    all_boxes = []
                    all_labels = []
                    frame_annos = frame['annos']
                    for key in frame_annos:
                        width, height = frame['width'], frame['height']
                        anno = frame_annos[key]
                        box = anno['box']
                        
                        assert box[0]<box[2] and box[1]<box[3], box
                        assert width==1280 and height==960, (width, height, box)

                        for bi in range(4):
                            assert 0<=box[bi]<=1.01, box
                            box[bi] = min(1.0, max(0, box[bi]))
                        
                        all_boxes.append(box)
                        box_labels = np.zeros(self.num_classes)
                        list_box_labels = []
                        cc = 1
                        for idx, name in enumerate(self.label_types):
                            filtered_ids = filter_labels(anno[name+'_ids'], final_annots['all_'+name+'_labels'], final_annots[name+'_labels'])
                            list_box_labels.append(filtered_ids)
                            for fid in filtered_ids:
                                box_labels[fid+cc] = 1
                                box_labels[0] = 1
                            cc += self.num_classes_list[idx+1]

                        all_labels.append(box_labels)

                        # for box_labels in all_labels:
                        for k, bls in enumerate(list_box_labels):
                            for l in bls:
                                counts[l, k] += 1 

                    all_labels = np.asarray(all_labels, dtype=np.float32)
                    all_boxes = np.asarray(all_boxes, dtype=np.float32)

                    if all_boxes.shape[0]>0:
                        frames_with_boxes += 1    
                    frame_level_annos[frame_index]['labels'] = all_labels
                    frame_level_annos[frame_index]['boxes'] = all_boxes

            logger.info('Frames with Boxes are {:d} out of {:d} in {:s}'.format(frames_with_boxes, numf, videoname))
            frame_level_list.append(frame_level_annos)  

            ## make ids
            start_frames = [ f for f in range(numf-self.MIN_SEQ_STEP*self.SEQ_LEN, -1,  -self.skip_step)]
            if self.full_test and 0 not in start_frames:
                start_frames.append(0)
            logger.info('number of start frames: '+ str(len(start_frames)))
            for frame_num in start_frames:
                step_list = [s for s in range(self.MIN_SEQ_STEP, self.MAX_SEQ_STEP+1) if numf-s*self.SEQ_LEN>=frame_num]
                shuffle(step_list)
                # print(len(step_list), self.num_steps)
                for s in range(min(self.num_steps, len(step_list))):
                    video_id = self.video_list.index(videoname)
                    self.ids.append([video_id, frame_num ,step_list[s]])
                    if videoname in self.labelled_videos:
                        self.labelled_ids.append(ids_counter)
                    else:
                        self.unlabelled_ids.append(ids_counter)
                    ids_counter += 1

        # pdb.set_trace()
        ptrstr = ''
        self.frame_level_list = frame_level_list
        self.all_classes = [['agent_ness']]
        for k, name in enumerate(self.label_types):
            labels = final_annots[name+'_labels']
            self.all_classes.append(labels)
            # self.num_classes_list.append(len(labels))
            for c, cls_ in enumerate(labels): # just to see the distribution of train and test sets
                ptrstr += '-'.join(self.SUBSETS) + ' {:05d} label: ind={:02d} name:{:s}\n'.format(
                                                counts[c,k] , c, cls_)
        
        ptrstr += 'Number of ids are {:d}\n'.format(len(self.ids))

        self.label_types = ['agent_ness'] + self.label_types
        self.childs = {'duplex_childs':final_annots['duplex_childs'], 'triplet_childs':final_annots['triplet_childs']}
        self.num_videos = len(self.video_list)
        self.print_str = ptrstr
        assert self.__len__() == len(self.labelled_ids) + len(self.unlabelled_ids), "self.__len__() is different from len(self.labelled_ids) + len(self.unlabelled_ids)!"
        print('Assertion passed: self.__len__() == len(self.labelled_ids) + len(self.unlabelled_ids)')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_info = self.ids[index]
        video_id, start_frame, step_size = id_info
        videoname = self.video_list[video_id]
        images = []
        frame_num = start_frame
        # ego_labels = np.zeros(self.SEQ_LEN)-1
        all_boxes = []
        labels = []
        if self.train:
            # is_unlabelled = [True] * self.SEQ_LEN if videoname in self.labelled_videos else [False] * self.SEQ_LEN  #TODO: def self.labelled_videos
            assert index in self.ulb_indices or index in self.lbl_indices
            is_unlabelled = [False] * self.SEQ_LEN if videoname in self.labelled_videos else [True] * self.SEQ_LEN
        else:
            is_unlabelled = [False] * self.SEQ_LEN

        mask = np.zeros(self.SEQ_LEN, dtype=np.int)
        # indexs = []
        for i in range(self.SEQ_LEN):
            
            img_name = self._imgpath + '/{:s}/{:05d}.jpg'.format(videoname, frame_num+1)

            img = Image.open(img_name).convert('RGB')
            images.append(img)
            if not is_unlabelled[0]:
                mask[i] = 1
                all_boxes.append(self.frame_level_list[video_id][frame_num]['boxes'].copy())
                labels.append(self.frame_level_list[video_id][frame_num]['labels'].copy())
                # check if is_pseudo_labelled
                # if 'confidence' in self.frame_level_list[video_id][frame_num]:
                #     is_pseudo_labelled.append(True) #self.frame_level_list[video_id][frame_num]['is_pseudo_labelled'])
                # else:
                #     is_pseudo_labelled.append(False)
            else:
                all_boxes.append(np.asarray([]))
                labels.append(np.asarray([]))
                # ego_labels.append(-1)
            frame_num += step_size

        clip = self.transform(images)
        height, width = clip.shape[-2:]
        wh = [height, width]
        # print('image', wh)
        if self.ANCHOR_TYPE == 'RETINA':
            for bb, boxes in enumerate(all_boxes):
                if boxes.shape[0]>0:
                    if boxes[0,0]>1:
                        print(bb, videoname)
                        pdb.set_trace()
                    boxes[:, 0] *= width # width x1
                    boxes[:, 2] *= width # width x2
                    boxes[:, 1] *= height # height y1
                    boxes[:, 3] *= height # height y2

        # return clip, all_boxes, labels, ego_labels, index, wh, self.num_classes
        return clip, all_boxes, labels, index, wh, self.num_classes, videoname, start_frame, is_unlabelled


def custom_collate(batch):
    
    images = []
    boxes = []
    targets = []
    # ego_targets = []
    image_ids = []
    whs = []
    videonames = []
    start_frames = []
    is_pseudo_labelled = []

    for sample in batch:
        images.append(sample[0])
        boxes.append(sample[1])
        targets.append(sample[2])
        # ego_targets.append(torch.LongTensor(sample[3]))
        image_ids.append(sample[3])
        whs.append(torch.LongTensor(sample[4]))
        num_classes = sample[5]
        videonames.append(sample[6])
        start_frames.append(sample[7])
        is_pseudo_labelled.append(sample[8])

    counts = []
    max_len = -1
    seq_len = len(boxes[0])
    for bs_ in boxes:
        temp_counts = []
        for bs in bs_:
            max_len = max(max_len, bs.shape[0])
            temp_counts.append(bs.shape[0])
        assert seq_len == len(temp_counts)
        counts.append(temp_counts)
    counts = np.asarray(counts, dtype=np.int)
    new_boxes = torch.zeros(len(boxes), seq_len, max_len, 4)
    new_targets = torch.zeros([len(boxes), seq_len, max_len, num_classes])
    for c1, bs_ in enumerate(boxes):
        for c2, bs in enumerate(bs_):
            if counts[c1,c2]>0:
                assert bs.shape[0]>0, 'bs'+str(bs)
                new_boxes[c1, c2, :counts[c1,c2], :] = torch.from_numpy(bs)
                targets_temp = targets[c1][c2]
                assert targets_temp.shape[0] == bs.shape[0], 'num of labels and boxes should be same'
                new_targets[c1, c2, :counts[c1,c2], :] = torch.from_numpy(targets_temp)

    # images = torch.stack(images, 0)
    images = get_clip_list_resized(images)
    # print(images.shape)
    # return images, new_boxes, new_targets, torch.stack(ego_targets,0), torch.LongTensor(counts), image_ids, torch.stack(whs,0)
    return images, new_boxes, new_targets, torch.LongTensor(counts), image_ids, torch.stack(whs,0), \
           videonames, start_frames, is_pseudo_labelled
