import torch

from constants import CONSTRAINTS_PATH, NUM_LABELS, NUM_REQ
from data.datasets import split_train_dataset
from modules import utils
from req_handler import createIs, createMs

logger = utils.get_logger(__name__)


def save_ulb_indices(args, full_train_dataset):
    train_dataset, ulb_train_dataset = split_train_dataset(full_train_dataset, args.unlabelled_proportion)
    import pickle as pkl
    indices_dict = {}
    indices_dict['labelled'] = train_dataset.indices
    indices_dict['unlabelled'] = ulb_train_dataset.indices
    for key in ['ANCHOR_TYPE', 'DATASET', 'SUBSETS', 'SEQ_LEN', 'skip_step', 'num_steps',
                'input_type', 'root', '_imgpath', 'anno_file', 'label_types', 'video_list',
                'numf_list', 'frame_level_list', 'all_classes']:
        indices_dict[key] = train_dataset.dataset.__dict__[key]
    from pathlib import Path
    indices_path = Path('./ulb_split_indices/')
    indices_path.mkdir(parents=True, exist_ok=True)

    ulb_indices_path = indices_path / 'ulb_indices_ssl-unlbl-prop-{:}-{:%m-%d-%H-%M-%S}.pkl'.format(args.unlabelled_proportion, args.DATETIME_NOW)

    pkl.dump(indices_dict, open(ulb_indices_path, "wb"), protocol=-1)
    logger.info("Saved ulb indices at {:}".format(ulb_indices_path))
    return train_dataset, ulb_train_dataset




def compute_req_matrices(args):

    # Read constraints from file and create the Ms and Is matrices
    Iplus_np, Iminus_np = createIs(CONSTRAINTS_PATH, NUM_LABELS)
    Mplus_np, Mminus_np = createMs(CONSTRAINTS_PATH, NUM_LABELS)

    Iplus, Iminus = torch.from_numpy(Iplus_np).float(), torch.from_numpy(Iminus_np).float()
    Mplus, Mminus = torch.from_numpy(Mplus_np).float(), torch.from_numpy(Mminus_np).float()

    if args.LOGIC == "Product":
        # These are already the negated literals
        # matrix of negative appearances in the conjunction
        Cminus = Iminus + torch.transpose(Mplus, 0, 1)
        # matrix of positive appearances in the conjunction
        Cplus = Iplus + torch.transpose(Mminus, 0, 1)
    else:  # elif args.LOGIC == "Godel" or args.LOGIC == "Lukasiewicz":
        # These are the literals as they appear in the disjunction
        # Matrix of the positive appearances in the disjunction
        Cplus = Iminus + torch.transpose(Mplus, 0, 1)
        # matrix of negative appearances in the conjunction
        Cminus = Iplus + torch.transpose(Mminus, 0, 1)

    if args.MULTI_GPUS:
        # Since we are splitting the foarward call on multiple GPUs, whatever we pass to the forward call
        # gets splitted along the 0 dimension. In order to have a replication and not a splitting we replicate
        # the matrices along the newly generated dimension 0.
        # Iplus, Iminus = Iplus.unsqueeze(0), Iminus.unsqueeze(0)
        # Mplus, Mminus = Mplus.unsqueeze(0), Mminus.unsqueeze(0)
        Cplus, Cminus = Cplus.unsqueeze(0), Cminus.unsqueeze(0)

        # Iplus = Iplus.expand(torch.cuda.device_count(), NUM_REQ, NUM_LABELS)
        # Iminus = Iminus.expand(torch.cuda.device_count(), NUM_REQ, NUM_LABELS)
        # Mplus = Mplus.expand(torch.cuda.device_count(), NUM_LABELS, NUM_REQ)
        # Mminus = Mminus.expand(torch.cuda.device_count(), NUM_LABELS, NUM_REQ)
        Cplus = Cplus.expand(torch.cuda.device_count(), NUM_REQ, NUM_LABELS)
        Cminus = Cminus.expand(torch.cuda.device_count(), NUM_REQ, NUM_LABELS)

    return Cplus, Cminus


