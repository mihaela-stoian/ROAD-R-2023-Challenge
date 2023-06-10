"""

Authored by Mihaela C. Stoian

"""

from req_handler import *


def get_size_of_tensor(data):  # in gigabytes
    return data.element_size() * data.nelement() / 1e9


def get_sparse_representation(req_matrix):
    req_matrix = req_matrix.to_sparse()
    return req_matrix.indices(), req_matrix.values()


def godel_disjunctions_sparse(sH, Cplus, Cminus, weighted_literals=False):
    constr_values = torch.zeros(sH.shape[0], NUM_REQ).to(sH.device)

    indices_nnz_plus, values_nnz_plus = get_sparse_representation(Cplus)
    indices_nnz_minus, values_nnz_minus = get_sparse_representation(Cminus)

    # predictions_at_nonzero_values is a matrix [num bboxes, num_nonzero_vals_in_Cplus] which contains
    # the predicted value associated with each label (ordered as they appear in the columns of Cplus)
    predictions_at_nnz_values_plus = sH[:, indices_nnz_plus[1, :]]
    predictions_at_nnz_values_minus = (1. - sH[:, indices_nnz_minus[1, :]])
    if weighted_literals:
        predictions_at_nnz_values_plus *= values_nnz_plus
        predictions_at_nnz_values_minus *= values_nnz_minus

    # the line inside the loop below essentially means that:
    # the constraints containing label k are each multiplied by the value of the prediction for label k
    for k in range(NUM_LABELS):
        # ind[0, ind[1] == k] returns a list of indices of the requirements in which the kth label appears
        # positively in the conjunction
        # ind[1] == k creates a mask of dim [460] which is equal to 1 if the ith element in the matrix of the
        # indexes is equal to k
        constr_values[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] = torch.maximum(
            constr_values[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]],
            predictions_at_nnz_values_plus[:, indices_nnz_plus[1] == k])
        constr_values[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]] = torch.maximum(
            constr_values[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]],
            predictions_at_nnz_values_minus[:, indices_nnz_minus[1] == k])

    req_loss = torch.mean(constr_values)
    # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements,
    # and hence we want to minimize the 1-p
    return 1 - req_loss


def lukasiewicz_disjunctions_sparse(sH, Cplus, Cminus, weighted_literals=False):
    constr_values_unbounded = torch.zeros(sH.shape[0], NUM_REQ).to(sH.device)

    # pred = sH.clone()  # grads propagated through the cloned pred tensor are propagated through the
    # original sH tensor as well (so grads are updated through sH, which is what we want)
    indices_nnz_plus, values_nnz_plus = get_sparse_representation(Cplus)
    indices_nnz_minus, values_nnz_minus = get_sparse_representation(Cminus)

    # predictions_at_nonzero_values is a matrix [num bboxes, num_nonzero_vals_in_Cplus] which contains
    # the predicted value associated with each label (ordered as they appear in the columns of Cplus)
    predictions_at_nnz_values_plus = sH[:, indices_nnz_plus[1, :]]
    predictions_at_nnz_values_minus = (1. - sH[:, indices_nnz_minus[1, :]])
    if weighted_literals:
        predictions_at_nnz_values_plus *= values_nnz_plus
        predictions_at_nnz_values_minus *= values_nnz_minus

    # the line inside the loop below essentially means that:
    # the constraints containing label k are each multiplied by the value of the prediction for label k
    for k in range(NUM_LABELS):
        # ind[0, ind[1] == k] returns a list of indices of the requirements in which the kth label appears
        # positively in the conjunction
        # ind[1] == k creates a mask of dim [460] which is equal to 1 if the ith element in the matrix of the
        # indexes is equal to k
        constr_values_unbounded[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] += predictions_at_nnz_values_plus[:,
                                                                                     indices_nnz_plus[1] == k]
        constr_values_unbounded[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]] += predictions_at_nnz_values_minus[
                                                                                       :,
                                                                                       indices_nnz_minus[1] == k]

    constr_values = torch.min(torch.ones_like(constr_values_unbounded), constr_values_unbounded)
    req_loss = torch.mean(constr_values)

    # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements, and hence we want to minimize the 1-p
    return 1 - req_loss


def product_disjunctions_sparse(sH, Cplus, Cminus, weighted_literals=False):
    # The disjunction is more complex to implement thant the conjunction
    # e.g., A and B --> A*B while A or B --> A + B - A*B
    # Thus we see the disjunction as the negation of the conjunction of the negations of all its
    # literals (i.e., A or B = neg (neg A and neg B))

    constr_values = torch.ones(sH.shape[0], NUM_REQ).to(sH.device)

    # pred = sH.clone()  # grads propagated through the cloned pred tensor are propagated through the
    # original sH tensor as well (so grads are updated through sH, which is what we want)
    indices_nnz_plus, values_nnz_plus = get_sparse_representation(Cplus)
    indices_nnz_minus, values_nnz_minus = get_sparse_representation(Cminus)

    # predictions_at_nonzero_values is a matrix [num bboxes, num_nonzero_vals_in_Cplus] which contains
    # the predicted value associated with each label (ordered as they appear in the columns of Cplus)
    predictions_at_nnz_values_plus = sH[:, indices_nnz_plus[1, :]]
    predictions_at_nnz_values_minus = (1. - sH[:, indices_nnz_minus[1, :]])
    if weighted_literals:
        predictions_at_nnz_values_plus *= values_nnz_plus
        predictions_at_nnz_values_minus *= values_nnz_minus

    # the line inside the loop below essentially means that:
    # the constraints containing label k are each multiplied by the value of the prediction for label k
    for k in range(NUM_LABELS):
        # ind[0, ind[1] == k] returns a list of indices of the requirements in which the kth label appears
        # positively in the conjunction
        # ind[1] == k creates a mask of dim [460] which is equal to 1 if the ith element in the matrix of the
        # indexes is equal to k
        constr_values[:, indices_nnz_plus[0, indices_nnz_plus[1] == k]] *= predictions_at_nnz_values_plus[:,
                                                                           indices_nnz_plus[1] == k]
        constr_values[:, indices_nnz_minus[0, indices_nnz_minus[1] == k]] *= predictions_at_nnz_values_minus[:,
                                                                             indices_nnz_minus[1] == k]

    # Negate the value of the conjunction
    req_loss = torch.mean(1. - constr_values)

    # We need to do 1-req_loss because we want to maximise the probability p of satisfying our requirements,
    # and hence we want to minimize the 1-p
    return 1 - req_loss


def logical_requirements_loss(preds, logic, Cplus, Cminus):
    '''Args::
        preds: sigmoid activated predictions
        labels: one hot encoded labels
        num_pos: number of positve samples
        alpha: weighting factor to baclence +ve and -ve
        gamma: Exponent factor to baclence easy and hard examples
       Return::
        loss: computed loss and reduced by sum and normlised by num_pos
     '''

    # Discard all the labels we are not interested in
    H = preds[:, 1:NUM_LABELS + 1]  # discard agentness class

    if len(H) == 0:
        req_loss = torch.zeros(1).cuda().squeeze()
        return req_loss

    # Since we have replicated now we have that the matrices are 3-dims tensors where dimension 0 has len 1
    # --> we need to unsqueeze the tensors to get back the original matrices
    # Iplus, Iminus = Iplus.squeeze(), Iminus.squeeze()
    # Mplus, Mminus = Mplus.squeeze(), Mminus.squeeze()
    Cplus, Cminus = Cplus.squeeze(), Cminus.squeeze()

    req_loss = torch.zeros([1]).cuda()

    if logic == "Godel":
        req_loss = godel_disjunctions_sparse(H, Cplus, Cminus)
    elif logic == "Lukasiewicz":
        req_loss = lukasiewicz_disjunctions_sparse(H, Cplus, Cminus)
    elif logic == "Product":
        req_loss = product_disjunctions_sparse(H, Cplus, Cminus)
    else:
        raise Exception("Cannot be here, logic {:} not defined".format(logic))

    return req_loss

