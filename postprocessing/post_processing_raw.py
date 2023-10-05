############
# POST-PROCESS YOUR PKL FILE TO MAKE IT COMPLIANT TO THE REQUIREMENTS. THIS VERSION TAKES INTO ACCOUNT THE CONFIDENCE OF THE MODEL FOR EACH LABEL BEFORE POST-PROCESSING THE OUTPUT. 
# IMPORTANT NOTE: IT TAKES AS INPUT A FILE IN THE SAME FORMAT AS THE ONE REQUIRED FOR TASK 1. IT OUTPUTS A FILE COMPLIANT WITH TASK 2
# INPUTS NECESSARY FOR MAIN:
#  --file_path : path to your pkl file
#  --requirements_path : path to the requirements in WDIMACS format
#  --threshold: threshold at which a label is considered predicted
############
# Based on https://github.com/EGiunchiglia/ROAD-R/blob/main/post_processing.py

import glob
import zipfile
import pickle
import argparse
import time
import subprocess


NUM_LABELS = 41



###########
# Write temporary file with requirements (hard constraints) and output (soft constraints)
# Note: hard constraints have wegith 50.0, soft constraints have weight depending on the confidence of the model in the prediction
###########
def write_temp_file(pred, temp_file_path, wdimacs_path, th):

    with open(wdimacs_path, 'r') as f:
        with open(temp_file_path, 'w') as wf:
            # Write the hard constraints
            for i, line in enumerate(f.readlines()):
                wf.write(line)
            # Write the soft constraints (i.e., the predictions)
            for i, p in enumerate(pred): 
                if p > th:
                    wf.write(f"{p} {i+1} 0\n")
                else:
                    wf.write(f"{1-p} -{i+1} 0\n")


###########
# Call the solver MaxHs
# The function returns:
# 1. cost: integer representing how many predictions needed to be flipped 
# 2. assignement: list contaning the new assignment for each variable, for each element n in the list -n indicates negative literal and n positive.
#    Example: ['-1', '-2', '3', '-4', '-5', '-6', '-7', '-8', '-9', '-10', '-11', '-12', '-13', '14', '15', '-16', '-17', '-18', '-19', '-20', '-21', '22', '-23', '-24', '-25', '-26', '-27', '-28', '-29', '-30', '-31', '-32', '33', '-34', '-35', '-36', '-37', '38', '39', '-40', '-41']
# 3. time_elapsed: time taken by the solver to find solution (in seconds)
###########
def call_solver(temp_file_path):
    
    start = time.time()
    output = subprocess.run(["./MaxHS/build/release/bin/maxhs", "-printSoln", temp_file_path], capture_output=True)
    end = time.time()
    time_elapsed = end-start
    solver_output = output.stdout.decode("utf-8").splitlines()

    assigned = False
    cost=-1
    for line in solver_output:
        if line[0] == 'o':
            cost = line.split()[1]
        elif line[0] =='v':
            assigned = True
            assignment = line.split()[1:]
            assert len(assignment)==41#, "length assignment (%d)" % len(assignment)
    if not assigned: 
        print("Error in processing solver's output")
        print("Solver's output:")
        print(output.stdout.decode("utf-8"))
        exit(-1)

    return float(cost), assignment, time_elapsed    


def main():
    
    # Parse input arguments    
    parser = argparse.ArgumentParser(description='Post-processing output')
    parser.add_argument('--file_path', type=str, help='Path to the output file to post-process')
    parser.add_argument('--requirements_path', type=str, default='WDIMACS_requirements.txt', help='Path to the requirements file in WDIMACS format')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold at which a label is considered as predicted. If label > threshold then it is considered predicted.')
    args = parser.parse_args()
    
    # Set the file path for the temp req file so that we can write the temporary file in the same location as the requirements file
    split_req_path = args.requirements_path.split('.')
    requirements_temp_path = f"{split_req_path[0]}_temp.txt"
    
    # Load preds here
    file = glob.glob(args.file_path)[0]
    with open(file, 'rb') as fff:
        preds = pickle.load(fff)
        del fff
    
    for videoname in preds.keys(): 
        print(videoname)
        preds_framenames = preds[videoname].keys()
        for pred_framename in preds_framenames: 
            
            new_bboxes = []
            for i, bbox in enumerate(preds[videoname][pred_framename]):
                pred_labels = bbox['labels']
                
                # We cannot correct boxes for which no labels are predicted
                if all([p < args.threshold for p in pred_labels]):
                    continue
                # Write temporary file containing hard and soft constraints
                write_temp_file(pred_labels, requirements_temp_path, args.requirements_path, args.threshold)
                # call the solver on the temporary file
                cost, new_assgn, time_elaps = call_solver(requirements_temp_path)            
                # compute the new predictions                
                #compute the new labels to attach to the bbox
                final_pred_labels = [int(l)-1 for l in new_assgn if '-' not in l] 

                new_bboxes.append({'bbox': preds[videoname][pred_framename][i]['bbox'], 'labels':final_pred_labels})
            preds[videoname][pred_framename] = new_bboxes  

    with open(f"post_processed_{args.file_path}", 'wb') as handle:
        pickle.dump(preds, handle)


if __name__ == '__main__':
    main()