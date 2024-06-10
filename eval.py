#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
import sys

import parameters
from time import gmtime, strftime
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates

def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    convert_to_polar = False

    dcase_output_test_folder = 1 if len(argv) < 3 else argv[-1]
    #dcase_output_test_folder = 'results_audio_depth/' + dcase_output_test_folder

    # Initialize evaluation metric class
    score_obj = ComputeSELDResults(params)

    if convert_to_polar:
        score_obj.write_polar_predictions_from_cartesian(dcase_output_test_folder)
        path = dcase_output_test_folder + "_polar"
    else:
        path = dcase_output_test_folder

    # Training setup
    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        if '2020' in params['dataset_dir']:
            test_splits = [1]
            val_splits = [2]
            train_splits = [[3, 4, 5, 6]]

        elif '2021' in params['dataset_dir']:
            test_splits = [6]
            val_splits = [5]
            train_splits = [[1, 2, 3, 4]]

        elif '2022' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]] 
        elif '2023' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]] 
        elif '2024' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]] # split 1 and 2 are simulated data, 3 and 4 are real recordings

        else:
            print('ERROR: Unknown dataset splits')
            exit()
    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')


        use_jackknife=False
        if params['evaluate_distance']:
            test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(path, is_jackknife=use_jackknife )
        else:
            test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(path, is_jackknife=use_jackknife )

        print('SELD score (early stopping metric): {:0.2f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
        #print('SED metrics: F-score: {:0.1f} {}'.format(100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
        print('SED metrics: Error rate: {:0.2f} {}, F-score: {:0.5f} {}'.format(test_ER[0]  if use_jackknife else test_ER, '[{:0.2f}, {:0.2f}]'.format(test_ER[1][0], test_ER[1][1]) if use_jackknife else '', 100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
        #print('DOA metrics: Angular error: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else ''))
        print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '', 100*test_LR[0]  if use_jackknife else 100*test_LR,'[{:0.2f}, {:0.2f}]'.format(100*test_LR[1][0], 100*test_LR[1][1]) if use_jackknife else ''))
        if params['evaluate_distance']:
            print('Distance metrics: {:0.2f} {}'.format(test_dist_err[0] if use_jackknife else test_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_dist_err[1][0], test_dist_err[1][1]) if use_jackknife else ''))
            print('Relative Distance metrics: {:0.2f} {}'.format(test_rel_dist_err[0] if use_jackknife else test_rel_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_rel_dist_err[1][0], test_rel_dist_err[1][1]) if use_jackknife else ''))

        if params['average']=='macro':
            if params['evaluate_distance']:
                print('Classwise results on unseen test data')
                print('Class\tF\tAE\tdist_err\treldist_err\tSELD_score')
                for cls_cnt in range(params['unique_classes']):
                    print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                        cls_cnt,

                        classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                                    classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                        classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                                    classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                        classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                                    classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                        classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                                    classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else '',

                        classwise_test_scr[0][6][cls_cnt] if use_jackknife else classwise_test_scr[6][cls_cnt],
                        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][6][cls_cnt][0],
                                                    classwise_test_scr[1][6][cls_cnt][1]) if use_jackknife else ''))
                    
            print('Classwise results on unseen test data')
            print('Class\tER\tF\tLE\tLR\tSELD_score')
            for cls_cnt in range(params['unique_classes']):
                print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                     cls_cnt,
                     classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                     classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))
                

                    
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

