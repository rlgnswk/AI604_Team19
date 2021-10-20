import os

def file_generator(name):

    '''
    Make folder and files for write the logs and outputs.

    make folder corresponded experiment name given as input

    1. make experiment_folder by input name
    2. make output image folder, tensorboard log folder, log.txt

    output: SR_output_Path, tensorBoard_logPath, log_path
    
    '''

    current_path = os.getcwd()

    experiment_folder = current_path + '\\' + name
    os.mkdir(experiment_folder)

    SR_outputPath = experiment_folder + '\\SR_output'
    os.mkdir(SR_outputPath)
    
    return SR_outputPath