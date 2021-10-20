import os

def file_generator(name):
    current_path = os.getcwd()

    experiment_folder = current_path + '\\' + name
    os.mkdir(experiment_folder)

    SR_outputPath = experiment_folder + '\\SR_output'
    os.mkdir(SR_outputPath)
    
    return SR_outputPath