import pandas as pd
import numpy as np
import os
import shutil
import sys


def get_most_common(data, column, number):
    """gets the number most common instances in a dataframe

    Args:
        data (dataframe): a dataframe of values
        column (str): the column to get the values from
        number (int): the number of most common values to get
    """
    glosses = (train['Gloss'].value_counts().sort_index().sort_values(ascending=False, kind='stable').head(100).index)
    return glosses


def remove_excess_folders(folderPath, glosses, newFolder, segmented = False):
    """puts all files with the given glosses into a new folder labeled newFolder

    Args:
        folderPath (str): filepath to the folder containing the videos
        glosses (list[str]): list of glosses for the videos
        newFolder (str): the name of the new folder with the videos
        segmented (bool, optional): represents if file has segmented- at the front. Defaults to False.

    Returns:
        str: path to newley created folder
    """
    folder_path = folderPath
    filenames = os.listdir(folder_path)
    stripped  = []
    #checks each file to see if it is one of the given glosses and adds the filename to stripped 
    for  f in filenames:
        name = os.path.splitext(f)[0]
        if segmented:
            name = name.split('-', 2)[2]
        elif '-' in name:
            name = name.split('-', 1)[1]
        name = name.replace(' ', '')
        if name in glosses:
            stripped.append(f)
    os.makedirs(newFolder, exist_ok=True)
    for filename in stripped:
        source = os.path.join(folder_path, filename)
        destination = os.path.join(newFolder, filename)
        shutil.copy2(source, destination)
    print(f"Copied {len(stripped)} files to {newFolder}")
    return os.path.abspath(newFolder)
    
    
def remove_excess_splits(trainFilepath, valFilepath, testFilepath, glosses):
    """removes excess splits from the csv files

    Args:
        trainFilepath (str): filePath to train data
        valFilepath (str): filePath to val data
        testFilepath (str): filepath to test data
        glosses (list[str]): list of strings to keep
    Returns:
        (dataframe, dataframe, dataframe): (train, val, test) post excess removal
    """
    train = pd.read_csv(trainFilepath)
    val = pd.read_csv(valFilepath)
    test = pd.read_csv(testFilepath)
    train = train[train['Gloss'].isin(glosses)]
    val = val[val['Gloss'].isin(glosses)]
    test = test[test['Gloss'].isin(glosses)] 
    return train, val, test
    
def remove_excess(videosFilepath, jointPath, trainFilepath, valFilepath, testFilepath, glosses, newFolder):
    """takes paths to several files removes all glosses not in glosses
        and then creates a new folder with the new data

    Args:
        videosFilepath (str): path to videos folder
        jointFilepath (str): path to joints data folder
        trainFilepath (str): path to training data
        valFilepath (str): path to validation data
        testFilepath (str): path to testing data
        glosses (str): list of glosses to keep
        newFolder (str): name of new folder
    """
    os.makedirs(newFolder, exist_ok=True)
    train, val, test = remove_excess_splits(trainFilepath, valFilepath, testFilepath, glosses)
    train.to_csv(os.path.join(newFolder, 'train.csv'), index=False)
    val.to_csv(os.path.join(newFolder, 'val.csv'), index=False)
    test.to_csv(os.path.join(newFolder, 'test.csv'), index=False)
    glossesEdited = []
    for g in glosses:
        g = g.replace('/', '-')
        if g[-1].isdigit():
            if g[-1] == "1":
                glossesEdited.append(g[:-1])
            else:
                glossesEdited.append(g[:-1] + " " + g[-1])
        else:
            glossesEdited.append(g)
        
    newVideosPath = remove_excess_folders(videosFilepath, glossesEdited, os.path.join(newFolder, "segmented-videos"), segmented =True)
    newJoinPath = remove_excess_folders(jointPath, glossesEdited, os.path.join(newFolder, "joint_data"))
    print(f"All filtered data saved to: {os.path.abspath(newFolder)}")
    return os.path.abspath(newFolder)

if __name__ == '__main__':
    if len(sys.argv) < 6:
            print(f"Usage: python {sys.argv[0]} <videosFilepath> <jointPath> <trainPath> <valPath> <testPath> [newName] [number]")
            sys.exit(1)
    videosFilepath =sys.argv[1]
    jointPath = sys.argv[2]
    trainPath = sys.argv[3]
    valPath = sys.argv[4]
    testPath = sys.argv[5]
    newName = "new_folder" # default value change if you want 
    if len(sys.argv) > 6:
        newName = sys.argv[6]
    num = 100   #default value change if you want
    if len(sys.argv) > 7 and sys.argv[7].isdigit():
        num = int(sys.argv[7]) 
    train = pd.read_csv(trainPath)
    glosses = get_most_common(train, 'Gloss',num)
    remove_excess(videosFilepath, jointPath, trainPath, valPath, testPath, glosses, newName)