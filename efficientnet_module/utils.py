import fnmatch
import os
import glob
from sklearn.utils import class_weight
from prettytable import PrettyTable
import json
import cv2
import pandas as pd
from tqdm.autonotebook import tqdm
from skimage.util import img_as_ubyte

# from PIL import Image

import numpy as np
# from .config import INPUT_SIZE, ARCHITECTURE, BATCH, DATASET_DIR, Optimizer_
dash_path = "\\" if os.name =="nt" else "/"

def to_onehot(labels, num_of_classes):
    if type(labels) is list:
        labels = [int(label) for label in labels]
        arr = np.array(labels, dtype=np.int)
        onehot = np.zeros((arr.size, num_of_classes))
        onehot[np.arange(arr.size), arr] = 1

    else:
        onehot = np.zeros((num_of_classes,), dtype=np.int)
        onehot[int(labels)] = 1

    return smoothed_target


def multi_threshold(Y, thresholds):
    if Y.shape[-1] != len(thresholds):
        raise ValueError('Mismatching thresholds and output classes')

    thresholds = np.array(thresholds)
    thresholds = thresholds.reshape((1, thresholds.shape[0]))
    keep = Y > thresholds
    score = keep * Y
    class_id = np.argmax(score, axis=-1)
    class_score = np.max(score, axis=-1)
    if class_score == 0:
        return None
    return class_id, class_score


def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results

def recursive_folder(parent_folder):
    # TODO: make the search more deep
    # Sallow search
    sub_folder = next(os.walk(parent_folder))[1]
    list_subFolder = []
    if len(sub_folder) != 0 :
        for folder in sub_folder:
            sub_parentFolder = os.path.join(parent_folder, folder)
            # print(sub_parentFolder)
            list_subFolder.append(sub_parentFolder)
        return list_subFolder
    else:
        return parent_folder
    # print(parent_folder)
    # print(sub_folder)

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    return "".join([x if x.isalnum() else "_" for x in value])

def compute_class_weight(class_count):
    y = [[i] * v for i, v in enumerate(class_count)]
    y = flatten_list(y)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    return class_weights

def flatten_list(list_):
    return [item for sublist in list_ for item in sublist]

def metadata_count(dir, classes, gt_list, show_table=True):
    test_dir = dir if isinstance(dir,list) else [dir]
    class_list = list(dict.fromkeys(gt_list))

    Table = PrettyTable()
    Table.field_names = ['Defect', 'Number of images']
    count_class = [0] * len(class_list)
    for i in range(len(gt_list)):
        for j in range(len(classes)):
            if gt_list[i] == classes[j]:
                count_class[j] += 1

    metadata = {}
    # Empty folder check
    if len(gt_list) != 0: 
        for i in range(len(classes)):
            metadata.update({classes[i]: count_class[i]})
            Table.add_row([classes[i],count_class[i]])
    if show_table:
        print(f"[DEBUG] Path: {test_dir}")
        print(Table)
    if any('train' in sub_testdir.lower() for sub_testdir in test_dir):
        # print("[DEBUG] Had return metadata value")
        return metadata

def config_dump(save_path, config):
    data ={
        "LEARNING_RATE":f"{config.LEARNING_RATE}",
        "INPUT_SIZE": f"{config.INPUT_SIZE}",
        "BATCH": f"{config.BATCH_SIZE}",
        "EPOCH": f"{config.EPOCH}",
        "ARCHITECTURE": f"{config.ARCHITECTURE}",
        "OPTIMIZER": f"{config.OPTIMIZER}",
        "DATASET_PATH": f"{config.DATASET_PATH}"
    }
    with open(save_path + "/config.json", "w") as outfile:
        json.dump(data, outfile)

def load_and_crop(image_path, input_size=0):
    """ Load image and return image with specific crop size

    Input:
        image_path : Ex:Dataset/Train/img01.bmp
        input_size : any specific size
        
    Output:
        image after crop
    """
    image = cv2.imread(image_path)
    json_path = image_path + ".json"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size_image = image.shape

    try :
        with open(json_path, encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            box = json_data['box']
            center_x = box['centerX'][0]
            center_y = box['centerY'][0]
            widthBox = box['widthBox'][0]
            heightBox = box['heightBox'][0]
            class_gt = json_data['classId'][0]
    except:
        print(f"Can't find {json_path}")
        # Crop center image if no json found
        center_x = int(size_image[1] / 2)
        center_y = int(size_image[0] / 2)
        widthBox = 0
        heightBox = 0
        class_gt = "Empty"

    new_w = max(input_size, widthBox)
    new_h = max(input_size, heightBox)

    left, right = center_x - new_w / 2, center_x + new_w / 2
    top, bottom = center_y - new_h / 2, center_y + new_h / 2

    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(size_image[1] - 0, right)), round(min(size_image[0] - 0, bottom))

    return image[int(top):int(bottom), int(left):int(right)], class_gt

"""
    From here is handcraft features stuffs
"""

def process_img(img_path):
    cropped_image, _ = load_and_crop(img_path, input_size=128)
    image_color = img_as_ubyte(cropped_image)
    return image_color

def IncreaseDescreaseMask(Mask, Size):
    from skimage.morphology import erosion, dilation, opening, closing, white_tophat
    from skimage.morphology import disk
    selem = disk(abs(Size))
    if(Size > 0):
        result = dilation(Mask, selem)
    else:
        result = erosion(Mask, selem)  
    return result

def evaluate_mean_n_std(listClassesvalues):
    valueList = []
    [valueList.extend(data_per_class) for data_per_class in listClassesvalues]
    mean = np.mean(np.array(valueList))
    std = np.std(np.array(valueList))
    return mean, std
def get_standard_score(listClassesvalues, mean, std):
    dataReturn = []
    [dataReturn.append(((np.array(data_per_class) - mean) / std)) for data_per_class in listClassesvalues]
    return dataReturn

def intToBitArray(img) :
    row ,col = img.shape
    list = []
    for i in range(row):
        for j in range(col):
             list.append (np.binary_repr( img[i][j] ,width=8  ) )
    return list #the binary_repr() fucntion returns binary values but in 
                #string 
                #, not integer, which has it's own perk as you will notice 
def bitplane(bitImgVal , img1D ):
    bitList = [  int(   i[bitImgVal]  )    for i in img1D]
    return bitList
def GetBitImage(index, image2D):
    ImageIn1D = intToBitArray(image2D)
    Imagebit = np.array( bitplane(index, ImageIn1D ) )
    Imagebit = np.reshape(Imagebit , image2D.shape )
    return Imagebit
def GetAllBitImage(image2D):
    image2D_Bit = list()
    for i in range(8):
        image2D_Bit.append(GetBitImage(i, image2D))
    return image2D_Bit

def pick_Threshold(listClasses):
    from skimage.filters import threshold_multiotsu
    listImage = []
    multi_threshold = []
    [listImage.extend(classList) for classList in listClasses]

    for image_path in listImage:
        img = process_img(image_path)
        image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_dilation = IncreaseDescreaseMask(image_gray, 3)
        image_erosion = IncreaseDescreaseMask(image_gray, -3)
        image_re_dilation = IncreaseDescreaseMask(image_erosion, 6)
        image_compare = image_re_dilation - image_dilation
        multi_threshold.append(threshold_multiotsu(image_compare))
    return np.mean(np.array(multi_threshold), axis=0).tolist()
    
def GetFeature_Differential_GrayInfo_Mean(image_color, multi_thresholds):
    from skimage.filters import threshold_multiotsu
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

    image_dilation = IncreaseDescreaseMask(image_gray, 6)
    image_erosion = IncreaseDescreaseMask(image_gray, -3)
    image_re_dilation = IncreaseDescreaseMask(image_erosion, 9)
    
    # image_compare_1 = image_dilation - image_re_dilation
    image_compare = image_re_dilation - image_dilation
    threshold_compare = threshold_multiotsu(image_compare)
    image_differential = (image_compare > multi_thresholds[0]) * (image_compare < multi_thresholds[1])

    return [np.mean(image_differential), np.mean(threshold_compare)]

def GetFeature_Differential_BitInfo_Mean(image_color):
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

    image_dilation = IncreaseDescreaseMask(image_gray, 3)
    image_erosion = IncreaseDescreaseMask(image_gray, -3)
    image_re_dilation = IncreaseDescreaseMask(image_erosion, 6)
    
    image2D_Bit_Dilation = GetAllBitImage(image_dilation)
    image2D_Bit_reDilation = GetAllBitImage(image_re_dilation)


    image_differential00 = image2D_Bit_Dilation[0] - image2D_Bit_reDilation[0]
    image_differential01 = image2D_Bit_Dilation[1] - image2D_Bit_reDilation[1]
    image_differential02 = image2D_Bit_Dilation[2] - image2D_Bit_reDilation[2]
    # image_differential00 = image2D_Bit_Dilation[0] - image2D_Bit_reDilation[0]

    get_differential00 = image_differential00 > 0
    get_differential01 = image_differential01 > 0
    get_differential02 = image_differential02 > 0

    feature_list = [np.mean(get_differential00), np.mean(get_differential01), np.mean(get_differential02)]
    # feature = np.mean(get_differential00)
    return feature_list

def SegmentByOtsu(Image):
    from skimage.filters import threshold_multiotsu
    IM = Image.copy()
    
    if(len(IM.shape) == 3):
        IM = cv2.cvtColor(IM, cv2.COLOR_RGB2GRAY)
        
    thresh = threshold_multiotsu(IM)
    Mask0 = (IM > thresh[0])
    Mask1 = (IM > thresh[1])
    Mask0 = Mask0.astype(int)
    Mask1 = Mask1.astype(int)

    return Mask0, Mask1, thresh

def GetFeature_ContoursCount(Image):
    image_mask00, image_mask01, image_thresh = SegmentByOtsu(Image)

    result00 = cv2.Canny((image_mask00 * 255).astype(np.uint8), image_thresh[0], image_thresh[1])
    result01 = cv2.Canny((image_mask01 * 255).astype(np.uint8), image_thresh[0], image_thresh[1])
    _, hierachy00 = cv2.findContours(result00, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, hierachy01 = cv2.findContours(result01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valueReturn = [len(hierachy00[0]), len(hierachy01[0])]
    return valueReturn

def GetDefectFeature(defect_color, FeatureSelected = [1, 1, 1]):
    image_color = defect_color.copy()
    feature_gray_diff = GetFeature_Differential_GrayInfo_Mean(image_color, multi_threshold)
    feature_bit_diff = GetFeature_Differential_BitInfo_Mean(image_color)
    feature_contours = GetFeature_ContoursCount(image_color)

    feature_all = [feature_gray_diff, feature_bit_diff, feature_contours]
    feature_vector = []
    for idx, feature_selected in enumerate(FeatureSelected):
        if(feature_selected == 1):
            feature_vector.append(feature_all[idx])
    
    feature_vector = [item for sublist in feature_vector for item in sublist]
    return feature_vector

def ReadImageToDataFrame(Path, DictofClasses):
    ActionDF = pd.DataFrame(columns = ["Path", "Class", "Feature01", "Feature02", "Feature03"])
    listOfImage = []
    listOfClass = []
    [listOfImage.extend(DictofClasses[i]) for i in DictofClasses]
    [listOfClass.extend(len(DictofClasses[i])*[i]) for i in DictofClasses]

    progress_bar = tqdm(listOfImage)
    for iter, image_path in enumerate(progress_bar):
        # for class_name in DictofClasses:
        #     for impath in DictofClasses[class_name]:
        ActionDF.loc[iter, "Path"] = image_path
        ActionDF.loc[iter, "Class"] = listOfClass[iter]
        progress_bar.update()
    return ActionDF

def get_dataframe_one(img):
    one_dataframe = pd.DataFrame()
    image_color = img
    
    multi_threshold = pd.read_pickle("otsu_threshold.pkl")['multi_threshold'].tolist()
    # print(multi_threshold)
    feature_gray_diff = GetFeature_Differential_GrayInfo_Mean(image_color, multi_threshold)
    feature_bit_diff = GetFeature_Differential_BitInfo_Mean(image_color)
    feature_contours = GetFeature_ContoursCount(image_color)
    DF_model = pd.DataFrame(columns=["Feature01", "Feature02", "Feature03"])
    DF_model.loc[0, "Feature01"] = feature_gray_diff
    DF_model.loc[0, "Feature02"] = feature_bit_diff
    DF_model.loc[0, "Feature03"] = feature_contours
    
    idx = 0
    Feature_Vector = []
    Feature_Vector.append(DF_model.loc[idx, "Feature01"])
    Feature_Vector.append(DF_model.loc[idx, "Feature02"])
    Feature_Vector.append(DF_model.loc[idx, "Feature03"])
    Feature_Vector = [item for sublist in Feature_Vector for item in sublist]

    FeaturesColumn = ["Feature " + str(i + 1) for i in range(len(Feature_Vector))]
    DataFrame_Model = pd.DataFrame(columns = FeaturesColumn)

    Feature_Vector = []
    Feature_Vector.append(DF_model.loc[0, "Feature01"])
    Feature_Vector.append(DF_model.loc[0, "Feature02"])
    Feature_Vector.append(DF_model.loc[0, "Feature03"])
    # Feature_Vector.append(TestDefectDF_Model.loc[idx, "Feature04"])
    # Feature_Vector.append(TestDefectDF_Model1.loc[idx, "Feature05"])
    Feature_Vector = [item for sublist in Feature_Vector for item in sublist]

    DataFrame_Model.loc[0, :] = Feature_Vector

    return DataFrame_Model

def get_dataframe(list_Path, fail_list_class):
    
    list_Dataframe = []
    # random.seed(0)
    for Path in list_Path:
        Reject_ls = [image for image in glob.glob(os.path.join(Path,"*.bmp")) if load_and_crop(image)[1] in fail_list_class]
        Pass_ls = [image for image in glob.glob(os.path.join(Path,"*.bmp")) if load_and_crop(image)[1] == "Pass"]

        DictofClasses = {"Reject" : Reject_ls, "Pass": Pass_ls}
        if "train" in Path.lower():
            multi_threshold = pick_Threshold([Reject_ls])
            otsu_threshold = pd.DataFrame()
            otsu_threshold['multi_threshold'] = multi_threshold
            otsu_threshold.to_pickle("otsu_threshold.pkl")
        else:
            pass
        
        list_Dataframe.append(ReadImageToDataFrame(Path, DictofClasses))

    for DataFrame_ in list_Dataframe:
        DF_model = DataFrame_.copy()
        progress_bar = tqdm(DF_model.loc[:, "Path"])
        for iter, imgpath in enumerate(progress_bar):
            image_color = process_img(imgpath)
            
            feature_gray_diff = GetFeature_Differential_GrayInfo_Mean(image_color, multi_threshold)
            feature_bit_diff = GetFeature_Differential_BitInfo_Mean(image_color)
            feature_contours = GetFeature_ContoursCount(image_color)

            DF_model.loc[iter, "Feature01"] = feature_gray_diff
            DF_model.loc[iter, "Feature02"] = feature_bit_diff
            DF_model.loc[iter, "Feature03"] = feature_contours
            progress_bar.update()
        # print(DF_model)
        idx = 0
        Feature_Vector = []
        Feature_Vector.append([DF_model.loc[idx, "Path"]])
        Feature_Vector.append([DF_model.loc[idx, "Class"]])
        Feature_Vector.append(DF_model.loc[idx, "Feature01"])
        Feature_Vector.append(DF_model.loc[idx, "Feature02"])
        Feature_Vector.append(DF_model.loc[idx, "Feature03"])
        Feature_Vector = [item for sublist in Feature_Vector for item in sublist]

        ColumnNames = ["Path", "Class"]
        FeaturesColumn = ["Feature " + str(i + 1) for i in range(len(Feature_Vector) - len(ColumnNames))]
        ColumnNames = ColumnNames + FeaturesColumn
        DataFrame_Model = pd.DataFrame(columns = ColumnNames)

        for idx in DF_model.index:
            Feature_Vector = []
            Feature_Vector.append([DF_model.loc[idx, "Path"]])
            Feature_Vector.append([DF_model.loc[idx, "Class"]])
            Feature_Vector.append(DF_model.loc[idx, "Feature01"])
            Feature_Vector.append(DF_model.loc[idx, "Feature02"])
            Feature_Vector.append(DF_model.loc[idx, "Feature03"])
            # Feature_Vector.append(TestDefectDF_Model.loc[idx, "Feature04"])
            # Feature_Vector.append(TestDefectDF_Model1.loc[idx, "Feature05"])
            Feature_Vector = [item for sublist in Feature_Vector for item in sublist]

            DataFrame_Model.loc[idx, :] = Feature_Vector

        [DataFrame_Model.to_pickle('TrainDF_Model.pkl') if "train" in DataFrame_Model.loc[0, "Path"].lower() else DataFrame_Model.to_pickle('ValDF_Model.pkl')]

def get_z_score_info(DF_model):
    TempData = DF_model.copy()
    TempData = TempData.drop(["Path","Class"], axis=1)
    list_distribute = []
    [list_distribute.append(columnFeature)  for columnFeature in TempData.columns.tolist() if any(TempData[columnFeature] > 1)]
    Mean_std_DF = pd.DataFrame()
    for feature in list_distribute:
        mean, std = evaluate_mean_n_std([TempData[feature].tolist()])
        Mean_std_DF[feature] = [mean, std]

    Mean_std_DF.to_pickle("Mean_std_value.pkl")

def get_z_score(DF_z_score_info, list_DF):
    Feature_ls = DF_z_score_info.columns.tolist()
    for DF_ in list_DF:
        for feature in Feature_ls:
            mean, std = DF_z_score_info[feature].tolist()
            standardized_data = get_standard_score([DF_[feature].tolist()], mean, std)
            DF_[feature] = standardized_data[0]

def SplitDataFrameToTrainAndTest(DataFrame, TrainDataRate, TargetAtt):
    # gets a random TrainDataRate % of the entire set
    training = DataFrame.sample(frac=TrainDataRate, random_state=1)
    # gets the left out portion of the dataset
    testing = DataFrame.loc[~DataFrame.index.isin(training.index)]

    X_train = training.drop(TargetAtt, 1)
    y_train = training[[TargetAtt]]
    X_test = testing.drop(TargetAtt, 1)
    y_test = testing[[TargetAtt]]

    return X_train, y_train, X_test, y_test