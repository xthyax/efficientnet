import os
import glob
import json
import shutil

bmp_path = "4th_batch_bmp_labelled\\Unknow"
mvsd_path = "4th_batch_mvsd"
mvsd_output_path = "4th_batch_mvsd\\Unknow"

os.makedirs(mvsd_output_path, exist_ok=True)

bmp_list = []
for image in glob.glob(bmp_path + "\\*.bmp"):    
    image_name = image.split("\\")[-1].split(".")[0]
    bmp_list.append(image_name)

# print(bmp_list)
mvsd_list = []
for image in glob.glob(mvsd_path + "\\*.mvsd"):
    image_name = image.split("\\")[-1].split(".")[0]
    mvsd_list.append(image_name)

for image_name in bmp_list:
    if image_name in mvsd_list:
        with open( os.path.join(bmp_path,image_name) + ".bmp.json", "r") as bmp_file:
            obj_bmp = json.load(bmp_file)

        with open( os.path.join(mvsd_path,image_name) + ".mvsd.json", "r") as mvsd_file:
            obj_mvsd = json.load(mvsd_file)
        
        obj_mvsd['Status'] = obj_bmp['Status']
        # print(obj_mvsd['Status'])
        obj_mvsd['classId'] = obj_bmp['classId']

        print(os.path.join(mvsd_path,image_name) + ".mvsd.json")
        with open( os.path.join(mvsd_output_path,image_name) + ".mvsd.json", "w") as outfile:
            json.dump(obj_mvsd, outfile)
        shutil.move(os.path.join(mvsd_path, image_name + ".mvsd"), os.path.join(mvsd_output_path, image_name + ".mvsd"))
    else:
        pass