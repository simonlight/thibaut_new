import xml.etree.cElementTree as ET

def ground_truth_bb_object(filename, category):
    xmltree = ET.ElementTree(file=filename)            
            
    #bb of objects of given class
    bbs=[]
    for elem in xmltree.iterfind('object'):
        for name in elem.iter('name'):
            
            if name.text == category:                            
                for coor in elem.iter('bndbox'):
                    xmax = [coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]
                    xmin = [coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]
                    ymax = [coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]
                    ymin = [coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]
                   
                    bbs.append([int(float(xmin)),int(float(ymin)),int(float(xmax)),int(float(ymax))])
    return bbs

def ground_truth_bb_all_action(filename):
    xmltree = ET.ElementTree(file=filename)            
            
    #bb of objects of given class
    bbs=[]
    for elem in xmltree.iterfind('object'):
        for name in elem.iter('name'):
            
            if name.text == "person":                            
                for coor in elem.iter('bndbox'):
                    xmax = [coorelem.text for coorelem in coor if coorelem.tag == 'xmax'][0]
                    xmin = [coorelem.text for coorelem in coor if coorelem.tag == 'xmin'][0]
                    ymax = [coorelem.text for coorelem in coor if coorelem.tag == 'ymax'][0]
                    ymin = [coorelem.text for coorelem in coor if coorelem.tag == 'ymin'][0]
                   
                    bbs.append([int(float(xmin)),int(float(ymin)),int(float(xmax)),int(float(ymax))])
    return bbs

def read_image_area(filename):
    xmltree = ET.ElementTree(file=filename)            
    for size in xmltree.iterfind('size'):
        for width in size.iter('width'):
            width = width.text
        for height in size.iter('width'):
            height = height.text
    return float(width)*float(height)
            
def pixel_bb_ratio_object(example_file_root, categories):
    annotation_root = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/Annotations"
    all_cnt=0
    all_pixel_bb_ratio = 0

    for category in categories:
        ef_name = '_'.join([category,"trainval_scale_100_matconvnet_m_2048_layer_20.txt"])
        ef = open(op.join(example_file_root, ef_name))
        ef.readline()
        pixel_bb_ratio = 0
        cnt=0
        for line in ef:
            filename = line.split(" ")[0].split('/')[0]
            xml_filename = op.join(annotation_root, filename+'.xml')
            bbs = ground_truth_bb_object(xml_filename, category)
            if bbs != []:
                cnt+=1
                all_cnt+=1
                bb_area_upper_bound = 0
                for bb in bbs:
                    xmin,ymin,xmax,ymax = bb
                    bb_area_upper_bound += (xmax-xmin)*(ymax-ymin)
                pixel_bb_ratio += bb_area_upper_bound / read_image_area(xml_filename)
                all_pixel_bb_ratio += bb_area_upper_bound / read_image_area(xml_filename)
        print "category:%s, pixel_bb_ratio=%f"%(category, pixel_bb_ratio/cnt)
    print "overall pixel bb ratio=%f"%(all_pixel_bb_ratio/all_cnt)
    

def pixel_bb_ratio_action(example_file_root, categories):
    annotation_root = "/local/wangxin/Data/VOCdevkit_trainset/VOC2012/Annotations"
    for category in categories:
        ef_name = '_'.join([category,"trainval_scale_100_matconvnet_m_2048_layer_20.txt"])
        ef = open(op.join(example_file_root, ef_name))
        ef.readline()
        pixel_bb_ratio = 0
        cnt=0
        for k,line in enumerate(ef):
            filename = line.split(" ")[0].split('/')[-1]
            xml_filename = op.join(annotation_root, filename+'.xml')
            bbs = ground_truth_bb_all_action(xml_filename)
            if bbs != []:
                cnt+=1
                bb_area_upper_bound = 0
                for bb in bbs:
                    xmin,ymin,xmax,ymax = bb
                    bb_area_upper_bound += (xmax-xmin)*(ymax-ymin)
                pixel_bb_ratio += bb_area_upper_bound / read_image_area(xml_filename)
                
        print "category:%s, pixel_bb_ratio=%f"%(category, pixel_bb_ratio/cnt)


if __name__ == "__main__":
    import os.path as op
    ferrari_example_file_root = "/local/wangxin/Data/ferrari_gaze/example_files/100"
    stefan_example_file_root = "/local/wangxin/Data/full_stefan_gaze/example_files/100"
    object_names = ["dog", "cat", "motorbike", "boat", "aeroplane", "horse" ,"cow", "sofa", "diningtable", "bicycle"]
    action_names = ["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]

#     pixel_bb_ratio_object(ferrari_example_file_root, object_names)
    pixel_bb_ratio_action(stefan_example_file_root, action_names)
    