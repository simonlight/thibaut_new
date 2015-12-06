
def generateFeatureFile(musk_raw, musk_feature_dest):
    bag_label=col.defaultdict(lambda: None)
    with open(musk_raw) as musk_raw_file:
        musk_raw_file.readline()
        last_bag_id = -1
        bag_instance_num = -1
        for line in musk_raw_file:
            instance = line.strip().split()
            instance_id, bag_id, label = instance[0].split(':')
            bag_label[bag_id]=label
            if int(bag_id) != int(last_bag_id):
                last_bag_id = bag_id
                bag_instance_num=0
            elif int(bag_id) == int(last_bag_id):
                bag_instance_num+=1
            feature_path = op.join(musk_feature_dest, '_'.join([bag_id, str(bag_instance_num)])+'.txt')
            
            features = [i.split(':')[1] for i in instance[1:]]
            
            with open(feature_path, 'w') as feature_file:
                for feature in features:
                    feature_file.write(feature+'\n')
    return bag_label                

def generateExampleFile(feature_dest, musk_examplefile_dest, musk_bag_label):
    """bagname label number_of_instance feature_file0 feature_file1 feature_file2 ..."""
    
    bag_instances = os.listdir(feature_dest)
    bag_instance_dict = col.defaultdict(lambda:[])
    for i in bag_instances: 
        bag, instance = i.split('.')[0].split('_')
        bag_instance_dict[bag].append(int(instance))
    print bag_instance_dict
    with open(op.join(musk_examplefile_dest, 'example_file.txt'), 'w') as example_f:
        example_f.write(str(len(bag_instance_dict))+'\n')

        for bagname, instances in sorted(bag_instance_dict.items()):
            label = musk_bag_label[bagname]
            if label == '-1':
                label='0'
            number_of_instance = len(instances)
            feature_paths = ' '.join([op.join(feature_dest, bagname+'_'+str(i)+'.txt') for i in sorted(instances)])
            output = ' '.join([bagname, label, str(number_of_instance), feature_paths]) + '\n'
            example_f.write(output)
        
if __name__ == "__main__":
    import os.path as op
    import os
    import collections as col
    data_root = "/local/wangxin/Data/MilData"
    
    musk1_raw = op.join(data_root, "Musk", "musk1norm.svm")
    musk2_raw = op.join(data_root, "Musk", "musk2norm.svm")
    elephant_raw = op.join(data_root, "Elephant", "data_100x100.svm")
    fox_raw = op.join(data_root, "Fox", "data_100x100.svm")
    tiger_raw = op.join(data_root, "Tiger", "data_100x100.svm")
    
    musk1_feature_dest = op.join(data_root, "feature_files", "musk1")
    musk2_feature_dest = op.join(data_root, "feature_files", "musk2")
    elephant_feature_dest = op.join(data_root, "feature_files", "elephant")
    fox_feature_dest = op.join(data_root, "feature_files", "fox")
    tiger_feature_dest = op.join(data_root, "feature_files", "tiger")
    
    
    musk1_examplefile_dest = op.join(data_root, "example_files", "musk1")
    musk2_examplefile_dest = op.join(data_root, "example_files", "musk2")
    elephant_examplefile_dest = op.join(data_root, "example_files", "elephant")
    fox_examplefile_dest = op.join(data_root, "example_files", "fox")
    tiger_examplefile_dest = op.join(data_root, "example_files", "tiger")

    
    if not op.exists(musk1_feature_dest):
        os.makedirs(musk1_feature_dest)
        
    if not op.exists(musk2_feature_dest):
        os.makedirs(musk2_feature_dest)
        
    if not op.exists(elephant_feature_dest):
        os.makedirs(elephant_feature_dest)
        
    if not op.exists(fox_feature_dest):
        os.makedirs(fox_feature_dest)
        
    if not op.exists(tiger_feature_dest):
        os.makedirs(tiger_feature_dest)
    "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"   
    if not op.exists(musk1_examplefile_dest):
        os.makedirs(musk1_examplefile_dest)
    
    if not op.exists(musk2_examplefile_dest):
        os.makedirs(musk2_examplefile_dest)
    
    if not op.exists(elephant_examplefile_dest):
        os.makedirs(elephant_examplefile_dest)
    
    if not op.exists(fox_examplefile_dest):
        os.makedirs(fox_examplefile_dest)
    
    if not op.exists(tiger_examplefile_dest):
        os.makedirs(tiger_examplefile_dest)
    
    musk1_bag_label = generateFeatureFile(musk1_raw, musk1_feature_dest)
    musk2_bag_label = generateFeatureFile(musk2_raw, musk2_feature_dest)
    elephant_bag_label = generateFeatureFile(elephant_raw, elephant_feature_dest)
    fox_bag_label = generateFeatureFile(fox_raw, fox_feature_dest)
    tiger_bag_label = generateFeatureFile(tiger_raw, tiger_feature_dest)

    generateExampleFile(musk1_feature_dest, musk1_examplefile_dest, musk1_bag_label)
    generateExampleFile(musk2_feature_dest, musk2_examplefile_dest, musk2_bag_label)
    generateExampleFile(elephant_feature_dest, elephant_examplefile_dest, elephant_bag_label)
    generateExampleFile(fox_feature_dest, fox_examplefile_dest, fox_bag_label)
    generateExampleFile(tiger_feature_dest, tiger_examplefile_dest, tiger_bag_label)
    
    