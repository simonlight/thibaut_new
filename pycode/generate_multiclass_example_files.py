root_folder = "/local/wangxin/Data/UPMC_Food_Gaze_20/example_files"
new_folder = "/local/wangxin/Data/UPMC_Food_Gaze_20/example_files_multi_class"
import os 
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

scale = ["100","90","80","70","60","50","40","30"]
key_word = ["test", "train", "trainval", "val"]
categories = [
        "apple-pie",
        "bread-pudding",
        "beef-carpaccio",
        "beet-salad",
        "chocolate-cake",
        "chocolate-mousse",
        "donuts",
        "beignets",
        "eggs-benedict",
        "croque-madame",
        "gnocchi",
        "shrimp-and-grits",
        "grilled-salmon",
        "pork-chop",
        "lasagna",
        "ravioli",
        "pancakes",
        "french-toast",
        "spaghetti-bolognese",
        "pad-thai"        
        ]
import collections
for s in scale:
    for kw in key_word:
        new_f = open(os.path.join(new_folder, '_'.join([kw, s, 'matconvnet_m_2048_layer_20.txt'])), 'w')
        name_label_dict = collections.defaultdict(lambda: None)
        f=open(os.path.join(root_folder, s, '_'.join(["apple-pie", kw, 'scale', s, 'matconvnet_m_2048_layer_20.txt'])))
        number= f.readline()
        new_f.write(number)
        for line in f:
            for cnt, c in enumerate(categories):
                id = line.split()[0]
                if c.replace('_','-') in line:
                    
                    label = str(cnt)
                    suffix = ' '.join(line.split()[2:])
            new_f.write(' '.join([id, label, suffix])+'\n')
