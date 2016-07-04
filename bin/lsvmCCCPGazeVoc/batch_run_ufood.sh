cls_arr=("apple-pie" "bread-pudding" "beef-carpaccio" "beet-salad" "chocolate-cake" "chocolate-mousse" "donuts" "beignets" "eggs-benedict" "croque-madame" "gnocchi" "shrimp-and-grits" "grilled-salmon" "pork-chop" "lasagna" "ravioli" "pancakes" "french-toast" "spaghetti-bolognese" "pad-thai")
scale_arr=("100" "90" "80" "70" "60" "50" "40" "30")
#scale_arr=("30")
k='oarsub -p "host='"'"'big31'"'"'  or host='"'"'big17'"'"' or host='"'"'big19'"'"' " -l "nodes=1/core=8,walltime=500:0:0" --notify "mail:biglip666@gmail.com" "/home/wangxin/lib/jdk1.8.0_25/bin/java -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/thibaut_new/bin:. 
lsvmCCCPGazeVoc/LSVM_console_ufood'
#EvaluationLSVMStefan5Fold
#EvaluationLSVMStefan
#LSVM_console_stefan
end='"'
space=' '
for scale in ${scale_arr[@]}
do
	for cls in ${cls_arr[@]}
    do
        eval $k$space$cls$space$scale$end
    done
done
