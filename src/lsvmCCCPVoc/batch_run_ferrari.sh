cls_arr=("aeroplane" "cow" "dog" "cat" "motorbike" "boat"  "horse"  "sofa" "diningtable" "bicycle")
#cls_arr=("dog" "cat" "motorbike" "boat" "aeroplane" "horse" "cow" "sofa" "diningtable" "bicycle")
#cls_arr=("dog" "cat" "motorbike" "boat"  "horse"  "sofa" "diningtable" "bicycle")
scale_arr=("90" "80" "70" "60" "50" "40" "30")
#cls_arr=("sofa")

#scale_arr=("90")

k='oarsub -p "host='"'"'big20'"'"' " -l "nodes=1/core=4,walltime=500:0:0" --notify "mail:biglip666@gmail.com" "/home/wangxin/lib/jdk1.8.0_25/bin/java -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/thibaut_new/bin:. 
lsvmCCCPVoc/LSVM_console_ferrari'


end='"'
space=' '
for scale in ${scale_arr[@]}
do
	for cls in ${cls_arr[@]}
    do
        eval $k$space$cls$space$scale$end
    done
done
