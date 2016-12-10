cls_arr=("aeroplane" "cow" "dog" "cat" "motorbike" "boat"  "horse"  "sofa" "diningtable" "bicycle")
scale_arr=( "40" "30")


k='oarsub -p "host='"'"'big6'"'"'  or host='"'"'big7'"'"' or host='"'"'big8'"'"' or host='"'"'big13'"'"' " -l "nodes=1/core=3,walltime=1000:0:0" --notify "mail:biglip666@gmail.com" "/home/wangxin/lib/jdk1.8.0_25/bin/java -Xms16000m -Xmx16000m -XX:-UseGCOverheadLimit -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/thibaut_new/bin:. 
lsvmCCCPGazeVoc_PosNeg_topInstance/LSVM_console_ferrari'


end='"'
space=' '
for scale in ${scale_arr[@]}
do
	for cls in ${cls_arr[@]}
    do
        eval $k$space$cls$space$scale$end
    done
done
