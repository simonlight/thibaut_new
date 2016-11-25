cls_arr=("jumping" "phoning" "playinginstrument" "reading" "ridingbike" "ridinghorse" "running" "takingphoto" "usingcomputer" "walking")
#scale_arr=("90" "80" "70" "60" "40" "30")
#cls_arr=("jumping")
scale_arr=("50")

k='oarsub -p "host='"'"'big20'"'"'  or host='"'"'big30'"'"' or host='"'"'big31'"'"' or host='"'"'big19'"'"' or host='"'"'big16'"'"' or host='"'"'big17'"'"' or host='"'"'big18'"'"' " -l "nodes=1/core=8,walltime=1000:0:0" --notify "mail:biglip666@gmail.com" "/home/wangxin/lib/jdk1.8.0_25/bin/java -classpath /home/wangxin/mosek/7/tools/platform/linux64x86/bin/mosek.jar:/home/wangxin/lib/commons-cli-1.2.jar:/home/wangxin/lib/jkernelmachines.jar:/home/wangxin/code/thibaut_new/bin:. 
lsvmCCCPGazeVoc_PosNeg_topInstance/LSVM_console_stefan'

end='"'
space=' '
for scale in ${scale_arr[@]}
do
	for cls in ${cls_arr[@]}
    do
        eval $k$space$cls$space$scale$end
    done
done
