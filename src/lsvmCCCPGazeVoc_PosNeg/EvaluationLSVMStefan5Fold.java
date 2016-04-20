/**
 * 
 */
package lsvmCCCPGazeVoc_PosNeg;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.variable.BagImage;
import fr.lip6.jkernelmachines.type.TrainingSample;;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class EvaluationLSVMStefan5Fold {
	public static void main(String[] args) {
	
	String dataSource= "local";//local or other things
	String gazeType = "stefan";

	String sourceDir = new String();
	String resDir = new String();

	if (dataSource=="local"){
		sourceDir = "/local/wangxin/Data/full_stefan_gaze/";
		resDir = "/local/wangxin/results/full_stefan_gaze/lsvm_negpos/";
	}
	else if (dataSource=="big"){
		sourceDir = "/home/wangxin/Data/full_stefan_gaze/";
		resDir = "/home/wangxin/results/full_stefan_gaze/lsvm_negpos/";
	}

	String initializedType = ".";//+0,+-,or other things
	boolean hnorm = false;
	
	String taskName = "lsvm_cccpgaze_posneg_inverse_jumping/";
	
	String resultFolder = resDir+taskName;
	
	String resultFilePath = resultFolder + "ap_summary_ecarttype.txt";
	String metricFolder = resultFolder + "metric/";
	String classifierFolder = resultFolder + "classifier/";
	String scoreFolder = resultFolder + "score/";

//	String[] classes = {args[0]};
//	int[] scaleCV = {Integer.valueOf(args[1])};
	
//	String[] classes = {"jumping", "phoning" ,"playinginstrument" ,"reading" ,"ridingbike" ,"ridinghorse" ,"running" ,"takingphoto", "usingcomputer", "walking"};
	String[] classes = {"jumping"};

//	int[] scaleCV = {90, 80,70,60,50,40,30};
	int[] scaleCV = {50};
//	String[] classes = {"walking"};
    double[] lambdaCV = {1e-4};
    double[] epsilonCV = {0};

//    double[] tradeoffCV = {0.0, 0.0001,0.001,0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    double[] tradeoffCV = {0};
		    	
	int maxCCCPIter = 100;
	int minCCCPIter = 1;

	int maxSGDEpochs = 100;
	
	boolean stochastic = false;
    
	int optim = 2;
	int numWords = 2048;
    
	System.out.println("experiment detail: "
			+ "\nsourceDir:\t "+sourceDir
			+ "\nresDir:\t"+resDir
			+ "\ngaze type:\t"+gazeType
			+ "\ninitilaize type:\t"+initializedType
			+ "\nhnorm:\t"+Boolean.toString(hnorm)
			+ "\ntask name:\t"+taskName
			+ "\nclasses CV:\t"+Arrays.toString(classes)
			+ "\nscale CV:\t"+Arrays.toString(scaleCV)
			+ "\nlambda CV:\t" + Arrays.toString(lambdaCV)
			+ "\nepsilon CV:\t" + Arrays.toString(epsilonCV)
			+ "\noptim:\t"+optim
			+ "\nmaxCCCPIter:\t"+maxCCCPIter
			+ "\nminCCCPIter:\t"+minCCCPIter
			+ "\nmaxSGDEpochs:\t"+maxSGDEpochs
			+ "\nnumWords:\t"+numWords
		    );
	
	int foldNum = 5;
	
	 for(String className: classes){
	    for(int scale : scaleCV) {
	    	
	    	String listTestPath =  sourceDir+"example_files/"+scale+"/"+className+"_val_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
	    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listTest = BagReader.readBagImageLatent(listTestPath, numWords, true, true, null, true, 0, dataSource);
			
	    	for(double epsilon : epsilonCV) {
		    	for(double lambda : lambdaCV) {
		    		for(double tradeoff : tradeoffCV) {
		    			
		    			int listsize = listTest.size();

		    			List<Integer> apListIndex = new ArrayList<Integer>();
		    			for (int m=0;m<listTest.size();m++){
		    				apListIndex.add(m);
		    			}
		    			
						Collections.shuffle(apListIndex);
		    			
						Double[] apList = new Double[foldNum];
    					for (int i=0;i<foldNum; i++){
    						int fromIndex = listsize * i/foldNum;
    						int toIndex = listsize * (i+1)/foldNum;
    						List<Integer> testList_1 = apListIndex.subList(0, fromIndex);
    						List<Integer> testList_2 = apListIndex.subList(toIndex, listsize);
    						List<Integer> testList = new ArrayList<Integer>();
    						testList.addAll(testList_1);
    						testList.addAll(testList_2);
							
		    				List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTest = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
							for(int j:testList) {
								exampleTest.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listTest.get(j).sample.x,0), listTest.get(j).label));
							}


							LSVMGradientDescentBag classifier = new LSVMGradientDescentBag(); 
							////
							File fileClassifier = new File(classifierFolder + "/" + className + "/"+ 
									className + "_" + scale + "_"+epsilon+"_"+lambda + 
									"_"+tradeoff+"_"+maxCCCPIter+"_"+minCCCPIter+"_"+maxSGDEpochs+
									"_"+optim+"_"+numWords+".lsvm");
							ObjectInputStream ois;
							System.out.println("read classifier " + fileClassifier.getAbsolutePath());
							try {
								ois = new ObjectInputStream(new FileInputStream(fileClassifier.getAbsolutePath()));
								classifier = (LSVMGradientDescentBag) ois.readObject();
								classifier.showParameters();
							} catch (FileNotFoundException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							} catch (IOException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							} catch (ClassNotFoundException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
	
		    				
		    				//test metric file		    				
							classifier.optimizeLatent(exampleTest);
		    				double ap_test = classifier.testAP(exampleTest);
		    				apList[i] = ap_test;
    					}
    					double average = 0;
    					for (double ap: apList){
    						average+=ap;
    					}
    					average /= apList.length;
    					double variance = 0;
    					for (double ap: apList){
    						variance+=Math.pow(ap-average, 2);
    					}
    					variance /= apList.length;
    					double std_variance = Math.sqrt(variance);
						//write ap 
	    				try {
							BufferedWriter out = new BufferedWriter(new FileWriter(resultFilePath, true));
							out.write("category:"+className+" tradeoff:"+tradeoff+" scale:"+scale+" lambda:"+lambda+" epsilon:"+epsilon+" ap_test:"+average+" std_variance:"+std_variance+"\n");
							out.flush();
							out.close();
							
						} catch (IOException e) {
							e.printStackTrace();
						}
					
		    		
		    	}
		    }
	    }
	   }
	}
	}
}
