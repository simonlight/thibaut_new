/**
 * 
 */
package linearNonChainGazeVocLocalizationClf;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.variable.BagImage;
import fr.lip6.jkernelmachines.type.TrainingSample;;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LSVM_console_ferrari {
	public static void main(String[] args) {
	
	String dataSource= "local";//local or other things
	String gazeType = "ferrari";
	String taskName = "test_chain_glsvm_motor_50_1e-4/";
	double[] lambdaCV = {1e-4};
    double[] epsilonCV = {0};
    ArrayList<Double> clfCCV=new ArrayList<Double>();
    for (Double cc=1.0;cc<100;){
    	clfCCV.add(cc);
    	cc+=1;
    }
    boolean classify=true;
//    String[] classes = {args[0]};
//	int[] scaleCV = {Integer.valueOf(args[1])};
//	String[] classes = {"aeroplane" ,"cow" ,"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
//	String[] classes = {"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
//	int[] scaleCV = {90,80,70,60,50,40,30};
	int[] scaleCV = {40};
	String[] classes = {"motorbike"};
//    double[] tradeoffCV = {0};
//    double[] tradeoffCV = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
	double[] tradeoffCV={0.2};
    String sourceDir = new String();
	String resDir = new String();
    double[] hardtradeoffCV = {0};
	if (dataSource=="local"){
		sourceDir = "/local/wangxin/Data/ferrari_gaze/";
		resDir = "/local/wangxin/results/ferrari_gaze/chain_glsvm_et/";
	}
	else if (dataSource=="big"){
		sourceDir = "/home/wangxin/Data/ferrari_gaze/";
		resDir = "/home/wangxin/results/ferrari_gaze/chain_glsvm_et/";
	}

	String initializedType = ".";//+0,+-,or other things
	boolean hnorm = false;
	
	String resultFolder = resDir+taskName;
	
	String resultFilePath = resultFolder + "ap_summary_ecarttype_seed1_detail.txt";
	String metricFolder = resultFolder + "metric/";
	String classifierFolder = resultFolder + "classifier/";
	String scoreFolder = resultFolder + "score/";
	String trainingDetailFolder = resultFolder + "trainingdetail/";
	
	int maxCCCPIter = 100;
	int minCCCPIter = 1;

	int maxSGDEpochs = 100;
	
	boolean stochastic = false;
    
	int optim = 2;
	int numWords = 2048;
	boolean saveClassifier = true;
    boolean loadClassifier = false;
    
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
			+ "\ntradeoff:\t"+Arrays.toString(tradeoffCV)
			+ "\nmaxCCCPIter:\t"+maxCCCPIter
			+ "\nminCCCPIter:\t"+minCCCPIter
			+ "\nmaxSGDEpochs:\t"+maxSGDEpochs
			+ "\nnumWords:\t"+numWords
			+ "\nsaveClassifier:\t"+Boolean.toString(saveClassifier)
		    + "\nloadClassifier:\t"+Boolean.toString(loadClassifier)
		    );
	
	int foldNum = 5;
	
	 for(String className: classes){
		    for(int scale : scaleCV) {
				String listTrainPath =  sourceDir+"example_files/"+scale+"/"+className+"_trainval_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
//				String listTrainPath =  sourceDir+"example_files/"+scale+"/"+className+"_train_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
//				String listValPath =  sourceDir+"example_files/"+scale+"/"+className+"_valtest_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";

		    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listTrain = BagReader.readBagImageLatent(listTrainPath, numWords, true, true, null, true, 0, dataSource);
//		    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listVal = BagReader.readBagImageLatent(listValPath, numWords, true, true, null, true, 0, dataSource);

		    	for(double epsilon : epsilonCV) {
			    	for(double lambda : lambdaCV) {
			    		for(double tradeoff : tradeoffCV) {
					    		
			    			int listsize = listTrain.size();

			    			List<Integer> apListIndex = new ArrayList<Integer>();
			    			for (int m=0;m<listTrain.size();m++){
			    				apListIndex.add(m);
			    			}
			    			Random seed = new Random(1);
							Collections.shuffle(apListIndex, seed);
			    			
							for (int i=0;i<1; i++){
		    					
//							for (int i=0;i<foldNum; i++){
	    						int fromIndex = listsize * i/foldNum;
	    						int toIndex = listsize * (i+1)/foldNum;
	    						List<Integer> trainList_1 = apListIndex.subList(0, fromIndex);
	    						List<Integer> trainList_2 = apListIndex.subList(toIndex, listsize);
	    						List<Integer> leftOutList = apListIndex.subList(fromIndex, toIndex);
	    						
	    						List<Integer> trainList = new ArrayList<Integer>();
	    						trainList.addAll(trainList_1);
	    						trainList.addAll(trainList_2);
			    			
			    			
							List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTrain = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
							for(int j:trainList) {
								exampleTrain.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listTrain.get(j).sample.x,0), listTrain.get(j).label));
							}
							
							List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleVal = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
							for(int j:leftOutList) {
								exampleVal.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listTrain.get(j).sample.x,0), listTrain.get(j).label));
							}
//							
//							List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleVal = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
//							for(int j=0;j<listVal.size();j++) {
//								exampleVal.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listVal.get(j).sample.x,0), listVal.get(j).label));
//							}

							LSVMGradientDescentBag classifier = new LSVMGradientDescentBag(); 
							File fileClassifier = new File(classifierFolder + "/" + className + "/"+ 
									className + "_" + scale + "_"+epsilon+"_"+lambda + 
									"_"+tradeoff+"_"+maxCCCPIter+"_"+minCCCPIter+"_"+maxSGDEpochs+
									"_"+optim+"_"+numWords+"_"+i+".lsvm");
							fileClassifier.getAbsoluteFile().getParentFile().mkdirs();
							
							if (loadClassifier && fileClassifier.exists()){
								ObjectInputStream ois;
								System.out.println("\nread classifier " + fileClassifier.getAbsolutePath());
								try {
									ois = new ObjectInputStream(new FileInputStream(fileClassifier.getAbsolutePath()));
									classifier = (LSVMGradientDescentBag) ois.readObject();
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
							}
							
							else{
								System.out.println("\ntraining classifier " + fileClassifier.getAbsolutePath());
								classifier.setOptim(optim);
								classifier.setMaxCCCPIter(maxCCCPIter);
								classifier.setMinCCCPIter(minCCCPIter);
								classifier.setEpsilon(epsilon);
								classifier.setLambda(lambda);
								classifier.setStochastic(stochastic);
								classifier.setVerbose(0);
								classifier.setTradeOff(tradeoff);
								classifier.setMaxEpochs(maxSGDEpochs);
								classifier.setGazeType(gazeType);								
								classifier.setLossDict(sourceDir+"ETLoss_dict/"+"ETLOSS+_"+scale+".loss");
								classifier.setHnorm(hnorm);
								classifier.setCurrentClass(className);
								//Initialize the region by fixations
//								for(STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer> ts : exampleTrain){
//									ts.input.h = lsvm.getGazeInitRegion(ts, scale, initializedType);
//								}

								File trainingDetailFile = new File(trainingDetailFolder + "/" + className + "/"+ 
										className + "_" + scale + "_"+epsilon+"_"+lambda + 
										"_"+tradeoff+"_"+maxCCCPIter+"_"+minCCCPIter+"_"+maxSGDEpochs+
										"_"+optim+"_"+numWords+"_"+i+".traindetail");
								trainingDetailFile.getAbsoluteFile().getParentFile().mkdirs();
								try {
									BufferedWriter trainingDetailFileOut = new BufferedWriter(new FileWriter(trainingDetailFile));
									classifier.train(exampleTrain, trainingDetailFileOut);
									trainingDetailFileOut.close();
								}	
							 catch (IOException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							 }
							
								}
							
		    				if (saveClassifier){
								
			    				ObjectOutputStream oos = null;
								try {
									oos = new ObjectOutputStream(new FileOutputStream(fileClassifier.getAbsolutePath()));
									oos.writeObject(classifier);
								} 
								catch (FileNotFoundException e) {
									e.printStackTrace();
								} 
								catch (IOException e) {
									e.printStackTrace();
								}
								finally {
									try {
										if(oos != null) {
											oos.flush();
											oos.close();
										}
									} catch (IOException e) {
										e.printStackTrace();
									}
								}
								System.out.println("wrote classifier successfully!");
							}
		    				
		    				if (classify){
		    					for(double clfC : clfCCV) {
		    						for(double tf: hardtradeoffCV){
		    							classifier.setClfC(clfC);
										System.out.println("clf C:"+clfC);
//					    				classifier.optimizeLatent(exampleTrain);
				
					    				classifier.init(exampleTrain);
					    				
					    				classifier.trainWC(exampleTrain);
				
										File trainMetricFile=new File(metricFolder+"/metric_train_"+scale+"_"+tradeoff+"_"+epsilon+"_"+lambda+"_"+className+"_"+i+".txt");
										trainMetricFile.getAbsoluteFile().getParentFile().mkdirs();
				
										double ap_train = classifier.testAPRegion(exampleTrain, trainMetricFile,tf);
										
										classifier.init(exampleVal);
//										classifier.optimizeLatent(exampleVal);

										File valMetricFile=new File(metricFolder+"/metric_val_"+scale+"_"+tradeoff+"_"+epsilon+"_"+lambda+"_"+className+"_"+i+".txt");
										valMetricFile.getAbsoluteFile().getParentFile().mkdirs();
				
										double ap_test = classifier.testAPRegion(exampleVal, valMetricFile,tf);
										
					    				try {
											BufferedWriter out = new BufferedWriter(new FileWriter(resultFilePath, true));
											out.write("category:"+className+" scale:"+scale+" hardtradeoff:"+tf+" clfC:"+clfC+" index:"+i+" ap_train:"+ap_train+" ap_test:"+ap_test+"\n");
											out.flush();
											out.close();
											
										} catch (IOException e) {
											e.printStackTrace();
										}
										}}
				//						classifier.optimizeLatent(exampleVal);
				//						double ap_val = classifier.testAP(exampleVal);
				//						System.err.println("train - ap= " + ap_val);
								
						//System.out.println(Arrays.toString())
				
				//		for(STrainingSample<LatentRepresentation<BagImage, Integer>,Integer> ex : exampleTrain) {
				//			Object[] res = classifier.predictionOutputLatent(ex.input.x);
				//			Integer h = (Integer)res[1];
				//			System.out.println(h);
				//		}
		    				}	}}}}}}}

	}
