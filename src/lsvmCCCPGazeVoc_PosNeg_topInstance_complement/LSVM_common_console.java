package lsvmCCCPGazeVoc_PosNeg_topInstance_complement;

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
import fr.durandt.jstruct.latent.LatentRepresentationTopK;
import fr.durandt.jstruct.variable.BagImage;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class LSVM_common_console {
	public static void console(String dataSource, String gazeType, String taskName,
						String sourceDir, String resDir, String gazeJmapFolder,
						int maxCCCPIter, int minCCCPIter, int maxSGDEpochs,
						int optim, int numWords, int foldNum, int randomSeed,
						boolean stochastic, boolean saveClassifier, boolean loadClassifier, boolean hnorm,
						double[] lambdaCV, double[] epsilonCV, double[] posTradeoffCV, double[] negTradeoffCV,
						String[] classes, int[] scaleCV, int maxK, int index
						) {
		String resultFolder = resDir+taskName;
		String resultFilePath = resultFolder + "ap_summary_ecarttype_seed1_detail.txt";
		String metricFolder = resultFolder + "metric/";
		String classifierFolder = resultFolder + "classifier/";
		String scoreFolder = resultFolder + "score/";
		String trainingDetailFolder = resultFolder + "trainingdetail/";

		System.out.println("experiment detail: "
				+ "\nmaxK:\t"+maxK
				+ "\noptim:\t"+optim
				+ "\nresDir:\t"+resDir
				+ "\nnumWords:\t"+numWords
				+ "\ntask name:\t"+taskName
				+ "\ngaze type:\t"+gazeType
				+ "\nsourceDir:\t "+sourceDir
				+ "\nmaxCCCPIter:\t"+maxCCCPIter
				+ "\nminCCCPIter:\t"+minCCCPIter
				+ "\nmaxSGDEpochs:\t"+maxSGDEpochs
				+ "\nhnorm:\t"+Boolean.toString(hnorm)
				+ "\nscale CV:\t"+Arrays.toString(scaleCV)
				+ "\nclasses CV:\t"+Arrays.toString(classes)
				+ "\nlambda CV:\t" + Arrays.toString(lambdaCV)
				+ "\nepsilon CV:\t" + Arrays.toString(epsilonCV)
				+ "\npostradeoff CV:\t"+Arrays.toString(posTradeoffCV)
				+ "\nnegtradeoff CV:\t"+Arrays.toString(negTradeoffCV)
				+ "\nsaveClassifier:\t"+Boolean.toString(saveClassifier)
				+ "\nloadClassifier:\t"+Boolean.toString(loadClassifier)
		    );
	for(String className: classes){
	    for(int scale : scaleCV) {
			//generate K, we can set the maximum number of k
	    	ArrayList<Integer> K = new ArrayList<Integer>();
			for (int KElement=maxK; KElement<=Math.min(maxK,convertScale(scale));KElement++){
				K.add(KElement);
				break;
			}
			String listTrainPath;

			if (gazeType.equals("ufood")){
				listTrainPath=  sourceDir+"example_files/"+scale+"/"+className+"_full_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
			}
			else{
				listTrainPath=  sourceDir+"example_files/"+scale+"/"+className+"_trainval_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";

			}
//			String listTrainPath =  sourceDir+"example_files/"+scale+"/"+className+"_train_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
//			String listValPath =  sourceDir+"example_files/"+scale+"/"+className+"_valtest_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";

	    	List<TrainingSample<LatentRepresentationTopK<BagImage,Integer>>> listTrain = BagReader.readBagImageLatentTopK(listTrainPath, numWords, true, true, null, true, 0, dataSource);
//	    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listVal = BagReader.readBagImageLatent(listValPath, numWords, true, true, null, true, 0, dataSource);

	    	for(double epsilon : epsilonCV) {
		    	for(double lambda : lambdaCV) {
		    		for(double postradeoff : posTradeoffCV) {
		    			for(double negtradeoff : negTradeoffCV) {
		    				for(int k : K){
				    			
			    			int listsize = listTrain.size();
	
			    			List<Integer> apListIndex = new ArrayList<Integer>();
			    			for (int m=0;m<listTrain.size();m++){
			    				apListIndex.add(m);
			    			}
			    			Random seed = new Random(randomSeed);
							Collections.shuffle(apListIndex, seed);
			    			for (int i=0;i<foldNum; i++){
			    				if (i!=index){
	    							continue;
	    						}
			    				int fromIndex = listsize * i/foldNum;
	    						int toIndex = listsize * (i+1)/foldNum;
	    						List<Integer> trainList_1 = apListIndex.subList(0, fromIndex);
	    						List<Integer> trainList_2 = apListIndex.subList(toIndex, listsize);
	    						List<Integer> leftOutList = apListIndex.subList(fromIndex, toIndex);
	    						
	    						List<Integer> trainList = new ArrayList<Integer>();
	    						trainList.addAll(trainList_1);
	    						trainList.addAll(trainList_2);
			    			
			    			
							List<TrainingSample<LatentRepresentationTopK<BagImage,Integer>>> exampleTrain = new ArrayList<TrainingSample<LatentRepresentationTopK<BagImage,Integer>>>();
							for(int j:trainList) {
								exampleTrain.add(new TrainingSample<LatentRepresentationTopK<BagImage, Integer>>(new LatentRepresentationTopK<BagImage, Integer>(listTrain.get(j).sample.x,new ArrayList<Integer>()), listTrain.get(j).label));
							}
							
							List<TrainingSample<LatentRepresentationTopK<BagImage,Integer>>> exampleVal = new ArrayList<TrainingSample<LatentRepresentationTopK<BagImage,Integer>>>();
							for(int j:leftOutList) {
								exampleVal.add(new TrainingSample<LatentRepresentationTopK<BagImage, Integer>>(new LatentRepresentationTopK<BagImage, Integer>(listTrain.get(j).sample.x,new ArrayList<Integer>()), listTrain.get(j).label));
							}
	
							LSVMGradientDescentBag classifier = new LSVMGradientDescentBag(); 
						
							File fileClassifier = new File(classifierFolder + "/" + className + "/"+ 
									className + "_" + scale + "_"+epsilon+"_"+lambda + 
									"_"+postradeoff+"_"+negtradeoff+"_"+maxCCCPIter+"_"+minCCCPIter+"_"+maxSGDEpochs+
									"_"+optim+"_"+numWords+"_"+k+"_"+i+".lsvm");
							fileClassifier.getAbsoluteFile().getParentFile().mkdirs();
							
							if (loadClassifier && fileClassifier.exists()){
								ObjectInputStream ois;
								System.out.println("\nread classifier " + fileClassifier.getAbsolutePath());
								try {
									ois = new ObjectInputStream(new FileInputStream(fileClassifier.getAbsolutePath()));
									classifier = (LSVMGradientDescentBag) ois.readObject();
								} catch (FileNotFoundException e) {
									e.printStackTrace();
								} catch (IOException e) {
									e.printStackTrace();
								} catch (ClassNotFoundException e) {
									e.printStackTrace();
								}
							}
							
							else {
								System.out.println("\ntraining classifier " + fileClassifier.getAbsolutePath());
								classifier.setK(k);
								classifier.setVerbose(0);
								classifier.setOptim(optim);
								classifier.setHnorm(hnorm);
								classifier.setScale(scale);
								classifier.setLambda(lambda);
								classifier.setEpsilon(epsilon);
								classifier.setGazeType(gazeType);								
								classifier.setStochastic(stochastic);
								classifier.setMaxEpochs(maxSGDEpochs);
								classifier.setCurrentClass(className);
								classifier.setMaxCCCPIter(maxCCCPIter);
								classifier.setMinCCCPIter(minCCCPIter);
								classifier.setPosTradeOff(postradeoff);
								classifier.setNegTradeOff(negtradeoff);
								classifier.setGazeRatioDict(sourceDir+gazeJmapFolder+"ETNB+_"+scale+".map");
								
								File trainingDetailFile = new File(trainingDetailFolder + "/" + className + "/"+ 
										className + "_" + scale + "_"+epsilon+"_"+lambda + 
										"_"+postradeoff+"_"+negtradeoff+"_"+maxCCCPIter+"_"+minCCCPIter+"_"+maxSGDEpochs+
										"_"+optim+"_"+numWords+"_"+k+"_"+i+".traindetail");
								trainingDetailFile.getAbsoluteFile().getParentFile().mkdirs();
								try {
									BufferedWriter trainingDetailFileOut = new BufferedWriter(new FileWriter(trainingDetailFile));
									classifier.train(exampleTrain, trainingDetailFileOut);
									trainingDetailFileOut.close();
								}	
								
							 catch (IOException e) {
								e.printStackTrace();
							}
							
							}
														
		    				if (saveClassifier){
			    				// save classifier
								
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
	
		    				classifier.optimizeLatent(exampleTrain);
							File trainMetricFile=new File(metricFolder+"/metric_train_"+scale+"_"+postradeoff+"_"+negtradeoff+"_"+epsilon+"_"+lambda+"_"+className+"_"+k+"_"+i+".txt");
							trainMetricFile.getAbsoluteFile().getParentFile().mkdirs();
	
							double ap_train = classifier.testAPRegion(exampleTrain, trainMetricFile);
		    				
							//without init there may be a bug, but i forgot why...
							classifier.init(exampleVal);
							
		    				classifier.optimizeLatent(exampleVal);
							File valMetricFile=new File(metricFolder+"/metric_val_"+scale+"_"+postradeoff+"_"+negtradeoff+"_"+epsilon+"_"+lambda+"_"+className+"_"+k+"_"+i+".txt");
							valMetricFile.getAbsoluteFile().getParentFile().mkdirs();
							 	             
							double ap_test = classifier.testAPRegion(exampleVal, valMetricFile);
		    				
		    				try {
								BufferedWriter out = new BufferedWriter(new FileWriter(resultFilePath, true));
								out.write("category:"+className+" lambda:"+lambda+" k:"+k+" scale:"+scale+" ptradeoff:"+postradeoff+" ntradeoff:"+negtradeoff+" index:"+i+" ap_test:"+ap_test+" ap_train:"+ap_train+"\n");
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
	}
	}

	public static int convertScale(int scale){
		return (int)(Math.pow((1+(100-scale)/10),2));
	}
}