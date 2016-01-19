package lsvmStandardVoc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.variable.BagImage;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class EvaluationLSVMFerrari {
	public static void main(String[] args) {
	
	String dataSource= "big";//local or other things
	String gazeType = "ferrari";

	String sourceDir = new String();
	String resDir = new String();

	if (dataSource=="local"){
		sourceDir = "/local/wangxin/Data/ferrari_gaze/";
		resDir = "/local/wangxin/results/ferrari_gaze/std_et/";
	}
	else if (dataSource=="big"){
		sourceDir = "/home/wangxin/Data/ferrari_gaze/";
		resDir = "/home/wangxin/results/ferrari_gaze/std_et/";
	}

	String initializedType = ".";//+0,+-,or other things
	boolean hnorm = false;
	
	String taskName = "lsvm_standard/";
	
	String resultFolder = resDir+taskName;
	
	String resultFilePath = resultFolder + "ap_summary.txt";
	String metricFolder = resultFolder + "metric_final/";
	String classifierFolder = resultFolder + "classifier/";
	String scoreFolder = resultFolder + "score/";

	String[] classes = {args[0]};
	int[] scaleCV = {Integer.valueOf(args[1])};
//	String[] classes = {"aeroplane" ,"cow" ,"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
//	int[] scaleCV = {90,80,70,60,50,40,30};
//	String[] classes = {"sofa"};
    double[] lambdaCV = {1e-4};
    double[] epsilonCV = {0};

		    	
	int maxCCCPIter = 100;
	int minCCCPIter = 2;

	int maxSGDEpochs = 100;
	
	int optim = 2;
	int numWords = 2048;
	boolean saveClassifier = false;
    boolean loadClassifier = true;
    
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
			+ "\nsaveClassifier:\t"+Boolean.toString(saveClassifier)
		    + "\nloadClassifier:\t"+Boolean.toString(loadClassifier)
		    );
	
	 for(String className: classes){
	    for(int scale : scaleCV) {
	    	String listTrainPath =  sourceDir+"example_files/"+scale+"/"+className+"_train_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
	    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listTrain = BagReader.readBagImageLatent(listTrainPath, numWords, true, true, null, true, 0, dataSource);

	    	String listValPath =  sourceDir+"example_files/"+scale+"/"+className+"_valval_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
	    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listVal = BagReader.readBagImageLatent(listValPath, numWords, true, true, null, true, 0, dataSource);
	    	
	    	String listTestPath =  sourceDir+"example_files/"+scale+"/"+className+"_valtest_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
	    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listTest = BagReader.readBagImageLatent(listTestPath, numWords, true, true, null, true, 0, dataSource);
			
	    	for(double epsilon : epsilonCV) {
		    	for(double lambda : lambdaCV) {
		
		    			List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTrain = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
						for(int i=0; i<listTrain.size(); i++) {
							exampleTrain.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listTrain.get(i).sample.x,0), listTrain.get(i).label));
						}
						List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleVal = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
						for(int i=0; i<listVal.size(); i++) {
							exampleVal.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listVal.get(i).sample.x,0), listVal.get(i).label));
						}
						List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTest = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
						for(int i=0; i<listTest.size(); i++) {
							exampleTest.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listTest.get(i).sample.x,0), listTest.get(i).label));
						}


						LSVMGradientDescentBag classifier = new LSVMGradientDescentBag(); 
						File fileClassifier = new File(classifierFolder + "/" + className + "/"+ 
								className + "_" + scale + "_"+epsilon+"_"+lambda + 
								"_"+maxCCCPIter+"_"+minCCCPIter+"_"+maxSGDEpochs+
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
						//train metric file
						File trainMetricFile=new File(metricFolder+scale+"/metric_train_"+scale+"_"+epsilon+"_"+lambda+"_"+className+".txt");
						trainMetricFile.getAbsoluteFile().getParentFile().mkdirs();
						classifier.optimizeLatent(exampleTrain);
						double ap_train = classifier.testAPRegion(exampleTrain,trainMetricFile);
	    				System.out.println("ap train:"+ap_train);

	    				//val metric file
						File valMetricFile=new File(metricFolder+scale+"/metric_valval_"+scale+"_"+epsilon+"_"+lambda+"_"+className+".txt");
						valMetricFile.getAbsoluteFile().getParentFile().mkdirs();
						classifier.optimizeLatent(exampleVal);
						double ap_val = classifier.testAPRegion(exampleVal,valMetricFile);
	    				System.out.println("ap val:"+ap_val);
	    				
	    				//test metric file		    				
	    				File testMetricFile=new File(metricFolder+scale+"/metric_valtest_"+scale+"_"+epsilon+"_"+lambda+"_"+className+".txt");
	    				testMetricFile.getAbsoluteFile().getParentFile().mkdirs();
						classifier.optimizeLatent(exampleTest);
	    				double ap_test = classifier.testAPRegion(exampleTest, testMetricFile);
	    				System.out.println("ap test:"+ap_test);
	    				
	    				//write ap 
	    				try {
							BufferedWriter out = new BufferedWriter(new FileWriter(resultFilePath, true));
							out.write("category:"+className+" scale:"+scale+" lambda:"+lambda+" epsilon:"+epsilon+" ap_train:"+ap_train+" ap_val:"+ap_val+" ap_test:"+ap_test+"\n");
							out.flush();
							out.close();
							
						} catch (IOException e) {
							e.printStackTrace();
						}
	    				System.err.format("train:%s category:%s scale:%s lambda:%s epsilon:%s %n ", ap_train, className, scale, lambda, epsilon); 
	    				System.err.format("val:%s category:%s scale:%s lambda:%s epsilon:%s %n ", ap_val, className, scale, lambda, epsilon); 
	    				System.err.format("test:%s category:%s scale:%s lambda:%s epsilon:%s %n ", ap_test, className, scale, lambda, epsilon); 
				
		    		}
		    	
		    }
	    }
	   }
	}

}
