/**
 * 
 */
package lsvmCCCPVoc;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
	String taskName = "lsvm_cccp_test/";
	double[] lambdaCV = {1e-4};
    double[] epsilonCV = {0};
//    double[] tradeoffCV = {0,0.1, 0.5, 1.0, 1.5, 2, 5, 10,100,1000};
//    String[] classes = {args[0]};
//    	int[] scaleCV = {Integer.valueOf(args[1])};
//	String[] classes = {"aeroplane" ,"cow" ,"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
//	String[] classes = {"aeroplane"};
	int[] scaleCV = {50};
	String[] classes = {"sofa"};
    
//    double[] tradeoffCV = {0.8,0.9};
	
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
	
	String resultFolder = resDir+taskName;
	
	String resultFilePath = resultFolder + "ap_summary.txt";
	String metricFolder = resultFolder + "metric/";
	String classifierFolder = resultFolder + "classifier/";
	String scoreFolder = resultFolder + "score/";

	
		    	
	int maxCCCPIter = 100;
	int minCCCPIter = 1;

	int maxSGDEpochs = 100;
	
	boolean semiConvexity = true;
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
			String listValPath =  sourceDir+"example_files/"+scale+"/"+className+"_valtest_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";

	    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listTrain = BagReader.readBagImageLatent(listTrainPath, numWords, true, true, null, true, 0, dataSource);
	    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listVal = BagReader.readBagImageLatent(listValPath, numWords, true, true, null, true, 0, dataSource);
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

						LSVMGradientDescentBag classifier = new LSVMGradientDescentBag(); 
					
						File fileClassifier = new File(classifierFolder + "/" + className + "/"+ 
								className + "_" + scale + "_"+epsilon+"_"+lambda + 
								"_"+maxCCCPIter+"_"+minCCCPIter+"_"+maxSGDEpochs+
								"_"+optim+"_"+numWords+".lsvm");
						fileClassifier.getAbsoluteFile().getParentFile().mkdirs();
						
						if (loadClassifier && fileClassifier.exists()){
							ObjectInputStream ois;
							System.out.println("\nread classifier " + fileClassifier.getAbsolutePath());
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
						}
						
						else {
							System.out.println("\ntraining classifier " + fileClassifier.getAbsolutePath());
							classifier.setOptim(optim);
							classifier.setMaxCCCPIter(maxCCCPIter);
							classifier.setMinCCCPIter(minCCCPIter);
							classifier.setSemiConvexity(semiConvexity);
							classifier.setEpsilon(epsilon);
							classifier.setLambda(lambda);
							classifier.setStochastic(stochastic);
							classifier.setVerbose(0);

							//Initialize the region by fixations
//							for(STrainingSample<LatentRepresentation<BagMIL, Integer>,Integer> ts : exampleTrain){
//								ts.input.h = lsvm.getGazeInitRegion(ts, scale, initializedType);
//							}
						
							classifier.train(exampleTrain);
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
						
	    				double ap_train = classifier.testAP(exampleTrain);
						System.err.println("train - ap= " + ap_train);
						double ap_val = classifier.testAP(exampleVal);
						System.err.println("train - ap= " + ap_val);
				
				
		//System.out.println(Arrays.toString())

//		for(STrainingSample<LatentRepresentation<BagImage, Integer>,Integer> ex : exampleTrain) {
//			Object[] res = classifier.predictionOutputLatent(ex.input.x);
//			Integer h = (Integer)res[1];
//			System.out.println(h);
//		}
	}}}}}

}
