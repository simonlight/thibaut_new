/**
 * 
 */
package svmVoc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.util.AveragePrecision;
import fr.durandt.jstruct.util.Pair;
import fr.durandt.jstruct.variable.BagImage;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LSVM_console_ferrari {
	public static void main(String[] args) {
	
	String dataSource= "local";//local or other things
	String gazeType = "ferrari";
	String taskName = "svm_rigid_split/";
	double[] lambdaCV = {1e-4};
    double[] epsilonCV = {0};
//    String[] classes = {args[0]};
//	int[] scaleCV = {Integer.valueOf(args[1])};
	String[] classes = {"aeroplane" ,"cow" ,"dog", "cat", "motorbike", "boat" , "horse" , "sofa" ,"diningtable", "bicycle"};
	int[] scaleCV = {100};
//	String[] classes = {"sofa"};
    
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
	File f = new File(resultFolder);
	f.mkdirs();
	String resultFilePath = resultFolder + "ap_summary_ecarttype_seed1_detail.txt";
	String metricFolder = resultFolder + "metric/";
	String classifierFolder = resultFolder + "classifier/";
	String scoreFolder = resultFolder + "score/";
	String trainingDetailFolder = resultFolder + "trainingdetail/";
	
		    	
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
	
	int foldNum = 5;
 	 for(String className: classes){
	    for(int scale : scaleCV) {
			String listTrainPath =  sourceDir+"example_files/"+scale+"/"+className+"_train_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
//			String listTrainPath =  sourceDir+"example_files/"+scale+"/"+className+"_train_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
//			String listValPath =  sourceDir+"example_files/"+scale+"/"+className+"_valtest_scale_"+scale+"_matconvnet_m_2048_layer_20.txt";
	    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listTrain = BagReader.readBagImageLatent(listTrainPath, numWords, true, true, null, true, 0, dataSource);
//	    	List<TrainingSample<LatentRepresentation<BagImage,Integer>>> listVal = BagReader.readBagImageLatent(listValPath, numWords, true, true, null, true, 0, dataSource);

	    	for(double epsilon : epsilonCV) {
		    	for(double lambda : lambdaCV) {
		    			int listsize = listTrain.size();

		    			List<Integer> apListIndex = new ArrayList<Integer>();
		    			for (int m=0;m<listTrain.size();m++){
		    				apListIndex.add(m);
		    			}
		    			Random seed = new Random(1);
						Collections.shuffle(apListIndex, seed);
						Double[] apList = new Double[foldNum];

						for (int i=0;i<foldNum; i++){
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
							
	  						List<TrainingSample<LatentRepresentation<BagImage,Integer>>> exampleTest = new ArrayList<TrainingSample<LatentRepresentation<BagImage,Integer>>>();
							
	  						for(int j:leftOutList) {
	  							exampleTest.add(new TrainingSample<LatentRepresentation<BagImage, Integer>>(new LatentRepresentation<BagImage, Integer>(listTrain.get(j).sample.x,0), listTrain.get(j).label));
							}
	  						
	  						Problem problem = new Problem();

	  						problem.l = exampleTrain.size();
	  						problem.y = new double[problem.l];
	  						problem.n = 2049;
	  						problem.x = new Feature[problem.l][problem.n];
	  						
	  						double C = 1e4;    // cost of constraints violation
	  						double eps = 0.001; 
	  						SolverType solver = SolverType.L2R_L2LOSS_SVC;
	  						Parameter parameter = new Parameter(solver, C, eps);

							for(int n=0; n<exampleTrain.size(); n++)
							{
						        double[] features = exampleTrain.get(n).sample.x.getInstance(0);
						        problem.y[n] = exampleTrain.get(n).label;
								
						        for (int j = 0; j < features.length; j++){
						        	problem.x[n][j]=new FeatureNode(j+1, features[j]);
						        }
							}
							
							
	  						
							Model model = Linear.train(problem, parameter);

							List<Pair<Integer,Double>> eval = new ArrayList<Pair<Integer,Double>>();
							for(int n=0; n<exampleTest.size(); n++) {
					        	// calcul score(x,y,h,w) = argmax_{y,h} <w, \psi(x,y,h)>
								double[] features = exampleTest.get(n).sample.x.getInstance(0);
								Feature[] instance = new Feature[problem.n];							
						        for (int j = 0; j < features.length; j++){
						        	instance[j]=new FeatureNode(j+1, features[j]);
						        }
								double score = Linear.predict(model, instance); // socre
//					        	System.out.println(l.get(i).label+" "+score);
					        	// label is changed to -1 1.
					        	eval.add(new Pair<Integer,Double>((exampleTest.get(n).label), score)); //
					        }
					        double ap = AveragePrecision.getAP(eval);
					        apList[i] = ap;
							
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

							out.write("category:"+className+" scale:"+scale+" lambda:"+lambda+" epsilon:"+epsilon+" ap_test:"+average+" std_variance:"+std_variance+"\n");
							out.flush();
							out.close();
							
						} catch (IOException e) {
							e.printStackTrace();
						}
					}
		    	}
	    }
	    
}
	}}
