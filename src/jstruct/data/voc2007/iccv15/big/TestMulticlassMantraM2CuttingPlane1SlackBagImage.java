package jstruct.data.voc2007.iccv15.big;

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

import jstruct.data.voc2007.VOC2007;
import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.mantra.iccv15.multiclass.FastMulticlassMantraCuttingPlane1SlackBagImageRegion;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * Tests of LSSVM on UIUC Sports with deep features
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class TestMulticlassMantraM2CuttingPlane1SlackBagImage {

	public static String simDir = "/home/durandt/simulation/VOC2007/";

	public static void main(String[] args) {

		int numWords = 2048;

		double[] lambdaCV = {1e-3, 1e-4, 1e-5, 1e-6, 1e-7};
		double[] epsilonCV = {1e-2};
		Integer[] scaleCV = {100,90,80,70,60,50,40,30};
		//int[] splitCV = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
		int[] splitCV = {Integer.parseInt(args[0])};

		int cpmax = 500;
		int cpmin = 5;
		int optim = 2;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean recompute = false;

		for(int scale : scaleCV) {
			for(int split : splitCV) {
				String cls = VOC2007.getClasses()[split];
				
				String classifierDir = simDir + "/ICCV15/classifier/MANTRA/M2/ACC/CuttingPlane1Slack/Multiclass/Fast/matconvnet_m_" + numWords + "_layer_20/BagImageRegion/";
				String predictionDir = simDir + "/ICCV15/prediction/MANTRA/M2/ACC/CuttingPlane1Slack/Multiclass/Fast/matconvnet_m_" + numWords + "_layer_20/BagImageRegion/";
				String scoreDir = simDir + "/ICCV15/scores/MANTRA/M2/ACC/CuttingPlane1Slack/Multiclass/Fast/matconvnet_m_" + numWords + "_layer_20/BagImageRegion/";
				String inputDir = simDir + "/files_BagImageRegion/";

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);

				boolean compute = false;
				for(double epsilon : epsilonCV) {
					for(double lambda : lambdaCV) {

						FastMulticlassMantraCuttingPlane1SlackBagImageRegion classifier = new FastMulticlassMantraCuttingPlane1SlackBagImageRegion(); 
						classifier.setLambda(lambda);
						classifier.setEpsilon(epsilon);
						classifier.setCpmax(cpmax);
						classifier.setCpmin(cpmin);
						classifier.setVerbose(1);
						classifier.setOptim(optim);

						String suffix = "_" + classifier.toString();
						File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
						if(fileClassifier == null) {
							compute = true;
						}

						File filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + scale + suffix + "_train.txt");
						if(!filePrediction.exists()) {
							compute = true;
						}

						filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + scale + suffix + "_test.txt");
						if(!filePrediction.exists()) {
							compute = true;
						}	

						File fileScores = new File(scoreDir + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_train.txt");
						if(!fileScores.exists()) {
							compute = true;
						}	

						fileScores = new File(scoreDir + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_test.txt");
						if(!fileScores.exists()) {
							compute = true;
						}	
					}
				}

				if(compute || recompute) {
					List<STrainingSample<BagImageRegion, Integer>> listTrain = BagReader.readBagImageRegion(inputDir + "/" + cls + "_train_matconvnet_m_" + numWords + "_layer_20_scale_" + scale + ".txt", numWords, true, true, null, true, 0);
					List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>>();
					for(int i=0; i<listTrain.size(); i++) {
						exampleTrain.add(new STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>(new LatentRepresentation<BagImageRegion, Integer>(listTrain.get(i).input,0), listTrain.get(i).output));
					}

					List<STrainingSample<BagImageRegion, Integer>> listTest = BagReader.readBagImageRegion(inputDir + "/" + cls + "_test_matconvnet_m_" + numWords + "_layer_20_scale_" + scale + ".txt", numWords, true, true, null, true, 0);
					List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>>();
					for(int i=0; i<listTest.size(); i++) {
						exampleTest.add(new STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>(new LatentRepresentation<BagImageRegion, Integer>(listTest.get(i).input,0), listTest.get(i).output));
					}

					for(double epsilon : epsilonCV) {
						for(double lambda : lambdaCV) {

							FastMulticlassMantraCuttingPlane1SlackBagImageRegion classifier = new FastMulticlassMantraCuttingPlane1SlackBagImageRegion(); 
							classifier.setLambda(lambda);
							classifier.setEpsilon(epsilon);
							classifier.setCpmax(cpmax);
							classifier.setCpmin(cpmin);
							classifier.setVerbose(1);
							classifier.setnThreads(1);
							classifier.setOptim(optim);

							String suffix = "_" + classifier.toString();
							File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
							if(recompute || compute && fileClassifier == null) {
								classifier.train(exampleTrain);
								double ap = classifier.averagePrecision(exampleTrain);
								System.err.println("train - " + cls + "\tscale= " + scale + "\tacc= " + ap + "\tlambda= " + lambda );

								ap = classifier.averagePrecision(exampleTest);
								System.err.println("test - " + cls + "\tscale= " + scale + "\tacc= " + ap + "\tlambda= " + lambda );
								System.out.println("\n");

								fileClassifier = new File(classifierDir + "/" + cls + "/" + cls + "_" + scale + suffix + "_ap_" + ap + ".ser");
								fileClassifier.getAbsoluteFile().getParentFile().mkdirs();
								System.out.println("save classifier " + fileClassifier.getAbsolutePath());
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
							}
							else {
								// load classifier
								ObjectInputStream ois;
								System.out.println("read classifier " + fileClassifier.getAbsolutePath());
								try {
									ois = new ObjectInputStream(new FileInputStream(fileClassifier.getAbsolutePath()));
									classifier = (FastMulticlassMantraCuttingPlane1SlackBagImageRegion) ois.readObject();
								} 
								catch (FileNotFoundException e) {
									e.printStackTrace();
								} 
								catch (IOException e) {
									e.printStackTrace();
								} 
								catch (ClassNotFoundException e) {
									e.printStackTrace();
								}

								double acc = classifier.accuracy(exampleTest);
								System.err.println("test - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda );
								System.out.println("\n");
							}

							// prediction
							File filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + scale + suffix + "_train.txt");
							if(!filePrediction.exists()) {
								classifier.writePrediction(filePrediction, exampleTrain);
							}

							filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + scale + suffix + "_test.txt");
							if(!filePrediction.exists()) {
								classifier.writePrediction(filePrediction, exampleTest);
							}

							// scores
							File fileScores = new File(scoreDir + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_train.txt");
							if(!fileScores.exists()) {
								classifier.writeScores(fileScores, exampleTrain);
							}

							fileScores = new File(scoreDir + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_test.txt");
							if(!fileScores.exists()) {
								classifier.writeScores(fileScores, exampleTest);
							}
						}
					}
				}
			}
		}
	}

	public static File testPresenceFile(String dir, String test) {
		boolean testPresence = false;
		File classifierDir = new File(dir);
		File file = null;
		if(classifierDir.exists()) {
			String[] f = classifierDir.list();
			//System.out.println(Arrays.toString(f));

			for(String s : f) {
				if(s.contains(test)) {
					testPresence = true;
					file = new File(dir + "/" + s);
				}
			}
		}
		System.out.println("presence " + testPresence + "\t" + dir + "\t" + test + "\tfile " + (file == null ? null : file.getAbsolutePath()));
		return file;
	}
}
