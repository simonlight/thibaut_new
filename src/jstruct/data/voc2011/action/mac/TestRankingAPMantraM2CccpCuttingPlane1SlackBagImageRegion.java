/**
 * 
 */
package jstruct.data.voc2011.action.mac;

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

import jstruct.data.voc2011.VOC2011;
import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInputBagImageRegion;
import fr.durandt.jstruct.latent.mantra.iccv15.ranking.LatentCoupleMinMax;
import fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingAPMantraM2CccpCuttingPlane1SlackBagImageRegion;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class TestRankingAPMantraM2CccpCuttingPlane1SlackBagImageRegion {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr_2013_tutoriel/";

	private static int numWords = 2405;

	public static void main(String[] args) {

		//double[] lambdaCV = {1e-4};
		double[] lambdaCV = {1e4, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
		//double[] epsilonCV = {1e-2, 1e-3, 1e-4, 1e-6};
		//double[] lambdaCV = {1e-3, 1e-4, 1e-5};
		double[] epsilonCV = {1e-3};
		double[] gammaCV = {1e-2, 1e-1, 1, 1e1};
		//double[] gammaCV = {0.1};
		int[] splitCV = {1,2,3,4,5};
		//int[] splitCV = {1};
		int targetLabel = 1;

		int cpmax = 500;
		int cpmin = 5;
		int maxCCCPIter = 20;
		int minCCCPIter = 2;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean recompute = false;

		for(int iCls=0; iCls<VOC2011.getActionClasses().length; iCls++) {
		//for(int iCls=VOC2011.getActionClasses().length-1; iCls >= 0; iCls--) {
			String cls = VOC2011.getActionClasses()[iCls];
			for(int split : splitCV) {			

				String classifierDir = simDir + "/ICCV15_2/classifier/Mantra/M2/AP/CCCP/";
				String predictionDir = simDir + "/ICCV15_2/prediction/MANTRA/M2/AP/CCCP/";
				String inputDir = simDir + "/files/";

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);

				boolean compute = false;
				for(double epsilon : epsilonCV) {
					for(double gamma : gammaCV) {
						for(double lambda : lambdaCV) {

							RankingAPMantraM2CccpCuttingPlane1SlackBagImageRegion classifier = new RankingAPMantraM2CccpCuttingPlane1SlackBagImageRegion();
							classifier.setLambda(lambda);
							classifier.setEpsilon(epsilon);
							classifier.setCpmax(cpmax);
							classifier.setCpmin(cpmin);
							classifier.setVerbose(1);
							classifier.setMaxCCCPIter(maxCCCPIter);
							classifier.setMinCCCPIter(minCCCPIter);
							classifier.setGamma(gamma);

							String suffix = "_" + classifier.toString();
							File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + split + suffix);
							if(fileClassifier == null) {
								compute = true;
							}

							/*File filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + split + suffix + "_train.txt");
						if(!filePrediction.exists()) {
							compute = true;
						}

						filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + split + suffix + "_test.txt");
						if(!filePrediction.exists()) {
							compute = true;
						}*/
						}
					}
				}


				if(compute || recompute) {

					List<STrainingSample<BagImageRegion, Integer>> listTrain = BagReader.readBagImageRegion(inputDir + "/" + cls + "_" + split + "_train.txt", numWords, true, true, null, true, 0);

					LatentRankingInputBagImageRegion rankTrain = new LatentRankingInputBagImageRegion(listTrain);
					RankingOutput rankTrainY = new RankingOutput();
					List<Integer> labels = new ArrayList<Integer>();
					List<LatentCoupleMinMax<Integer>> latent = new ArrayList<LatentCoupleMinMax<Integer>>();
					for(int i=0; i<listTrain.size(); i++) {
						labels.add(listTrain.get(i).output == targetLabel ? 1 : 0);
						latent.add(i, new LatentCoupleMinMax<Integer>(0,0));
					}
					rankTrainY.initialize(labels);

					List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>> train = new ArrayList<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>>();
					train.add(new STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>(new LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>,List<LatentCoupleMinMax<Integer>>>(rankTrain, latent), rankTrainY));

					List<STrainingSample<BagImageRegion, Integer>> listTest = BagReader.readBagImageRegion(inputDir + "/" + cls + "_" + split + "_test.txt", numWords, true, true, null, true, 0);

					LatentRankingInputBagImageRegion rankTest = new LatentRankingInputBagImageRegion(listTest);
					RankingOutput rankTestY = new RankingOutput();
					labels = new ArrayList<Integer>();
					latent = new ArrayList<LatentCoupleMinMax<Integer>>();
					for(int i=0; i<listTest.size(); i++) {
						labels.add(listTest.get(i).output == targetLabel ? 1 : 0);
						latent.add(i, new LatentCoupleMinMax<Integer>(0,0));
					}
					rankTestY.initialize(labels);

					List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>> test = new ArrayList<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>>();
					test.add(new STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>(new LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>,List<LatentCoupleMinMax<Integer>>>(rankTest, latent), rankTestY));


					for(double epsilon : epsilonCV) {
						for(double gamma : gammaCV) {
							for(double lambda : lambdaCV) {

								RankingAPMantraM2CccpCuttingPlane1SlackBagImageRegion classifier = new RankingAPMantraM2CccpCuttingPlane1SlackBagImageRegion();
								classifier.setLambda(lambda);
								classifier.setEpsilon(epsilon);
								classifier.setCpmax(cpmax);
								classifier.setCpmin(cpmin);
								classifier.setVerbose(1);
								classifier.setMaxCCCPIter(maxCCCPIter);
								classifier.setMinCCCPIter(minCCCPIter);
								classifier.setGamma(gamma);

								String suffix = "_" + classifier.toString();
								File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + split + suffix);
								if(recompute || compute && fileClassifier == null) {
									classifier.train(train);
									double ap = classifier.averagePrecision(train);
									System.err.println("train - " + cls + "\tsplit= " + split + "\tap= " + ap + "\tlambda= " + lambda );

									ap = classifier.averagePrecision(test);
									System.err.println("test - " + cls + "\tsplit= " + split + "\tap= " + ap + "\tlambda= " + lambda );
									System.out.println("\n");

									fileClassifier = new File(classifierDir + "/" + cls + "/" + cls + "_" + split + suffix + "_ap_" + ap + ".ser");
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
										classifier = (RankingAPMantraM2CccpCuttingPlane1SlackBagImageRegion) ois.readObject();
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

									double ap = classifier.averagePrecision(train);
									System.err.println("train - " + cls + "\tsplit= " + split + "\tap= " + ap + "\tlambda= " + lambda );
									System.out.println("\n");
								}

								// prediction
								/*File filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + split + suffix + "_train.txt");
							if(!filePrediction.exists()) {
								classifier.writePrediction(filePrediction, train);
							}

							filePrediction = new File(predictionDir + "/" + cls + "/predict_" + cls + "_" + split + suffix + "_test.txt");
							if(!filePrediction.exists()) {
								classifier.writePrediction(filePrediction, test);
							}*/
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
