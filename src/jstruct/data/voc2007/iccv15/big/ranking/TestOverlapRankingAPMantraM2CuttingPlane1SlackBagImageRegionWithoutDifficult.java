/**
 * 
 */
package jstruct.data.voc2007.iccv15.big.ranking;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import jstruct.data.voc2007.VOC2007;
import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInputBagImageRegion;
import fr.durandt.jstruct.latent.mantra.iccv15.ranking.LatentCoupleMinMax;
import fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingAPMantraM2CuttingPlane1SlackBagImageRegion;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class TestOverlapRankingAPMantraM2CuttingPlane1SlackBagImageRegionWithoutDifficult {

	public static String simDir = "/home/durandt/simulation/VOC2007/";

	private static int numWords = 2048;

	public static void main(String[] args) {

		double[] lambdaCV = {1e-4};
		double[] epsilonCV = {1e-3};
		//int[] splitCV = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
		int[] splitCV = {Integer.parseInt(args[0])};
		Integer[] scaleCV = {100,90,80,70,60,50,40,30};
		int targetLabel = 1;

		int cpmax = 500;
		int cpmin = 5;
		int optim = 2;

		double[] overlapCV = {0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1};
		//double[] overlapCV = {Double.parseDouble(args[1])};
		int[] nb = {1,1,1,1,2,3,4,5};

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean recompute = false;

		for(double overlap : overlapCV) {
			for(int s=0; s<scaleCV.length; s++) {
				int scale = scaleCV[s];
				for(int split : splitCV) {
					String cls = VOC2007.getClasses()[split];

					String classifierDir = simDir + "/ICCV15/overlap/classifier/MantraCVPR/M2/AP/overlap_" + overlap + "/CuttingPlane1Slack/BagImageRegion/without_difficults/";
					String predictionDir = simDir + "/ICCV15/overlap/prediction/MANTRA/M2/AP/overlap_" + overlap + "/CuttingPlane1Slack/BagImageRegion/without_difficults/";
					String scoreDir = simDir + "/ICCV15/overlap/scores/MANTRA/M2/AP/overlap_" + overlap + "/CuttingPlane1Slack/BagImageRegion/without_difficults/";
					String scoreMaxMinDir = simDir + "/ICCV15/overlap/scoresMaxMin/MANTRA/M2/AP/overlap_" + overlap + "/CuttingPlane1Slack/BagImageRegion/without_difficults/";
					String inputDir = simDir + "/files_BagImageRegion/without_difficults/";

					System.out.println("classifierDir: " + classifierDir + "\n");
					System.err.println("split " + split + "\t cls " + cls);

					boolean compute = false;
					for(double epsilon : epsilonCV) {
						for(double lambda : lambdaCV) {

							RankingAPMantraM2CuttingPlane1SlackBagImageRegion classifier = new RankingAPMantraM2CuttingPlane1SlackBagImageRegion();
							classifier.setLambda(lambda);
							classifier.setEpsilon(epsilon);
							classifier.setCpmax(cpmax);
							classifier.setCpmin(cpmin);
							classifier.setVerbose(1);
							classifier.setOptim(optim);

							String suffix = "_" + classifier.toString();
							for(int t=0; t<nb[s]; t++) {

								/*File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/" + t + "/", cls + "_" + scale + suffix);
							if(fileClassifier == null) {
								//compute = true;
							}*/

								File fileScores = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_train.txt");
								File fileScores2 = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_train_null.txt");
								if(!fileScores.exists() && !fileScores2.exists()) {
									compute = true;
								}	
								System.out.println(fileScores.getAbsolutePath() + "\t" + fileScores.exists());
								System.out.println(fileScores2.getAbsolutePath() + "\t" + fileScores2.exists());

								fileScores = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_test.txt");
								fileScores2 = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_test_null.txt");
								if(!fileScores.exists() && !fileScores2.exists()) {
									compute = true;
								}	
								System.out.println(fileScores.getAbsolutePath() + "\t" + fileScores.exists());
								System.out.println(fileScores2.getAbsolutePath() + "\t" + fileScores2.exists());

								File filePrediction = new File(predictionDir + "/" + cls + "/" + t + "/predict_" + cls + "_" + scale + suffix + "_test.txt");
								File filePrediction2 = new File(predictionDir + "/" + cls + "/" + t + "/predict_" + cls + "_" + scale + suffix + "_test_null.txt");
								if(!filePrediction.exists() && !filePrediction2.exists()) {
									compute = true;
								}
								System.out.println(filePrediction.getAbsolutePath() + "\t" + filePrediction.exists());
								System.out.println(filePrediction2.getAbsolutePath() + "\t" + filePrediction2.exists());

								filePrediction = new File(predictionDir + "/" + cls + "/" + t + "/predict_" + cls + "_" + scale + suffix + "_train.txt");
								filePrediction2 = new File(predictionDir + "/" + cls + "/" + t + "/predict_" + cls + "_" + scale + suffix + "_train_null.txt");
								if(!filePrediction.exists() && !filePrediction2.exists()) {
									compute = true;
								}	
								System.out.println(filePrediction.getAbsolutePath() + "\t" + filePrediction.exists());
								System.out.println(filePrediction2.getAbsolutePath() + "\t" + filePrediction2.exists());

								fileScores = new File(scoreMaxMinDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_train.txt");
								fileScores2 = new File(scoreMaxMinDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_train_null.txt");
								if(!fileScores.exists() && !fileScores2.exists()) {
									compute = true;
								}	
								System.out.println(fileScores.getAbsolutePath() + "\t" + fileScores.exists());
								System.out.println(fileScores2.getAbsolutePath() + "\t" + fileScores2.exists());

								fileScores = new File(scoreMaxMinDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_test.txt");
								fileScores2 = new File(scoreMaxMinDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_test_null.txt");
								if(!fileScores.exists() && !fileScores2.exists()) {
									compute = true;
								}	
								System.out.println(fileScores.getAbsolutePath() + "\t" + fileScores.exists());
								System.out.println(fileScores2.getAbsolutePath() + "\t" + fileScores2.exists());


							}
						}
					}


					if(compute || recompute) {

						List<STrainingSample<BagImageRegion, Integer>> listTrain = BagReader.readBagImageRegion(inputDir + "/" + cls + "_train_matconvnet_m_" + numWords + "_layer_20_scale_" + scale + ".txt", numWords, true, true, null, true, 0);
						List<STrainingSample<BagImageRegion, Integer>> listTest = BagReader.readBagImageRegion(inputDir + "/" + cls + "_test_matconvnet_m_" + numWords + "_layer_20_scale_" + scale + ".txt", numWords, true, true, null, true, 0);


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
							for(double lambda : lambdaCV) {

								RankingAPMantraM2CuttingPlane1SlackBagImageRegion classifier = new RankingAPMantraM2CuttingPlane1SlackBagImageRegion();
								classifier.setLambda(lambda);
								classifier.setEpsilon(epsilon);
								classifier.setCpmax(cpmax);
								classifier.setCpmin(cpmin);
								classifier.setVerbose(1);
								classifier.setOptim(optim);

								String suffix = "_" + classifier.toString();

								boolean stop = false;

								for(int t=0; t<nb[s]; t++) {
									if(!stop) {

										File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/" + t + "/", cls + "_" + scale + suffix);
										if(recompute || compute && fileClassifier == null) {
											classifier.train(train);
											double ap = classifier.averagePrecision(train);
											System.err.println("train - " + cls + "\tscale= " + scale + "\tap= " + ap + "\tlambda= " + lambda );

											ap = classifier.averagePrecision(test);
											System.err.println("test - " + cls + "\tscale= " + scale + "\tap= " + ap + "\tlambda= " + lambda );
											System.out.println("\n");

											fileClassifier = new File(classifierDir + "/" + cls + "/" + t + "/" + cls + "_" + scale + suffix + "_ap_" + ap + ".ser");
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
												classifier = (RankingAPMantraM2CuttingPlane1SlackBagImageRegion) ois.readObject();
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
											System.err.println("test - " + cls + "\tscale= " + scale + "\tap= " + ap + "\tlambda= " + lambda );
											System.out.println("\n");
										}

										// scores
										File fileScores = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_train.txt");
										if(!fileScores.exists()) {
											classifier.writeScores(fileScores, train);
										}

										fileScores = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_test.txt");
										if(!fileScores.exists()) {
											classifier.writeScores(fileScores, test);
										}
										
										fileScores = new File(scoreMaxMinDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_train.txt");
										if(!fileScores.exists()) {
											classifier.writeScoresMaxMin(fileScores, train);
										}

										fileScores = new File(scoreMaxMinDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_test.txt");
										if(!fileScores.exists()) {
											classifier.writeScoresMaxMin(fileScores, test);
										}

										// prediction
										File filePrediction = new File(predictionDir + "/" + cls + "/" + t + "/predict_" + cls + "_" + scale + suffix + "_test.txt");
										if(!filePrediction.exists()) {
											classifier.writePrediction(filePrediction, test);
										}

										filePrediction = new File(predictionDir + "/" + cls + "/" + t + "/predict_" + cls + "_" + scale + suffix + "_train.txt");
										if(!filePrediction.exists()) {
											classifier.writePrediction(filePrediction, train);
										}

										List<BagImageRegion> listPrediction = classifier.readPrediction(filePrediction);
										stop = removeOverlap(listPrediction, train.get(0).input.x, overlap);
										if(stop) {
											System.out.println("Not enought instances");
										}
									}
									else {
										File fileScores = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_train_null.txt");
										System.out.println("write " + fileScores.getAbsolutePath());
										try {
											fileScores.getParentFile().mkdirs();
											OutputStream ops = new FileOutputStream(fileScores); 
											OutputStreamWriter opsr = new OutputStreamWriter(ops);
											BufferedWriter bw = new BufferedWriter(opsr);
											bw.write(" ");
											bw.close();
										}
										catch (IOException e) {
											System.out.println("Error parsing file "+ fileScores);
											return;
										}

										fileScores = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_test_null.txt");
										System.out.println("write " + fileScores.getAbsolutePath());
										try {
											fileScores.getParentFile().mkdirs();
											OutputStream ops = new FileOutputStream(fileScores); 
											OutputStreamWriter opsr = new OutputStreamWriter(ops);
											BufferedWriter bw = new BufferedWriter(opsr);
											bw.write(" ");
											bw.close();
										}
										catch (IOException e) {
											System.out.println("Error parsing file "+ fileScores);
											return;
										}
										
										fileScores = new File(scoreMaxMinDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_train_null.txt");
										System.out.println("write " + fileScores.getAbsolutePath());
										try {
											fileScores.getParentFile().mkdirs();
											OutputStream ops = new FileOutputStream(fileScores); 
											OutputStreamWriter opsr = new OutputStreamWriter(ops);
											BufferedWriter bw = new BufferedWriter(opsr);
											bw.write(" ");
											bw.close();
										}
										catch (IOException e) {
											System.out.println("Error parsing file "+ fileScores);
											return;
										}

										fileScores = new File(scoreMaxMinDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_test_null.txt");
										System.out.println("write " + fileScores.getAbsolutePath());
										try {
											fileScores.getParentFile().mkdirs();
											OutputStream ops = new FileOutputStream(fileScores); 
											OutputStreamWriter opsr = new OutputStreamWriter(ops);
											BufferedWriter bw = new BufferedWriter(opsr);
											bw.write(" ");
											bw.close();
										}
										catch (IOException e) {
											System.out.println("Error parsing file "+ fileScores);
											return;
										}

										File filePrediction = new File(predictionDir + "/" + cls + "/" + t + "/predict_" + cls + "_" + scale + suffix + "_train_null.txt");
										System.out.println("write " + filePrediction.getAbsolutePath());
										try {
											filePrediction.getParentFile().mkdirs();
											OutputStream ops = new FileOutputStream(filePrediction); 
											OutputStreamWriter opsr = new OutputStreamWriter(ops);
											BufferedWriter bw = new BufferedWriter(opsr);
											bw.write(" ");
											bw.close();
										}
										catch (IOException e) {
											System.out.println("Error parsing file "+ filePrediction);
											return;
										}

										filePrediction = new File(predictionDir + "/" + cls + "/" + t + "/predict_" + cls + "_" + scale + suffix + "_test_null.txt");
										System.out.println("write " + filePrediction.getAbsolutePath());
										try {
											filePrediction.getParentFile().mkdirs();
											OutputStream ops = new FileOutputStream(filePrediction); 
											OutputStreamWriter opsr = new OutputStreamWriter(ops);
											BufferedWriter bw = new BufferedWriter(opsr);
											bw.write(" ");
											bw.close();
										}
										catch (IOException e) {
											System.out.println("Error parsing file "+ filePrediction);
											return;
										}

									}
									System.out.println("\n");
								}
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

	public static boolean removeOverlap(List<BagImageRegion> l, LatentRankingInput<BagImageRegion, Integer> x, double overlapMax) {
		int nbInstances = 0;
		System.out.print("remove features overlap " + overlapMax + "\t");
		if(l.size() != x.getNumberOfExamples()) {
			System.out.println("\tError - 2 different size " + l.size() + " vs " + x.getNumberOfExamples());
			System.exit(0);
		}

		boolean stop = false;
		for(int i=0; i<x.getNumberOfExamples(); i++) {

			BagImageRegion bag = x.getExample(i);

			//System.out.println(ts.input.x.getName());
			//System.out.println(l.get(i).getName());
			if(bag.getName().compareToIgnoreCase(l.get(i).getName()) == 0) {
				bag.removeOverlap(l.get(i).getRegion(0), overlapMax);

				nbInstances += bag.numberOfInstances();
				if(bag.numberOfInstances() == 0) {
					stop = true;
				}
			}
			else {
				System.out.println("ERROR name " + bag.getName() + "\t" + l.get(i).getName());
				System.exit(0);
			}
		}
		System.out.println("\t nb bags: " + x.getNumberOfExamples() + "\tnb instances: " + nbInstances + "\tnb moyen instances: " + (nbInstances/x.getNumberOfExamples()));
		return stop;

	}
}
