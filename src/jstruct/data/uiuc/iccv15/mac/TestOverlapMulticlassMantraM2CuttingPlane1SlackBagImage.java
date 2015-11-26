package jstruct.data.uiuc.iccv15.mac;

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
public class TestOverlapMulticlassMantraM2CuttingPlane1SlackBagImage {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/UIUCSports/";

	public static void main(String[] args) {

		int numWords = 4096;

		//double[] lambdaCV = {1e-4, 1e-5, 1e-6, 1e-7};
		double[] lambdaCV = {1e-6};
		//double[] epsilonCV = {1e-2, 1e-3};
		double[] epsilonCV = {1e-2};
		//int[] scaleCV = {100,90,80,70,60,50,40,30};
		int[] scaleCV = {100};
		//int[] splitCV = {1,2,3,4,5};
		int[] splitCV = {1};

		double[] overlapCV = {0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1};
		int[] nb = {1,2,1,1,2,3,4,5};

		int cpmax = 500;
		int cpmin = 5;
		int optim = 2;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean recompute = false;
		String features = "hybrid";

		for(double overlap : overlapCV) {
			for(int s=0; s<scaleCV.length; s++) {
				int scale = scaleCV[s];
				for(int split : splitCV) {

					String cls = String.valueOf(split);

					String classifierDir = simDir + "/ICCV15/overlap/classifier/MANTRA/M2/overlap_" + overlap + "/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/BagImageRegion/";
					String predictionDir = simDir + "/ICCV15/overlap/prediction/MANTRA/M2/overlap_" + overlap + "/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/BagImageRegion/";
					String scoreDir = simDir + "/ICCV15/overlap/scores/MANTRA/M2/overlap_" + overlap + "/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/BagImageRegion/";
					String inputDir = simDir + "Split_" + cls + "/files_BagImageRegion/";

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
							}
						}
					}

					if(compute || recompute) {

						for(double epsilon : epsilonCV) {
							for(double lambda : lambdaCV) {

								List<STrainingSample<BagImageRegion, Integer>> listTrain = BagReader.readBagImageRegion(inputDir + "/multiclass_" + features + "_train_scale_" + scale + ".txt", numWords, true, true, null, true, 0);
								List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>>();
								for(int i=0; i<listTrain.size(); i++) {
									exampleTrain.add(new STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>(new LatentRepresentation<BagImageRegion, Integer>(listTrain.get(i).input,0), listTrain.get(i).output));
								}

								List<STrainingSample<BagImageRegion, Integer>> listTest = BagReader.readBagImageRegion(inputDir + "/multiclass_" + features + "_test_scale_" + scale + ".txt", numWords, true, true, null, true, 0);
								List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>>();
								for(int i=0; i<listTest.size(); i++) {
									exampleTest.add(new STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>(new LatentRepresentation<BagImageRegion, Integer>(listTest.get(i).input,0), listTest.get(i).output));
								}


								FastMulticlassMantraCuttingPlane1SlackBagImageRegion classifier = new FastMulticlassMantraCuttingPlane1SlackBagImageRegion(); 
								classifier.setLambda(lambda);
								classifier.setEpsilon(epsilon);
								classifier.setCpmax(cpmax);
								classifier.setCpmin(cpmin);
								classifier.setVerbose(1);
								classifier.setnThreads(1);
								classifier.setOptim(optim);

								String suffix = "_" + classifier.toString();
								boolean stop = false;

								for(int t=0; t<nb[s]; t++) {
									if(!stop) {

										File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/" + t + "/", cls + "_" + scale + suffix);
										if(recompute || compute && fileClassifier == null) {
											classifier.train(exampleTrain);
											double acc = classifier.accuracy(exampleTrain);
											System.err.println("train - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda );

											acc = classifier.accuracy(exampleTest);
											System.err.println("test - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda );
											System.out.println("\n");

											fileClassifier = new File(classifierDir + "/" + cls + "/" + t + "/" + cls + "_" + scale + suffix + "_acc_" + acc + ".ser");
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

										// scores
										File fileScores = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_train.txt");
										if(!fileScores.exists()) {
											classifier.writeScores(fileScores, exampleTrain);
										}

										fileScores = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_test.txt");
										if(!fileScores.exists()) {
											classifier.writeScores(fileScores, exampleTest);
										}

										// prediction
										File filePrediction = new File(predictionDir + "/" + cls + "/" + t + "/predict_" + cls + "_" + scale + suffix + "_test.txt");
										if(!filePrediction.exists()) {
											classifier.writePrediction(filePrediction, exampleTest);
										}

										filePrediction = new File(predictionDir + "/" + cls + "/" + t + "/predict_" + cls + "_" + scale + suffix + "_train.txt");
										if(!filePrediction.exists()) {
											classifier.writePrediction(filePrediction, exampleTrain);
										}

										List<BagImageRegion> listPrediction = classifier.readPrediction(filePrediction);
										stop = removeOverlap(listPrediction, exampleTrain, overlap);
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


	public static boolean removeOverlap(List<BagImageRegion> l, List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>> lr, double overlapMax) {
		int nbInstances = 0;
		System.out.print("remove features overlap " + overlapMax + "\t");
		if(l.size() != lr.size()) {
			System.out.println("\tError - 2 different size " + l.size() + " vs " + lr.size());
			System.exit(0);
		}

		boolean stop = false;
		for(int i=0; i<lr.size(); i++) {

			STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer> ts = lr.get(i);

			//System.out.println(ts.input.x.getName());
			//System.out.println(l.get(i).getName());

			ts.input.x.removeOverlap(l.get(i).getRegion(0), overlapMax);

			nbInstances += ts.input.x.numberOfInstances();
			if(ts.input.x.numberOfInstances() == 0) {
				stop = true;
			}
		}
		System.out.println("\t nb bags: " + lr.size() + "\tnb instances: " + nbInstances + "\tnb moyen instances: " + (nbInstances/lr.size()));
		return stop;

	}
}
