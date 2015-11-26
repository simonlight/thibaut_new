package jstruct.data.mit67.iccv15.big;

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
import fr.durandt.jstruct.latent.lssvm.multiclass.FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * Tests of LSSVM on UIUC Sports with deep features
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class TestMulticlassLSSVMCuttingPlane1SlackBagImage {

	public static String simDir = "/home/durandt/simulation/MIT67/";

	public static void main(String[] args) {

		int numWords = 4096;

		double[] lambdaCV = {1e-6};
		double[] epsilonCV = {1e-2};
		//Integer[] scaleCV = {100,90,80,70,60,50,40,30};
		Integer[] scaleCV = {Integer.parseInt(args[0])};
		//int[] splitCV = {1,2,3,4,5};
		int[] splitCV = {1};

		int cpmax = 500;
		int cpmin = 5;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean recompute = false;
		String features = "places";

		for(int scale : scaleCV) {
			for(int split : splitCV) {

				String cls = String.valueOf(split);

				String classifierDir = simDir + "/ICCV15/classifier/LSSVM/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/BagImageRegionC/";
				String predictionDir = simDir + "/ICCV15/prediction/LSSVM/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/BagImageRegionC/";
				String scoreDir = simDir + "/ICCV15/scores/LSSVM/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/BagImageRegionC/";
				String scoreDirR = simDir + "/ICCV15/scores_regions/LSSVM/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/BagImageRegionC/";
				String inputDir = simDir + "Split_" + cls + "/files_BagImageRegion/";

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);

				List<STrainingSample<BagImageRegion, Integer>> listTrain = BagReader.readBagImageRegion(inputDir + "/multiclass_" + features + "_train_scale_" + scale + ".txt", numWords, true, true, null, false, 0);
				List<STrainingSample<BagImageRegion, Integer>> listTest = BagReader.readBagImageRegion(inputDir + "/multiclass_" + features + "_test_scale_" + scale + ".txt", numWords, true, true, null, false, 0);

				boolean compute = false;
				for(double epsilon : epsilonCV) {
					for(double lambda : lambdaCV) {

						FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion classifier = new FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion(); 
						classifier.setLambda(lambda);
						classifier.setEpsilon(epsilon);
						classifier.setCpmax(cpmax);
						classifier.setCpmin(cpmin);
						classifier.setVerbose(1);

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

						for(STrainingSample<BagImageRegion, Integer> ts : listTrain) {
							String name = ts.input.getName();
							String[] tmp = name.split("/");
							fileScores = new File(scoreDirR + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_train/" + tmp[tmp.length-2] + "/" + tmp[tmp.length-1] + ".txt");
							if(!fileScores.exists()) {
								compute = true;
							}
						}

						for(STrainingSample<BagImageRegion, Integer> ts : listTest) {
							String name = ts.input.getName();
							String[] tmp = name.split("/");
							fileScores = new File(scoreDirR + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_test/" + tmp[tmp.length-2] + "/" + tmp[tmp.length-1] + ".txt");
							if(!fileScores.exists()) {
								compute = true;
							}
						}
					}
				}

				if(compute || recompute) {
					listTrain = BagReader.readBagImageRegion(inputDir + "/multiclass_" + features + "_train_scale_" + scale + ".txt", numWords, true, true, null, true, 0);
					List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>> exampleTrain = new ArrayList<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>>();
					for(int i=0; i<listTrain.size(); i++) {
						exampleTrain.add(new STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>(new LatentRepresentation<BagImageRegion, Integer>(listTrain.get(i).input,0), listTrain.get(i).output));
					}

					listTest = BagReader.readBagImageRegion(inputDir + "/multiclass_" + features + "_test_scale_" + scale + ".txt", numWords, true, true, null, true, 0);
					List<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>> exampleTest = new ArrayList<STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>>();
					for(int i=0; i<listTest.size(); i++) {
						exampleTest.add(new STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer>(new LatentRepresentation<BagImageRegion, Integer>(listTest.get(i).input,0), listTest.get(i).output));
					}

					for(double epsilon : epsilonCV) {
						for(double lambda : lambdaCV) {

							FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion classifier = new FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion(); 
							classifier.setLambda(lambda);
							classifier.setEpsilon(epsilon);
							classifier.setCpmax(cpmax);
							classifier.setCpmin(cpmin);
							classifier.setVerbose(1);
							classifier.setnThreads(1);

							String suffix = "_" + classifier.toString();
							File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
							if(recompute || compute && fileClassifier == null) {
								classifier.train(exampleTrain);
								double acc = classifier.accuracy(exampleTrain);
								System.err.println("train - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda );

								acc = classifier.accuracy(exampleTest);
								System.err.println("test - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda );
								System.out.println("\n");

								fileClassifier = new File(classifierDir + "/" + cls + "/" + cls + "_" + scale + suffix + "_acc_" + acc + ".ser");
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
									classifier = (FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion) ois.readObject();
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

							// scores
							for(STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer> ts : exampleTrain) {
								String name = ts.input.x.getName();
								String[] tmp = name.split("/");
								fileScores = new File(scoreDirR + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_train/" + tmp[tmp.length-2] + "/" + tmp[tmp.length-1] + ".txt");
								if(!fileScores.exists()) {
									classifier.writeScores(fileScores, ts);
								}
							}

							for(STrainingSample<LatentRepresentation<BagImageRegion, Integer>,Integer> ts : exampleTest) {
								String name = ts.input.x.getName();
								String[] tmp = name.split("/");
								fileScores = new File(scoreDirR + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_test/" + tmp[tmp.length-2] + "/" + tmp[tmp.length-1] + ".txt");
								if(!fileScores.exists()) {
									classifier.writeScores(fileScores, ts);
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
}
