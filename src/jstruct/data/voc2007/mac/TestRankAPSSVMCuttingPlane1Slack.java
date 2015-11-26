package jstruct.data.voc2007.mac;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import jstruct.data.voc2007.VOC2007;
import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.ssvm.ranking.DoubleRankAPSSVMCuttingPlane1Slack;
import fr.durandt.jstruct.ssvm.ranking.RankingInput;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.lip6.jkernelmachines.type.TrainingSample;

public class TestRankAPSSVMCuttingPlane1Slack {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/VOC2007/";

	private static int numWords = 2048;

	public static void main(String[] args) {

		double[] lambdaCV = {1e-3, 1e-4, 1e-5, 1e-6, 1e-8};
		double[] epsilonCV = {1e-2,1e-3};
		int[] scaleCV = {100};
		int[] splitCV = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
		//int[] splitCV = {0};
		int scale = 100;
		int targetLabel = 1;

		int cpmax = 500;
		int cpmin = 5;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean recompute = false;
		//String features = "hybrid";

		for(int split : splitCV) {
			String cls = VOC2007.getClasses()[split];

			String classifierDir = simDir + "classifier/ssvm_ranking/AP/matconvnet_m_" + numWords + "_layer_20/";
			String inputDir = simDir + "/files/";

			System.out.println("classifierDir: " + classifierDir + "\n");
			System.err.println("split " + split + "\t cls " + cls);

			boolean compute = false;
			for(double epsilon : epsilonCV) {
				for(double lambda : lambdaCV) {

					DoubleRankAPSSVMCuttingPlane1Slack classifier = new DoubleRankAPSSVMCuttingPlane1Slack(); 
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
				}
			}


			if(compute || recompute) {
				List<TrainingSample<BagMIL>> listTrain = BagReader.readBagMIL(inputDir + "/" + cls + "_train_scale_" + scale + "_matconvnet_m_" + numWords + "_layer_20.txt", numWords);
				List<STrainingSample<double[], Integer>> exampleTrain = new ArrayList<STrainingSample<double[], Integer>>();
				for(int i=0; i<listTrain.size(); i++) {
					exampleTrain.add(new STrainingSample<double[], Integer>(listTrain.get(i).sample.getFeature(0), listTrain.get(i).label == targetLabel ? 1 : 0));
				}

				RankingInput rankTrain = new RankingInput(exampleTrain);
				RankingOutput rankTrainY = new RankingOutput(exampleTrain);
				List<STrainingSample<RankingInput,RankingOutput>> train = new ArrayList<STrainingSample<RankingInput,RankingOutput>>();
				train.add(new STrainingSample<RankingInput,RankingOutput>(rankTrain,rankTrainY));

				List<TrainingSample<BagMIL>> listTest = BagReader.readBagMIL(inputDir + "/" + cls + "_test_scale_" + scale + "_matconvnet_m_" + numWords + "_layer_20.txt", numWords);
				List<STrainingSample<double[],Integer>> exampleTest = new ArrayList<STrainingSample<double[],Integer>>();
				for(int i=0; i<listTest.size(); i++) {
					exampleTest.add(new STrainingSample<double[], Integer>(listTest.get(i).sample.getFeature(0), listTest.get(i).label == targetLabel ? 1 : 0));
				}

				RankingInput rankTest = new RankingInput(exampleTest);
				RankingOutput rankTestY = new RankingOutput(exampleTest);
				List<STrainingSample<RankingInput,RankingOutput>> test = new ArrayList<STrainingSample<RankingInput,RankingOutput>>();
				test.add(new STrainingSample<RankingInput,RankingOutput>(rankTest,rankTestY));

				for(double epsilon : epsilonCV) {
					for(double lambda : lambdaCV) {

						DoubleRankAPSSVMCuttingPlane1Slack classifier = new DoubleRankAPSSVMCuttingPlane1Slack(); 
						classifier.setLambda(lambda);
						classifier.setEpsilon(epsilon);
						classifier.setCpmax(cpmax);
						classifier.setCpmin(cpmin);
						classifier.setVerbose(1);

						String suffix = "_" + classifier.toString();
						File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
						if(recompute || compute && fileClassifier == null) {
							classifier.train(train);
							double ap = classifier.averagePrecision(train);
							System.err.println("train - " + cls + "\tscale= " + scale + "\tap= " + ap + "\tlambda= " + lambda );

							ap = classifier.averagePrecision(test);
							System.err.println("test - " + cls + "\tscale= " + scale + "\tap= " + ap + "\tlambda= " + lambda );
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