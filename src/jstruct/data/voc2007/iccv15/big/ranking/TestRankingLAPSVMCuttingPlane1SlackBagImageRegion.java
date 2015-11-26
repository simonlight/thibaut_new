/**
 * 
 */
package jstruct.data.voc2007.iccv15.big.ranking;

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
import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.ranking.LAPSVMCuttingPlane1SlackBagImageRegion;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInputBagImageRegion;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class TestRankingLAPSVMCuttingPlane1SlackBagImageRegion {

	public static String simDir = "/home/durandt/simulation/VOC2007/";

	private static int numWords = 2048;

	public static void main(String[] args) {

		double[] lambdaCV = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
		double[] epsilonCV = {1e-2, 1e-3};
		//int[] splitCV = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
		int[] splitCV = {Integer.parseInt(args[0])};
		Integer[] scaleCV = {100,90,80,70,60,50,40,30};
		int targetLabel = 1;

		int cpmax = 500;
		int cpmin = 5;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean recompute = false;

		for(int scale : scaleCV) {
			for(int split : splitCV) {
				String cls = VOC2007.getClasses()[split];

				String classifierDir = simDir + "/ICCV15/classifier/lapsvm/CuttingPlane1Slack/";
				String inputDir = simDir + "/files_BagImageRegion/";

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);

				boolean compute = false;
				for(double epsilon : epsilonCV) {
					for(double lambda : lambdaCV) {

						LAPSVMCuttingPlane1SlackBagImageRegion classifier = new LAPSVMCuttingPlane1SlackBagImageRegion();
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

					List<STrainingSample<BagImageRegion, Integer>> listTrain = BagReader.readBagImageRegion(inputDir + "/" + cls + "_train_matconvnet_m_" + numWords + "_layer_20_scale_" + scale + ".txt", numWords, true, true, null, true, 0);

					LatentRankingInputBagImageRegion rankTrain = new LatentRankingInputBagImageRegion(listTrain);
					RankingOutput rankTrainY = new RankingOutput();
					List<Integer> labels = new ArrayList<Integer>();
					List<Integer> latent = new ArrayList<Integer>();
					for(int i=0; i<listTrain.size(); i++) {
						labels.add(listTrain.get(i).output == targetLabel ? 1 : 0);
						latent.add(i,0);
					}
					rankTrainY.initialize(labels);

					List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<Integer>>, RankingOutput>> train = new ArrayList<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<Integer>>, RankingOutput>>();
					train.add(new STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<Integer>>, RankingOutput>(new LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>,List<Integer>>(rankTrain, latent), rankTrainY));

					List<STrainingSample<BagImageRegion, Integer>> listTest = BagReader.readBagImageRegion(inputDir + "/" + cls + "_test_matconvnet_m_" + numWords + "_layer_20_scale_" + scale + ".txt", numWords, true, true, null, true, 0);

					LatentRankingInputBagImageRegion rankTest = new LatentRankingInputBagImageRegion(listTest);
					RankingOutput rankTestY = new RankingOutput();
					labels = new ArrayList<Integer>();
					latent = new ArrayList<Integer>();
					for(int i=0; i<listTest.size(); i++) {
						labels.add(listTest.get(i).output == targetLabel ? 1 : 0);
						latent.add(i,0);
					}
					rankTestY.initialize(labels);

					List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<Integer>>, RankingOutput>> test = new ArrayList<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<Integer>>, RankingOutput>>();
					test.add(new STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<Integer>>, RankingOutput>(new LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>,List<Integer>>(rankTest, latent), rankTestY));


					for(double epsilon : epsilonCV) {
						for(double lambda : lambdaCV) {

							LAPSVMCuttingPlane1SlackBagImageRegion classifier = new LAPSVMCuttingPlane1SlackBagImageRegion();
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
