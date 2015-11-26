/**
 * 
 */
package jstruct.data.voc2007.iccv15.mac.ranking;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.text.html.Option;

import jstruct.data.voc2007.VOC2007;
import sun.tools.jar.CommandLine;
import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.data.io.ScoresReader;
import fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingAPMantraM2CuttingPlane1SlackBagImageRegion;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.AveragePrecision;
import fr.durandt.jstruct.util.Pair;
import fr.durandt.jstruct.util.VectorOp;
import fr.durandt.jstruct.variable.BagImageRegion;
import fr.lip6.jkernelmachines.classifier.DoubleSGD;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class CombinaisonObjectBankLinearMantraM2PlattWithoutDifficult {

	static Option epochsOption = OptionBuilder.withArgName("epochs")
			.hasArg()
			.withDescription("epochs")
			.withLongOpt("epochs")
			.create("e");
	static Option normOption = OptionBuilder.withArgName("norm")
			.hasArg()
			.withDescription("norm")
			.withLongOpt("norm")
			.create("n");
	static Option dirOption = OptionBuilder.withArgName("scores directory")
			.hasArg()
			.withDescription("dir")
			.withLongOpt("dir")
			.create("d");
	static Option lambdaOption = OptionBuilder.withArgName("lambda parameter")
			.hasArg()
			.withDescription("lambda")
			.withLongOpt("lambda")
			.create("l");
	static Option epsilonOption = OptionBuilder.withArgName("epsilon parameter")
			.hasArg()
			.withDescription("epsilon")
			.withLongOpt("epsilon")
			.create("eps");

	static Options options = new Options();

	static {
		options.addOption(dirOption);
		options.addOption(lambdaOption);
		options.addOption(epochsOption);
		options.addOption(normOption);
		options.addOption(epsilonOption);
	}

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/VOC2007/";

	public static void main(String[] args) {

		int numWords = 2048;

		int[] scaleCV = {100,90,80,70,60,50,40,30};
		//int[] scaleCV = {100};
		int[] splitCV = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
		//int[] splitCV = {0};
		//double[] cCV = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9};
		double[] cCV = {1e-5};
		//int[] epochsCV = {5, 10, 50, 100, 200, 500, 1000};
		int[] epochsCV = {500};

		double[] aCV = {0.1};
		double[] bCV = {0.};

		String trainset = "train";
		String testset = "test";

		double lambda = 1e-4;
		double epsilon = 1e-3;
		int cpmax = 500;
		int cpmin = 5;
		int optim = 2;

		int norm = -1;

		// Option parsing		
		// Create the parser
		CommandLineParser parser = new GnuParser();
		try {
			// parse the command line arguments
			CommandLine line = parser.parse( options, args );

			if(line.hasOption("dir")) {
				//dir = line.getOptionValue("d");
			}
			if(line.hasOption("lambda")) {
				lambda = Double.parseDouble(line.getOptionValue("l"));
			}
			if(line.hasOption("norm")) {
				norm = Integer.parseInt(line.getOptionValue("n"));
			}
			if(line.hasOption("epsilon")) {
				epsilon = Double.parseDouble(line.getOptionValue("eps"));
			}
		}
		catch( ParseException exp ) {
			// oops, something went wrong
			System.err.println( "Parsing failed.  Reason: " + exp.getMessage() );
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp( "Parameters", options );
			System.exit(-1);
		}

		RankingAPMantraM2CuttingPlane1SlackBagImageRegion classifier = new RankingAPMantraM2CuttingPlane1SlackBagImageRegion();
		classifier.setLambda(lambda);
		classifier.setEpsilon(epsilon);
		classifier.setCpmax(cpmax);
		classifier.setCpmin(cpmin);
		classifier.setVerbose(1);
		classifier.setnThreads(1);
		classifier.setOptim(optim);

		String suffix = "_" + classifier.toString();

		double[][] acc = new double[scaleCV.length][splitCV.length];
		int n=0;
		for(int split : splitCV) {

			String cls = VOC2007.getClasses()[split];

			List<List<double[]>> listScoresTrain = new ArrayList<List<double[]>>();
			List<List<double[]>> listScoresTest = new ArrayList<List<double[]>>();

			String scoreDir = simDir + "/ICCV15/scoresMaxMin/MANTRA/M2/AP/CuttingPlane1Slack/BagImageRegion/without_difficults/";
			//String finalScoreDir = simDir + "/ICCV15/final_scores/MANTRA/M2/AP/CuttingPlane1Slack/BagImageRegion/without_difficults/";
			String finalScoreDir = simDir + "/results/Main/";
			String inputDir = simDir + "/files_BagImageRegion/without_difficults/";

			List<STrainingSample<BagImageRegion, Integer>> listTrain = BagReader.readBagImageRegion(inputDir + "/" + cls + "_" + trainset + "_matconvnet_m_" + numWords + "_layer_20_scale_100.txt", numWords, true, true, null, false, 0);
			List<STrainingSample<BagImageRegion, Integer>> listTest = BagReader.readBagImageRegion(inputDir + "/" + cls + "_" + testset + "_matconvnet_m_" + numWords + "_layer_20_scale_100.txt", numWords, true, true, null, false, 0);

			for(STrainingSample<BagImageRegion, Integer> ts : listTrain) {
				if(ts.output == 0) {
					ts.output = -1;
				}
			}
			for(STrainingSample<BagImageRegion, Integer> ts : listTest) {
				if(ts.output == 0) {
					ts.output = -1;
				}
			}

			int m=0;
			for(int scale : scaleCV) {
				File file = new File(scoreDir + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_" + trainset + ".txt");
				List<double[]> l = ScoresReader.readFile(file);
				if(norm >= 1) {
					for(double[] t : l) {
						VectorOp.normL2(t);
						if(norm == 2) {
							VectorOp.mul(t,(scale/100.)*(scale/100.));
						}
					}
				}
				listScoresTrain.add(l);
				l = null;

				file = new File(scoreDir + "/" + cls + "/scores_" + cls + "_" + scale + suffix + "_" + testset + ".txt");
				l = ScoresReader.readFile(file);
				if(norm >= 1) {
					for(double[] t : l) {
						VectorOp.normL2(t);
						if(norm == 2) {
							VectorOp.mul(t,(scale/100.)*(scale/100.));
						}
					}
				}
				listScoresTest.add(l);
			}

			double scoreMax = 0;
			for(double a : aCV) {
				for(double b : bCV) {
					List<TrainingSample<double[]>> finalScoresTrain = null;
					List<TrainingSample<double[]>> finalScoresTest = null;

					finalScoresTrain = concatenation(listScoresTrain, listTrain, norm, a, b);
					finalScoresTest = concatenation(listScoresTest, listTest, norm, a, b);

					for(TrainingSample<double[]> ts : finalScoresTrain) {
						//System.out.println(ts.label + "\t" + Arrays.toString(ts.sample));
					}

					for(double c : cCV) {
						for(int epochs : epochsCV) {
							System.out.print("lambda= " + c + "\tepochs= " + epochs);
							DoubleSGD svm = new DoubleSGD();
							svm.setLambda(c);
							svm.setEpochs(epochs);

							/*ApEvaluator<double[]> eval = new ApEvaluator<double[]>(svm, finalScoresTrain, finalScoresTest);
							//eval.evaluate();
							//System.out.println(cls + "\tscore= " + eval.getScore() + "\tc= " + c + "\ttrain");
							//eval = new ApEvaluator<double[]>(svm, finalScoresTrain, finalScoresTest);
							eval.evaluate();
							double s = eval.getScore();
							 */

							svm.train(finalScoresTrain);
							List<Pair<Integer,Double>> scores = new ArrayList<Pair<Integer,Double>>();
							for(TrainingSample<double[]> ts : finalScoresTest) {
								scores.add(new Pair<Integer,Double>(ts.label, svm.valueOf(ts.sample)));
							}
							

							File file = new File(finalScoreDir + "/comp2_cls_test_" + cls + ".txt");
							writeScores(file,listTest,scores);
							
							double s = AveragePrecision.getAP(scores);

							System.out.println("\tap= " + s);
							if(s>scoreMax) {
								scoreMax = s;
							}
							
						}
					}

					acc[m][n] = scoreMax;
					System.err.println(cls + " \t ap= " + scoreMax);
				}
			}
			n++;
		}
		System.out.println("Accuracy");
		for(double[] tab : acc) {
			System.out.println(VectorOp.mean(tab) + "\t" + Arrays.toString(tab));
		}
	}

	public static List<TrainingSample<double[]>> concatenation(List<List<double[]>> listScores, List<STrainingSample<BagImageRegion,Integer>> listBag, int norm) {

		List<TrainingSample<double[]>> list = new ArrayList<TrainingSample<double[]>>();
		int nb = listScores.size();
		int dim = listScores.get(0).get(0).length;
		for(int i=0; i<listScores.get(0).size(); i++) {
			double[] scores = new double[dim * nb];
			for(int j=0; j<nb; j++) {
				for(int k=0; k<dim; k++) {
					scores[j*dim + k] = listScores.get(j).get(i)[k];
				}
			}
			//System.out.println(i + "\t" + Arrays.toString(scores));
			if(norm >= 0) {
				VectorOp.normL2(scores);
			}
			list.add(new TrainingSample<double[]>(scores, listBag.get(i).output));
		}
		System.out.println("Concatenation - " + list.size() + " x " + list.get(0).sample.length);
		return list;
	}

	/**
	 * Platt
	 * @param listScores
	 * @param listBag
	 * @param norm
	 * @return
	 */
	public static List<TrainingSample<double[]>> concatenation(List<List<double[]>> listScores, List<STrainingSample<BagImageRegion,Integer>> listBag, int norm, double a, double b) {

		List<TrainingSample<double[]>> list = new ArrayList<TrainingSample<double[]>>();
		int nb = listScores.size();
		int dim = listScores.get(0).get(0).length;
		for(int i=0; i<listScores.get(0).size(); i++) {
			double[] scores = new double[dim * nb];
			for(int j=0; j<nb; j++) {
				for(int k=0; k<dim; k++) {
					scores[j*dim + k] = 1./(1+Math.exp(a*listScores.get(j).get(i)[k] + b));
				}
			}
			//System.out.println(i + "\t" + Arrays.toString(scores));
			if(norm >= 0) {
				VectorOp.normL2(scores);
			}
			list.add(new TrainingSample<double[]>(scores, listBag.get(i).output));
		}
		System.out.println("Concatenation - " + list.size() + " x " + list.get(0).sample.length);
		return list;
	}

	public static void writeScores(File file, List<STrainingSample<BagImageRegion, Integer>> list, List<Pair<Integer,Double>> scores) {

		System.out.println("Write scores file " + file.getAbsolutePath());

		// Create the directory if not exist
		file.getAbsoluteFile().getParentFile().mkdirs();

		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);

			for(int i=0; i<list.size(); i++) {
				String[] tmp = list.get(i).input.getName().split("/");
				tmp = tmp[tmp.length-1].split(".jpg");
				bw.write(tmp[0] + "\t" + scores.get(i).getValue());
				bw.write("\n");
			}

			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			e.printStackTrace();
			return;
		}

	}
}
