/**
 * 
 */
package jstruct.data.voc2007.iccv15.big.ranking;

import java.io.File;
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
public class CombinaisonObjectBankLinearMantraM2PlattWithoutDifficultOverlap {

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

	public static String simDir = "/home/durandt/simulation/VOC2007/";

	public static void main(String[] args) {

		int numWords = 2048;

		int[] scaleCV = {100,90,80,70,60,50,40,30};
		//int[] scaleCV = {100};
		int[] splitCV = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
		//int[] splitCV = {0};
		//double[] cCV = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9};
		double[] cCV = {Double.parseDouble(args[2])};
		//int[] epochsCV = {5, 10, 50, 100, 200, 500, 1000};
		int[] epochsCV = {Integer.parseInt(args[3])};

		double[] aCV = {Double.parseDouble(args[1])};
		double[] bCV = {0.};

		String trainset = "train";
		String testset = "test";

		double lambda = Double.parseDouble(args[0]);
		double epsilon = 1e-3;
		int cpmax = 500;
		int cpmin = 5;
		int optim = 2;

		double[] overlapCV = {0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1};
		//int[] nb = {1,1,1,1,2,3,4,5};
		int[] nb = {1,1,1,1,1,1,1,1};

		int norm = -1;
		
		System.err.println("overlap \t" + Arrays.toString(overlapCV));
		System.err.println("nb \t" + Arrays.toString(nb));

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

		for(double overlap : overlapCV) {
			String scoreDir = simDir + "/ICCV15/overlap/scoresMaxMin/MANTRA/M2/AP/overlap_" + overlap + "/CuttingPlane1Slack/BagImageRegion/without_difficults/";
			double[] acc = new double[splitCV.length];
			for(int m=0; m<splitCV.length; m++) {

				String cls = VOC2007.getClasses()[m];

				List<List<double[]>> listScoresTrain = new ArrayList<List<double[]>>();
				List<List<double[]>> listScoresTest = new ArrayList<List<double[]>>();

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


				for(int s=0; s<scaleCV.length; s++) {
					int scale = scaleCV[s];
					for(int t=0; t<nb[s]; t++) {
						File file = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_" + trainset + ".txt");
						if(file.exists()) {
							List<double[]> l = ScoresReader.readFile(file);
							if(norm >= 1) {
								for(double[] f : l) {
									VectorOp.normL2(f);
									if(norm == 2) {
										VectorOp.mul(f,(scale/100.)*(scale/100.));
									}
								}
							}
							listScoresTrain.add(l);
						}

						file = new File(scoreDir + "/" + cls + "/" + t + "/scores_" + cls + "_" + scale + suffix + "_" + testset + ".txt");
						if(file.exists()) {
							List<double[]> l = ScoresReader.readFile(file);
							if(norm >= 1) {
								for(double[] f : l) {
									VectorOp.normL2(f);
									if(norm == 2) {
										VectorOp.mul(f,(scale/100.)*(scale/100.));
									}
								}
							}
							listScoresTest.add(l);
						}
					}
				}

				double scoreMax = 0;
				for(double a : aCV) {
					for(double b : bCV) {
						List<TrainingSample<double[]>> finalScoresTrain = null;
						List<TrainingSample<double[]>> finalScoresTest = null;

						finalScoresTrain = concatenation(listScoresTrain, listTrain, norm, a, b);
						finalScoresTest = concatenation(listScoresTest, listTest, norm, a, b);

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
								double s = AveragePrecision.getAP(scores);

								System.out.println("\tap= " + s);
								if(s>scoreMax) {
									scoreMax = s;
								}
							}
						}
						acc[m] = Math.max(scoreMax,acc[m]);
					}
				}
			}
			System.out.println("Accuracy \t" + VectorOp.mean(acc) + "\t" + Arrays.toString(acc));
			System.err.println("Accuracy \t" + overlap + "\t" + VectorOp.mean(acc) + "\t" + Arrays.toString(acc));
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
}
