/**
 * 
 */
package jstruct.data.uiuc.iccv15.big;

import java.io.File;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.text.html.Option;

import sun.tools.jar.CommandLine;
import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.data.io.ScoresReader;
import fr.durandt.jstruct.latent.lssvm.multiclass.FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.VectorOp;
import fr.durandt.jstruct.variable.BagImageRegion;
import fr.lip6.jkernelmachines.classifier.DoubleSGD;
import fr.lip6.jkernelmachines.classifier.multiclass.OneAgainstAll;
import fr.lip6.jkernelmachines.evaluation.MulticlassAccuracyEvaluator;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class CombinaisonObjectBankLinearLSSVM {

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

	public static String simDir = "/home/durandt/simulation/UIUCSports/";

	public static void main(String[] args) {

		int numWords = 4096;

		int[] scaleCV = {100,90,80,70,60,50,40,30};
		int[] splitCV = {1,2,3,4,5};
		//int[] splitCV = {1,2,3,4,5,6,7,8,9,10};
		double[] cCV = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9};

		String trainset = "train";
		String testset = "test";

		String features = "hybrid";

		double lambda = 1e-6;
		double epsilon = 1e-2;
		int cpmax = 500;
		int cpmin = 5;

		int norm = 2;
		int[] epochsCV = {10, 50, 100, 200, 500, 1000};

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

		FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion classifier = new FastMulticlassLSSVMCuttingPlane1SlackBagImageRegion(); 
		classifier.setLambda(lambda);
		classifier.setEpsilon(epsilon);
		classifier.setCpmax(cpmax);
		classifier.setCpmin(cpmin);
		classifier.setVerbose(1);

		String suffix = "_" + classifier.toString();

		double[][] acc = new double[scaleCV.length][splitCV.length];
		int n=0;
		for(int split : splitCV) {

			String cls = String.valueOf(split);

			List<List<double[]>> listScoresTrain = new ArrayList<List<double[]>>();
			List<List<double[]>> listScoresTest = new ArrayList<List<double[]>>();

			String scoreDir = simDir + "/ICCV15/scores/LSSVM/CuttingPlane1Slack/Multiclass/Fast/" + features + "_caffe_6_relu/BagImageRegion/";
			String inputDir = simDir + "Split_" + cls + "/files_BagImageRegion/";

			List<STrainingSample<BagImageRegion, Integer>> listTrain = BagReader.readBagImageRegion(inputDir + "/multiclass_" + features + "_train_scale_100.txt", numWords, true, true, null, false, 0);
			List<STrainingSample<BagImageRegion, Integer>> listTest = BagReader.readBagImageRegion(inputDir + "/multiclass_" + features + "_test_scale_100.txt", numWords, true, true, null, false, 0);

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

			List<TrainingSample<double[]>> finalScoresTrain = null;
			List<TrainingSample<double[]>> finalScoresTest = null;

			finalScoresTrain = concatenation(listScoresTrain, listTrain, norm);
			finalScoresTest = concatenation(listScoresTest, listTest, norm);


			double scoreMax = 0;

			for(double c : cCV) {
				for(int epochs : epochsCV) {
					System.out.println("lambda= " + c + "\tepochs= " + epochs);
					DoubleSGD svm = new DoubleSGD();
					svm.setLambda(c);
					svm.setEpochs(epochs);
					OneAgainstAll<double[]> mcsvm = new OneAgainstAll<double[]>(svm);

					MulticlassAccuracyEvaluator<double[]> eval = new MulticlassAccuracyEvaluator<double[]>();
					eval.setClassifier(mcsvm);
					eval.setTrainingSet(finalScoresTrain);
					eval.setTestingSet(finalScoresTest);
					eval.evaluate();
					double s = eval.getScore();
					if(s>scoreMax) {
						scoreMax = s;
					}
				}
			}

			acc[m][n] = scoreMax;

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
}
