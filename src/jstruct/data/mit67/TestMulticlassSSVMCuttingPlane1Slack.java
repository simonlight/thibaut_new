package jstruct.data.mit67;

import java.io.File;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.text.html.Option;

import sun.tools.jar.CommandLine;
import fr.durandt.jstruct.data.io.BagReader;
import fr.durandt.jstruct.ssvm.multiclass.DoubleMulticlassSSVMCuttingPlane1Slack;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * Tests of ML3 on MIT67 with deep features
 * @author Thibaut Durand <durand.tibo@gmail.com>
 *
 */
public class TestMulticlassSSVMCuttingPlane1Slack {

	static Option cOption = OptionBuilder.withArgName("regularization parameter C")
			.hasArg()
			.withDescription("c value")
			.withLongOpt("c")
			.create("c");
	static Option initOption = OptionBuilder.withArgName("init type")
			.hasArg()
			.withDescription("init")
			.withLongOpt("init")
			.create("i");
	static Option optimOption = OptionBuilder.withArgName("optim")
			.hasArg()
			.withDescription("optim")
			.withLongOpt("optim")
			.create("o");
	static Option cpmaxOption = OptionBuilder.withArgName("cutting plane")
			.hasArg()
			.withDescription("maximum number of cutting plane")
			.withLongOpt("cuttingPlaneMax")
			.create("cpmax");
	static Option cpminOption = OptionBuilder.withArgName("cutting plane")
			.hasArg()
			.withDescription("minimum number of cutting plane")
			.withLongOpt("cuttingPlaneMax")
			.create("cpmin");
	static Option epsilonOption = OptionBuilder.withArgName("epsilon")
			.hasArg()
			.withDescription("epsilon")
			.withLongOpt("epsilon")
			.create("eps");

	static Option scaleOption = OptionBuilder.withArgName("scale")
			.hasArg()
			.withDescription("scale")
			.withLongOpt("scale")
			.create("s");
	static Option splitOption = OptionBuilder.withArgName("slit")
			.hasArg()
			.withDescription("split")
			.withLongOpt("split")
			.create("sp");
	static Option numWordsOption = OptionBuilder.withArgName("numWords")
			.hasArg()
			.withDescription("numWords")
			.withLongOpt("numWords")
			.create("w");

	static Options options = new Options();

	static {
		options.addOption(cOption);
		options.addOption(initOption);
		options.addOption(optimOption);
		options.addOption(cpmaxOption);
		options.addOption(cpminOption);
		options.addOption(epsilonOption);
		options.addOption(scaleOption);
		options.addOption(splitOption);
		options.addOption(numWordsOption);
	}

	private static int cpmax = 500;
	private static int cpmin = 5;
	private static double lambda = 1e-4;
	private static int init = 0;
	private static int optim = 1;
	private static double epsilon = 1e-2;

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/MIT67/";

	public static int split = 1;
	public static int scale = 100;
	private static int numWords = 4096;

	public static void main(String[] args) {

		// Option parsing		
		// Create the parser
		CommandLineParser parser = new GnuParser();
		try {
			// parse the command line arguments
			CommandLine line = parser.parse( options, args );

			if(line.hasOption("init")) {
				init = Integer.parseInt(line.getOptionValue("i"));
			}
			if(line.hasOption("optim")) {
				optim = Integer.parseInt(line.getOptionValue("o"));
			}
			if(line.hasOption("cuttingPlaneMax")) {
				cpmax = Integer.parseInt(line.getOptionValue("cpmax"));
			}
			if(line.hasOption("cuttingPlaneMin")) {
				cpmin = Integer.parseInt(line.getOptionValue("cpmin"));
			}

			if(line.hasOption("numWords")) {
				numWords = Integer.parseInt(line.getOptionValue("w"));
			}

		}
		catch(ParseException exp) {
			// oops, something went wrong
			System.err.println( "Parsing failed.  Reason: " + exp.getMessage() );
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp( "Parameters", options );
			System.exit(-1);
		}

		double[] lambdaCV = {1e-3};
		double[] epsilonCV = {1e-2};
		int scale = 100;
		//int[] splitCV = {1,2,3,4,5};
		int[] splitCV = {1};

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + scale);
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		boolean compute = true;
		String features = "places";

		String cls = "multiclass";

		String classifierDir = simDir + "classifier/ssvm/" + features + "_caffe_6_relu/";
		String inputDir = simDir + "/files_BagImage/";

		System.out.println("classifierDir: " + classifierDir + "\n");
		System.err.println("cls " + cls);

		for(double epsilon : epsilonCV) {
			for(double lambda : lambdaCV) {

				DoubleMulticlassSSVMCuttingPlane1Slack svm = new DoubleMulticlassSSVMCuttingPlane1Slack();
				//DoubleFastMulticlassSSVMCuttingPlane1Slack svm = new DoubleFastMulticlassSSVMCuttingPlane1Slack(); 
				svm.setLambda(lambda);
				svm.setCpmax(cpmax);
				svm.setCpmin(cpmin);
				svm.setEpsilon(epsilon);

				String suffix = "_" + svm.toString();
				boolean testFile = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
				if(!testFile) {
					compute = true;
				}
			}
		}

		if(compute) {
			List<TrainingSample<BagMIL>> listTrain = BagReader.readBagMIL(inputDir + "/multiclass_" + features + "_train_scale_" + scale + ".txt", numWords);
			List<STrainingSample<double[], Integer>> exampleTrain = new ArrayList<STrainingSample<double[], Integer>>();
			for(int i=0; i<listTrain.size(); i++) {
				exampleTrain.add(new STrainingSample<double[], Integer>(listTrain.get(i).sample.getFeature(0), listTrain.get(i).label));
			}

			List<TrainingSample<BagMIL>> listTest = BagReader.readBagMIL(inputDir + "/multiclass_" + features + "_test_scale_" + scale + ".txt", numWords);
			List<STrainingSample<double[],Integer>> exampleTest = new ArrayList<STrainingSample<double[],Integer>>();
			for(int i=0; i<listTest.size(); i++) {
				exampleTest.add(new STrainingSample<double[], Integer>(listTest.get(i).sample.getFeature(0), listTest.get(i).label));
			}

			for(double epsilon : epsilonCV) {
				for(double lambda : lambdaCV) {

					DoubleMulticlassSSVMCuttingPlane1Slack svm = new DoubleMulticlassSSVMCuttingPlane1Slack();
					//DoubleFastMulticlassSSVMCuttingPlane1Slack svm = new DoubleFastMulticlassSSVMCuttingPlane1Slack(); 
					svm.setLambda(lambda);
					svm.setCpmax(cpmax);
					svm.setCpmin(cpmin);
					svm.setEpsilon(epsilon);
					svm.setnThreads(1);

					String suffix = "_" + svm.toString();
					boolean testFile = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + scale + suffix);
					if(compute || !testFile) {
						svm.train(exampleTrain);
						double acc = svm.multiclassAccuracy(exampleTrain);
						System.err.println("train - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda + "\tepsilon= " + epsilon);

						acc = svm.multiclassAccuracy(exampleTest);
						System.err.println("test - " + cls + "\tscale= " + scale + "\tacc= " + acc + "\tlambda= " + lambda + "\tepsilon= " + epsilon);
						System.out.println("\n");
					}
				}
			}
		}
	}

	public static boolean testPresenceFile(String dir, String test) {
		boolean testPresence = false;

		File classifierDir = new File(dir);

		if(classifierDir.exists()) {
			String[] f = classifierDir.list();
			//System.out.println(Arrays.toString(f));

			for(String s : f) {
				if(s.contains(test)) {
					testPresence = true;
				}
			}
		}
		System.out.println("presence " + testPresence + "\t" + dir + "\t" + test);
		return testPresence;
	}
}
