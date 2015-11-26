package jstruct.data.voc2011.action.mac;

import java.io.File;
import java.util.Arrays;

import jstruct.data.voc2011.VOC2011;
import fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingAPMantraM2CuttingPlane1SlackBagImageRegion;
import fr.durandt.jstruct.util.VectorOp;

public class ClassifyRankingAPMantraCuttingPlane1Slack {

	public static String simDir = "/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr_2013_tutoriel/";

	public static void main(String[] args) {

		double[] lambdaCV = {1e4, 1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6};
		//double[] lambdaCV = {1e-2, 1e-3, 1e-4, 1e-5};
		//double[] lambdaCV = {1e4};
		//double[] epsilonCV = {1e-2, 1e-3};
		double[] epsilonCV = {1e-3};
		int[] scaleCV = {100};
		int[] splitCV = {1,2,3,4,5};
		//int[] splitCV = {0};

		int cpmax = 500;
		int cpmin = 5;
		int optim = 2;

		System.out.println("lambda " + Arrays.toString(lambdaCV));
		System.out.println("epsilon " + Arrays.toString(epsilonCV));
		System.out.println("scale " + Arrays.toString(scaleCV));
		System.out.println("split " + Arrays.toString(splitCV) + "\n");

		double[][] map = new double[VOC2011.getActionClasses().length][splitCV.length];
		double[] mapAll = new double[VOC2011.getActionClasses().length];
		
		double[][] bestLambda = new double[VOC2011.getActionClasses().length][splitCV.length];

		for(int iCls=0; iCls<VOC2011.getActionClasses().length; iCls++) {
			String cls = VOC2011.getActionClasses()[iCls];
			for(int split : splitCV) {			

				//String classifierDir = simDir + "classifier/Mantra/M2/AP/";
				String classifierDir = simDir + "/ICCV15_2/classifier/Mantra/M2/AP/";

				System.out.println("classifierDir: " + classifierDir + "\n");
				System.err.println("split " + split + "\t cls " + cls);

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
						File fileClassifier = testPresenceFile(classifierDir + "/" + cls + "/", cls + "_" + split + suffix);
						if(fileClassifier != null) {
							String[] tm = fileClassifier.getAbsolutePath().split(".ser")[0].split("_");
							int n=0;
							while(tm[n].compareTo("ap") != 0) {
								n++;
							}
							double ap = Double.parseDouble(tm[++n]);
							if(ap > map[iCls][split-1]) {
								map[iCls][split-1] = ap;
								bestLambda[iCls][split-1] = lambda;
							}
						}
					}
				}
			}
			mapAll[iCls] = VectorOp.mean(map[iCls]);
		}

		for(int iCls=0; iCls<VOC2011.getActionClasses().length; iCls++) {
			System.out.println(Arrays.toString(map[iCls]));
		}
		
		double[] meanSplit = new double[splitCV.length];
		for(int split : splitCV) {
			double[] tmp = new double[VOC2011.getActionClasses().length];
			for(int iCls=0; iCls<VOC2011.getActionClasses().length; iCls++) {
				tmp[iCls] = map[iCls][split-1];
			}
			meanSplit[split-1] += VectorOp.mean(tmp);
		}

		System.out.println("map= " + VectorOp.mean(meanSplit) + " +/- " + VectorOp.stddev(meanSplit) + "\t" + Arrays.toString(mapAll));
	
		System.out.println("best lambda ");
		for(int i=0; i<bestLambda.length; i++) {
			for(int j=0; j<bestLambda[i].length; j++) {
				System.out.print(bestLambda[i][j] + "\t");
			}
			System.out.println();
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