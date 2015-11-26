package jstruct.data.uiuc.mac;

import java.io.File;
import java.util.Arrays;

public class Classif {

	static Options options = new Options();

	public static String classifierDir = null;

	private static double lambda = -1;
	private static double p = -1;
	private static double m = -1;
	private static int maxCCCPIter = 30;

	private static int nbSplit = 5;
	private static boolean info = true;

	public static void main(String[] args) {
		
		classifierDir = "/Volumes/Eclipse/LIP6/simulation/UIUCSports/classifier/latentML3/hybrid_caffe_6_relu/";
		//classifierDir = "/Volumes/Eclipse/LIP6/simulation/UIUCSports/classifier/ml3/hybrid_caffe_6_relu/";


		Integer[] scales = {100,90,80,70,60,50,40,30};

		double[][] scores = new double[nbSplit][scales.length];

		for(int t=0; t<nbSplit; t++) {
			String cls = String.valueOf(t+1);
			String dir = classifierDir + "/" + cls + "/";
			for(int s=0; s<scales.length; s++) {
				scores[t][s] = getMaxScoreACC(dir, cls, scales[s]);
			}
		}

		System.out.println();
		for(int t=0; t<nbSplit; t++) {
			System.out.println((t+1) + "\t" + Arrays.toString(scores[t]));
		}

		double map = 0;
		double[] scoresACC = new double[scales.length];
		for(int s=0; s<scales.length; s++) {
			map = 0;
			for(int t=0; t<nbSplit; t++) {
				map += scores[t][s];
			}
			map /= scores.length;
			scoresACC[s] = map;
		}

		System.out.println("\nACC= " + Arrays.toString(scoresACC));
		for(int s=0; s<scales.length; s++) {
			System.out.print(scoresACC[s] + "\t");
		}
		System.out.println();

		System.out.println("\n");
		for(int t=0; t<nbSplit; t++) {
			for(int s=0; s<scales.length; s++) {
				System.out.print(scores[t][s] + "\t");
			}
			System.out.println();
		}
	}

	public static double getMaxScoreACC(String dir, String cls, int scale) {

		//System.out.println("search classifier in " + dir);

		File classifierDir = new File(dir);
		double scoreMax = 0;
		String file = null;

		if(classifierDir.exists()) {
			String[] f = classifierDir.list();
			//System.out.println(Arrays.toString(f));

			for(String s : f) {
				File tp = new File(dir + "/" + s);
				if(!tp.isHidden() && tp.isFile()) {
					boolean test = true;

					String cmp = null;
					if(lambda != -1) {
						cmp = "_lambda_" + lambda + "_";
						if(!s.contains(cmp)) test = false;
					}
					if(p != -1) {
						cmp = "_p_" + p + "_";
						if(!s.contains(cmp)) test = false;
					}
					if(m != -1) {
						cmp = "_m_" + m + "_";
						if(!s.contains(cmp)) test = false;
					}
					if(maxCCCPIter != -1) {
						cmp = "_maxCCCPIter_" + maxCCCPIter + "_";
						if(!s.contains(cmp)) test = false;
					}
					if(scale != -1) {
						cmp = cls + "_" + scale + "_";
						if(!s.contains(cmp)) test = false;
					}

					if(test) {
						String[] tm = s.split(".ser")[0].split("_");
						int n=0;
						while(tm[n].compareTo("acc") != 0) {
							n++;
						}
						double ap = Double.parseDouble(tm[++n]);

						if(ap > scoreMax) {
							scoreMax = ap;
							file = s;
						}
						//System.out.println(s);
					}
				}
			}
		}
		if(info) {
			System.out.println(file);
		}

		return scoreMax;
	}
}
