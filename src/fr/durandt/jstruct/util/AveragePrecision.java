package fr.durandt.jstruct.util;

import java.util.Collections;
import java.util.List;

public class AveragePrecision {

	/**
	 * Compute the average precision. The key variable of Pair is the label of the example,
	 * and the value variable is the score.
	 * @param l list of examples with the label (key) and the score (value)
	 * @return average precision
	 */
	public static double getAP(List<Pair<Integer,Double>> l) {
		
		if(l == null)
			return Double.NaN;
		
		// Sort the examples per decreasing scores 
		Collections.sort(l, Collections.reverseOrder());
		
		int[] tp = new int[l.size()];
		int[] fp = new int[l.size()];
		
		int i = 0;
		int cumtp = 0, cumfp = 0;
		int totalpos = 0;
		
		//cumsum of true positives and false positives
		for(Pair<Integer,Double> e : l) {
			if(e.getKey() == 1) {
				cumtp++;
				totalpos++;
			}
			else {
				cumfp++;
			}
			tp[i] = cumtp;
			fp[i] = cumfp;
			i++;
		}
		
		//precision / recall
		double[] prec = new double[tp.length];
		double[] reca = new double[tp.length];
		
		for(i = 0 ; i < tp.length ; i++) {
			reca[i] = ((double)tp[i])/((double)totalpos);
			prec[i] = ((double)tp[i])/((double)(tp[i]+fp[i]));
		}
		
		double[] mrec = new double[reca.length+2];
		for(int j=0; j<reca.length; j++) {
			mrec[j+1] = reca[j];
		}
		mrec[mrec.length-1] = 1;
		
		double[] mpre = new double[prec.length+2];
		for(int j=0; j<prec.length; j++) {
			mpre[j+1] = prec[j];
		}
		
		for(int j=mpre.length-2; j>=0; j--) {
		    mpre[j] = Math.max(mpre[j],mpre[j+1]);
		}
		
		double map = 0.;
		for(int j=1; j<mpre.length-1; j++) {
			map += (mrec[j]-mrec[j-1])*mpre[j];
		}
		
		return map;
	}
}
