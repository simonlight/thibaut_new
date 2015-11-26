/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.ranking;

import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class ROCAreaLSSVMCuttingPlane1SlackBagImageRegion extends ROCAreaLSSVMCuttingPlane1Slack<BagImageRegion,Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6569169213857115207L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.ranking.LAPSVMCuttingPlane1Slack#latentPrediction(java.lang.Object, double[])
	 */
	@Override
	protected Integer latentPrediction(BagImageRegion x, double[] w) {
		double max = -Double.MAX_VALUE;
		int hpredict = -1; // Latent prediction
		for(int h=0; h<x.numberOfInstances(); h++) {	// For each region
			// Compute the score of region h
			double score = linear.valueOf(w, x.getInstance(h));
			if(score > max) {
				max = score;
				hpredict = h;
			}
		}
		return hpredict;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<Integer>>, RankingOutput>> l) {

		// Print the classes and the number of relevant/irrelevant examples
		System.out.println("Rank ROCArea LSSVM \t P= " + l.get(0).input.x.getNpos() + "\t N= " + l.get(0).input.x.getNneg());

		// Define the dimension of w
		dim = l.get(0).input.x.getFeature(0,0).length;

		// Initialize w
		w = new double[dim];
		w[0] = 1;

	}

}
