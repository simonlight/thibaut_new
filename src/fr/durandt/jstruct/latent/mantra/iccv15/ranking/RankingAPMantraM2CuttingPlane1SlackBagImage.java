/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15.ranking;

import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImage;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class RankingAPMantraM2CuttingPlane1SlackBagImage extends RankingAPMantraM2CuttingPlane1Slack<BagImage, Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -7851177813774157677L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingMantraM2#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImage, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>> l) {
		// Print the classes and the number of relevant/irrelevant examples
		System.out.println("Rank AP MANTRA \t P= " + l.get(0).input.x.getNpos() + "\t N= " + l.get(0).input.x.getNneg());

		// Define the dimension of w
		dim = l.get(0).input.x.getFeature(0,0).length;

		// Initialize w
		w = new double[dim];
		w[0] = 1;

	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingMantraM2#valueOfHPlusMinus(java.lang.Object, double[])
	 */
	@Override
	protected Object[] valueOfHPlusMinus(BagImage x, double[] w) {
		Integer hmax = null;
		Integer hmin = null;
		double valmax = -Double.MAX_VALUE;
		double valmin = Double.MAX_VALUE;
		for(int h=0; h<x.numberOfInstances(); h++) {
			double[] phi = x.getInstance(h);
			double val = linear.valueOf(w, phi);
			if(val>valmax){
				valmax = val;
				hmax = h;
			}
			if(val<valmin){
				valmin = val;
				hmin = h;
			}
		}
		Object[] res = new Object[4];
		res[0] = hmax;
		res[1] = valmax;
		res[2] = hmin;
		res[3] = valmin;
		return res;
	}

}
