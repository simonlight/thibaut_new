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
public class ROCAreaMantraCuttingPlane1SlackBagImage extends ROCAreaMantraCuttingPlane1Slack<BagImage,Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1714387637625101447L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.ROCAreaMantraCuttingPlane1Slack#phi(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] phi(BagImage x, Integer h) {
		//System.out.println(h + "\t" + x);
		return x.getInstance(h);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.ROCAreaMantraCuttingPlane1Slack#valueOfHPlusMinus(java.lang.Object, double[])
	 */
	@Override
	protected Object[] valueOfHPlusMinus(BagImage x, double[] w) {
		Integer hmax = null;
		Integer hmin = null;
		double valmax = -Double.MAX_VALUE;
		double valmin = Double.MAX_VALUE;
		for(int h=0; h<x.numberOfInstances(); h++) {
			double[] phi = phi(x, h);
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

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImage, Integer>, List<List<LatentCouple<Integer>>>>, RankingOutput>> l) {

		// Print the classes and the number of relevant/irrelevant examples
		System.out.println("Rank ROC MANTRA \t P= " + l.get(0).input.x.getNpos() + "\t N= " + l.get(0).input.x.getNneg());

		// Define the dimension of w
		dim = l.get(0).input.x.getFeature(0,0).length;

		// Initialize w
		w = new double[dim];
		w[0] = 1;
	}

}
