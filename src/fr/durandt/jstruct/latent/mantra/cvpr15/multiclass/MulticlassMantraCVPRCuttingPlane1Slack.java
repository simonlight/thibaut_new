/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.cvpr15.multiclass;

import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPRCuttingPlane1Slack;
import fr.durandt.jstruct.struct.STrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class MulticlassMantraCVPRCuttingPlane1Slack<X,H> extends MantraCVPRCuttingPlane1Slack<X, Integer, H> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6528177403309258908L;

	/**
	 * list of classes {0,1,...,c-1}
	 */
	protected List<Integer> listClass = null;

	protected abstract Object[] valueOfHPlusMinus(X x, int y, double[] w);

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPRCuttingPlane1Slack#inferPrediction(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected Object[] inferPrediction(X x, Integer y, double[] w) {
		ComputedScoresMinMax<H> precomputedScore = precomputedScores(x, w);
		Integer ym = precomputedScore.getMaxY(y);
		Object[] res = new Object[3];
		res[0] = precomputedScore.getHmax(y);
		res[1] = ym;
		res[2] = precomputedScore.getHmin(ym);
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Object[] lossAugmentedInference(STrainingSample<LatentRepresentation<X, H>, Integer> ts, double[] w) {

		ComputedScoresMinMax<H> precomputedScore = precomputedScores(ts.input.x, w);

		double max = -Double.MAX_VALUE;
		Object[] res = new Object[4];
		for(int y : listClass) {
			int ym = precomputedScore.getMaxY(y);
			double score = delta(ts.output, y, precomputedScore.getHmax(ts.output)) 
					+ precomputedScore.getVmax(y) - precomputedScore.getVmin(ym);
			if(score > max){
				max = score;
				res[0] = y;
				res[1] = precomputedScore.getHmax(y);
				res[2] = ym;
				res[3] = precomputedScore.getHmin(ym);
			}
		}
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#predictionOutputLatent(java.lang.Object, double[])
	 */
	@Override
	protected Object[] predictionOutputLatent(X x, double[] w) {
		ComputedScoresMinMax<H> precomputedScore = precomputedScores(x, w);
		double max = -Double.MAX_VALUE;
		Integer yp = -1;
		for(Integer y :listClass) {
			double score = valueOf(x, y, precomputedScore);
			if(score > max) {
				max = score;
				yp = y;
			}
		}
		Object[] res = new Object[2];
		res[0] = yp;
		res[1] = precomputedScore.getHmax(yp);
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#prediction(fr.durandt.jstruct.latent.LatentRepresentation, double[])
	 */
	@Override
	protected Integer prediction(LatentRepresentation<X, H> x, double[] w) {
		return (Integer)predictionOutputLatent(x.x, w)[0];
	}

	/**
	 * Pre-compute the min and max score for each class 
	 * @param rep
	 * @param w
	 * @return
	 */
	protected ComputedScoresMinMax<H> precomputedScores(final X x, final double[] w) {
		ComputedScoresMinMax<H> scores = new ComputedScoresMinMax<H>();
		// For each class
		for(Integer y : listClass) {
			// Compute the maximum and minimum scores, and the predicted latent variables
			Object[] or = valueOfHPlusMinus(x, y, w);
			scores.add((H)or[0], (Double)or[1], (H)or[2], (Double)or[3]);
		}
		return scores;
	}

	protected double valueOf(X x, Integer y, ComputedScoresMinMax<H> precomputedScore) {
		Integer ym = precomputedScore.getMaxY(y);
		double score = precomputedScore.getVmax(y) - precomputedScore.getVmin(ym);
		return score;
	}


}
