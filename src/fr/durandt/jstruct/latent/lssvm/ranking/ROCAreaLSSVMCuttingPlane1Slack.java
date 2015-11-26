/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.ranking;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.LSSVMCuttingPlane1Slack;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class ROCAreaLSSVMCuttingPlane1Slack<X,H> extends LSSVMCuttingPlane1Slack<LatentRankingInput<X,H>,RankingOutput, List<H>> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2080057776568255395L;

	protected abstract H latentPrediction(X x, double[] w);

	protected H latentPrediction(X x) {
		return latentPrediction(x, w);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#psi(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(LatentRankingInput<X, H> x, RankingOutput y, List<H> h) {

		double[] psi = new double[dim];

		int[] count = new int[x.getNumberOfExamples()];
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			for(int j=0; j<x.getNumberOfExamples(); j++) {
				if(y.getY(i, j) == 1) { // Yij = +1
					count[i]++;
					count[j]--;
				}
				else {	// Yij = -1
					count[i]--;
					count[j]++;
				}
			}
		}

		for(int i=0; i<x.getNumberOfExamples(); i++) {
			double[] psii = x.getFeature(i, h.get(i));
			double c = count[i];
			for(int d=0; d<dim; d++) {
				psi[d] += psii[d] * c;
			}
		}

		// Divide by the number of relevant examples * number of irrelevant examples
		double c = (double)(x.getNpos()*x.getNneg());
		for(int d=0; d<dim; d++) {
			psi[d] /= c;
		}

		return psi;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#delta(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double delta(RankingOutput yi, RankingOutput yp, List<H> hp) {
		return 1.0 - RankingOutput.rocAreaY(yi, yp);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Object[] lossAugmentedInference(
			STrainingSample<LatentRepresentation<LatentRankingInput<X, H>, List<H>>, RankingOutput> ts, double[] w) {

		double[] scores = new double[ts.input.x.getNumberOfExamples()];
		List<H> latentPrediction = new ArrayList<H>(ts.input.x.getNumberOfExamples());

		for(int i=0; i<ts.input.x.getNumberOfExamples(); i++) {

			if(ts.input.x.getLabel(i) != 1) {
				// Compute the latent variable for negative examples
				H hpredict = latentPrediction(ts.input.x.getExample(i));
				ts.input.h.set(i, hpredict);
			}
			latentPrediction.add(i, ts.input.h.get(i));

			// Compute the score with the predicted latent variable
			double score = linear.valueOf(w, ts.input.x.getFeature(i,ts.input.h.get(i)));
			scores[i] = score;
		}

		Integer[][] y = new Integer[ts.input.x.getNumberOfExamples()][ts.input.x.getNumberOfExamples()];
		for(int i=0; i<ts.input.x.getNumberOfExamples(); i++) {
			for(int j=0; j<ts.input.x.getNumberOfExamples(); j++) {
				double val = scores[i] - 0.25 - (scores[j] + 0.25);
				if(val > 0) {
					y[i][j] = 1;
				}
				else if(val < 0) {
					y[i][j] = -1;
				}
				else {
					y[i][j] = 0;
				}
			}	
		}

		RankingOutput rank = new RankingOutput(ts.input.x.getNpos(), ts.input.x.getNpos());
		rank.setY(y);
		if(verbose > 1) {
			System.out.println("ROC= " + RankingOutput.rocAreaY(ts.output, rank));
			System.out.println("delta= " + delta(ts.output, rank, latentPrediction));
		}
		Object[] res = new Object[2];
		res[0] = rank;
		res[1] = latentPrediction;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#prediction(java.lang.Object, java.lang.Object, double[])
	 */
	@Override
	protected List<H> prediction(LatentRankingInput<X, H> x, RankingOutput y, double[] w) {
		List<H> latentPrediction = new ArrayList<H>(x.getNumberOfExamples());
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			// Predict the best latent value for example i
			H hpredict = latentPrediction(x.getExample(i));
			latentPrediction.add(i, hpredict);
		}
		return latentPrediction;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#predictionOutputLatent(java.lang.Object, double[])
	 */
	@Override
	protected Object[] predictionOutputLatent(LatentRankingInput<X, H> x, double[] w) {

		double[] scores = new double[x.getNumberOfExamples()];
		List<H> latentPrediction = new ArrayList<H>(x.getNumberOfExamples());

		for(int i=0; i<x.getNumberOfExamples(); i++) {
			H hpredict = latentPrediction(x.getExample(i));
			latentPrediction.add(i, hpredict);
			double score = linear.valueOf(w, x.getFeature(i, hpredict));
			scores[i] = score;
		}

		Integer[][] y = new Integer[x.getNumberOfExamples()][x.getNumberOfExamples()];
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			for(int j=0; j<x.getNumberOfExamples(); j++) {
				double val = scores[i] - scores[j];
				if(val > 0) {
					y[i][j] = 1;
				}
				else if(val < 0) {
					y[i][j] = -1;
				}
				else {
					y[i][j] = 0;
				}
			}	
		}

		RankingOutput prediction = new RankingOutput(x.getNpos(), x.getNpos());
		prediction.setY(y);

		Object[] res = new Object[2];
		res[0] = prediction;
		res[1] = latentPrediction;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#prediction(fr.durandt.jstruct.latent.LatentRepresentation, double[])
	 */
	@Override
	protected RankingOutput prediction(LatentRepresentation<LatentRankingInput<X, H>, List<H>> x, double[] w) {
		RankingOutput prediction = (RankingOutput)predictionOutputLatent(x.x, w)[0];
		return prediction;
	}

	public double averagePrecision(List<STrainingSample<LatentRepresentation<LatentRankingInput<X, H>, List<H>>, RankingOutput>> l) {
		RankingOutput prediction = prediction(l.get(0).input);
		return RankingOutput.averagePrecisionY(l.get(0).output, prediction);
	}

	public double rocArea(List<STrainingSample<LatentRepresentation<LatentRankingInput<X, H>, List<H>>, RankingOutput>> l) {
		RankingOutput prediction = prediction(l.get(0).input);
		return RankingOutput.rocAreaY(l.get(0).output, prediction);
	}
}
