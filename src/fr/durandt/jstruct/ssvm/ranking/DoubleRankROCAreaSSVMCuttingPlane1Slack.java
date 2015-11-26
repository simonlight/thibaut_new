/**
 * 
 */
package fr.durandt.jstruct.ssvm.ranking;

import java.util.List;

import fr.durandt.jstruct.ssvm.SSVMCuttingPlane1Slack;
import fr.durandt.jstruct.struct.STrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class DoubleRankROCAreaSSVMCuttingPlane1Slack extends SSVMCuttingPlane1Slack<RankingInput,RankingOutput> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6690579966467733508L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected RankingOutput lossAugmentedInference(STrainingSample<RankingInput, RankingOutput> ts, double[] w) {

		double[] scores = new double[ts.input.getNumberOfExamples()];
		for(int i=0; i<ts.input.getNumberOfExamples(); i++) {
			double score = linear.valueOf(w, ts.input.getFeature(i));
			scores[i] = score;
		}

		Integer[][] y = new Integer[ts.input.getNumberOfExamples()][ts.input.getNumberOfExamples()];
		for(int i=0; i<ts.input.getNumberOfExamples(); i++) {
			for(int j=0; j<ts.input.getNumberOfExamples(); j++) {
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

		RankingOutput rank = new RankingOutput(ts.input.getNpos(), ts.input.getNpos());
		rank.setY(y);
		//rank.printY();
		if(verbose > 1) {
			System.out.println("ROC= " + RankingOutput.rocAreaY(ts.output, rank));
			System.out.println("delta= " + delta(ts.output, rank));
		}
		return rank;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#prediction(java.lang.Object, double[])
	 */
	@Override
	protected RankingOutput prediction(RankingInput x, double[] w) {

		double[] scores = new double[x.getNumberOfExamples()];
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			double score = linear.valueOf(w, x.getFeature(i));
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
		return prediction;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#delta(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double delta(RankingOutput yi, RankingOutput y) {
		return 1.0 - RankingOutput.rocAreaY(yi, y);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.ssvm.SSVM#psi(java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(RankingInput x, RankingOutput y) {

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
			double[] psii = x.getFeature(i);
			for(int d=0; d<dim; d++) {
				psi[d] += psii[d] * count[i];
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
	 * @see fr.durandt.jstruct.ssvm.SSVM#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<RankingInput, RankingOutput>> l) {
		// Print the classes and the number of relevant/irrelevant examples
		System.out.println("Rank ROCArea SSVM \t P= " + l.get(0).input.getNpos() + "\t N= " + l.get(0).input.getNneg());

		// Define the dimension of w
		dim = l.get(0).input.getFeature(0).length;

		// Initialize w
		w = new double[dim];
		w[0] = 1;
		for(int i=0; i<dim; i++) {
			w[i] = 1.;
		}

	}

	public double averagePrecision(List<STrainingSample<RankingInput, RankingOutput>> l) {
		RankingOutput prediction = prediction(l.get(0).input);
		return RankingOutput.averagePrecisionY(l.get(0).output, prediction);
	}

	public double rocArea(List<STrainingSample<RankingInput, RankingOutput>> l) {
		RankingOutput prediction = prediction(l.get(0).input);
		return RankingOutput.rocAreaY(l.get(0).output, prediction);
	}
}
