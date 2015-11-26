/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15.ranking;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPRCuttingPlane1Slack;
import fr.durandt.jstruct.latent.mantra.cvpr15.multiclass.ComputedScoresMinMax;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class ROCAreaMantraCuttingPlane1Slack<X,H> extends MantraCVPRCuttingPlane1Slack<LatentRankingInput<X,H>,RankingOutput,List<List<LatentCouple<H>>>> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8703965623199287109L;

	protected abstract double[] phi(X x, H h);

	/**
	 * 
	 * @param x
	 * @param y
	 * @param w
	 * @return (hmax, max, hmin, min) where: <br/>
	 * hmax = argmax_h &lt w, phi(x,h) &gt <br/>
	 * max = max_h &lt w, phi(x,h) &gt <br/>
	 * hmin = argmin_h &lt w, phi(x,h) &gt <br/>
	 * min = min_h &lt w, phi(x,h) &gt <br/>
	 */
	protected abstract Object[] valueOfHPlusMinus(X x, double[] w);

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPRCuttingPlane1Slack#inferPrediction(java.lang.Object, java.lang.Object, double[])
	 */
	@Override
	protected Object[] inferPrediction(LatentRankingInput<X, H> x, RankingOutput y, double[] w) {

		List<List<LatentCouple<H>>> hp = prediction(x, y, w);

		// Precompute the scores
		List<ComputedScoresMinMax<H>> precomputedScores = precomputedScores(x, w);
		// Predicted ranking matrix
		Integer[][] rankMatrix = new Integer[x.getNumberOfExamples()][x.getNumberOfExamples()];
		// Predicted latent variable
		List<List<LatentCouple<H>>> latent = new ArrayList<List<LatentCouple<H>>>(x.getNumberOfExamples());
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			List<LatentCouple<H>> latenti = new ArrayList<LatentCouple<H>>(x.getNumberOfExamples());
			for(int j=0; j<x.getNumberOfExamples(); j++) {
				latenti.add(j,null);
			}
			latent.add(i,latenti);
		}

		for(int i=0; i<x.getNumberOfExamples(); i++) {
			for(int j=0; j<x.getNumberOfExamples(); j++) {
				Object[] res = inferPrediction(precomputedScores.get(i), precomputedScores.get(j), y.getY(i,j));
				rankMatrix[i][j] = (Integer)res[0];
				latent.get(i).set(j, (LatentCouple<H>) res[1]);
			}
		}

		RankingOutput rank = new RankingOutput();
		rank.setY(rankMatrix);

		Object[] res = new Object[3];
		res[0] = hp;
		res[1] = rank;
		res[2] = latent;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#psi(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double[] psi(LatentRankingInput<X, H> x, RankingOutput y, List<List<LatentCouple<H>>> h) {

		double[] psi = new double[dim];
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			if(x.getLabel(i) == 1) {	// i is positive
				for(int j=0; j<x.getNumberOfExamples(); j++) {
					if(x.getLabel(j) == 0) {	// j is negative
						int yij = y.getY(i, j);
						if(yij != 0) {
							double[] phii = phi(x.getExample(i), h.get(i).get(j).getHi());
							double[] phij = phi(x.getExample(j), h.get(i).get(j).getHj());
							for(int d=0; d<dim; d++) {
								psi[d] += yij*(phii[d] - phij[d]);
							}
						}
					}
				}
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
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#delta(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double delta(RankingOutput y, RankingOutput yp, List<List<LatentCouple<H>>> hp) {
		return 1 - RankingOutput.rocAreaY(y, yp);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Object[] lossAugmentedInference(
			STrainingSample<LatentRepresentation<LatentRankingInput<X, H>, List<List<LatentCouple<H>>>>, RankingOutput> ts,
			double[] w) {

		// Precompute the scores
		List<ComputedScoresMinMax<H>> precomputedScores = precomputedScores(ts.input.x, w);

		// Initialize ranking matrix yp
		Integer[][] yp = new Integer[ts.input.x.getNumberOfExamples()][ts.input.x.getNumberOfExamples()];

		// Initialize ranking matrix ym
		Integer[][] ym = new Integer[ts.input.x.getNumberOfExamples()][ts.input.x.getNumberOfExamples()];

		// Initialize latent variable hp
		List<List<LatentCouple<H>>> hp = new ArrayList<List<LatentCouple<H>>>(ts.input.x.getNumberOfExamples());
		for(int i=0; i<ts.input.x.getNumberOfExamples(); i++) {
			List<LatentCouple<H>> latenti = new ArrayList<LatentCouple<H>>(ts.input.x.getNumberOfExamples());
			for(int j=0; j<ts.input.x.getNumberOfExamples(); j++) {
				latenti.add(j,null);
			}
			hp.add(i,latenti);
		}

		// Initialize latent variable hm
		List<List<LatentCouple<H>>> hm = new ArrayList<List<LatentCouple<H>>>(ts.input.x.getNumberOfExamples());
		for(int i=0; i<ts.input.x.getNumberOfExamples(); i++) {
			List<LatentCouple<H>> latenti = new ArrayList<LatentCouple<H>>(ts.input.x.getNumberOfExamples());
			for(int j=0; j<ts.input.x.getNumberOfExamples(); j++) {
				latenti.add(j,null);
			}
			hm.add(i,latenti);
		}

		for(int i=0; i<ts.input.x.getNumberOfExamples(); i++) {
			for(int j=0; j<ts.input.x.getNumberOfExamples(); j++) {
				//Object[] res = lossAugmentedInference(precomputedScores.get(i), precomputedScores.get(j), ts.output.getY(i,j));
				Object[] res = lossAugmentedInference(precomputedScores.get(i), precomputedScores.get(j), 1);
				yp[i][j] = (Integer)res[0];
				hp.get(i).set(j, (LatentCouple<H>) res[1]);
				ym[i][j] = (Integer)res[2];
				hm.get(i).set(j, (LatentCouple<H>) res[3]);
			}
		}

		RankingOutput rankp = new RankingOutput();
		rankp.setY(yp);

		//System.out.println("roc= " + RankingOutput.rocAreaY(ts.output, rankp));

		RankingOutput rankm = new RankingOutput();
		rankm.setY(ym);

		Object[] res = new Object[4];
		res[0] = rankp;
		res[1] = hp;
		res[2] = rankm;
		res[3] = hm;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#prediction(java.lang.Object, java.lang.Object, double[])
	 */
	@Override
	protected List<List<LatentCouple<H>>> prediction(LatentRankingInput<X, H> x, RankingOutput y, double[] w) {

		// Precompute the scores
		List<ComputedScoresMinMax<H>> precomputedScores = precomputedScores(x,w);
		// Predicted latent variable
		List<List<LatentCouple<H>>> latent = new ArrayList<List<LatentCouple<H>>>(x.getNumberOfExamples());
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			List<LatentCouple<H>> latenti = new ArrayList<LatentCouple<H>>(x.getNumberOfExamples());
			for(int j=0; j<x.getNumberOfExamples(); j++) {
				latenti.add(j,null);
			}
			latent.add(i,latenti);
		}

		for(int i=0; i<x.getNumberOfExamples(); i++) {
			for(int j=0; j<x.getNumberOfExamples(); j++) {
				if(y.getY(i, j) == 1) {
					latent.get(i).set(j, new LatentCouple<H>(precomputedScores.get(i).getHmax(0), precomputedScores.get(j).getHmin(0)));
				}
				else if(y.getY(i, j) == -1) {
					latent.get(i).set(j, new LatentCouple<H>(precomputedScores.get(i).getHmin(0), precomputedScores.get(j).getHmax(0)));
				}
				else {
					latent.get(i).set(j, null);
				}
			}
		}

		return latent;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#predictionOutputLatent(java.lang.Object, double[])
	 */
	@Override
	protected Object[] predictionOutputLatent(LatentRankingInput<X, H> x, double[] w) {

		// Precompute the scores
		List<ComputedScoresMinMax<H>> precomputedScores = precomputedScores(x,w);
		// Predicted ranking matrix
		Integer[][] y = new Integer[x.getNumberOfExamples()][x.getNumberOfExamples()];
		// Predicted latent variable
		List<List<LatentCouple<H>>> latent = new ArrayList<List<LatentCouple<H>>>(x.getNumberOfExamples());
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			List<LatentCouple<H>> latenti = new ArrayList<LatentCouple<H>>(x.getNumberOfExamples());
			for(int j=0; j<x.getNumberOfExamples(); j++) {
				latenti.add(j,null);
			}
			latent.add(i,latenti);
		}

		for(int i=0; i<x.getNumberOfExamples(); i++) {
			for(int j=0; j<x.getNumberOfExamples(); j++) {
				Object[] res = prediction(precomputedScores.get(i), precomputedScores.get(j));
				y[i][j] = (Integer)res[0];
				latent.get(i).set(j, (LatentCouple<H>) res[1]);
			}
		}

		RankingOutput rank = new RankingOutput(x.getNpos(), x.getNneg());
		rank.setY(y);

		Object[] res = new Object[2];
		res[0] = rank;
		res[1] = latent;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.cvpr15.MantraCVPR#prediction(fr.durandt.jstruct.latent.LatentRepresentation, double[])
	 */
	@Override
	protected RankingOutput prediction(LatentRepresentation<LatentRankingInput<X, H>, List<List<LatentCouple<H>>>> x, double[] w) {
		return (RankingOutput) predictionOutputLatent(x.x, w)[0];
	}

	/**
	 * max_yij { max_{hi,hj} yij (&lt w, phi(xi,hi) &gt - &lt w, phi(xj,hj) &gt - 
	 * max_y'!=yij { min_{hi,hj} y'(&lt w, phi(xi,hi) &gt - &lt w, phi(xj,hj) &gt } }
	 * @param x
	 * @param w
	 * @return
	 * yij = argmax_yij { max_{hi,hj} yij (&lt w, phi(xi,hi) &gt - &lt w, phi(xj,hj) &gt - 
	 * max_y'!=yij { min_{hi,hj} y'(&lt w, phi(xi,hi) &gt - &lt w, phi(xj,hj) &gt } }<br/>
	 * hi = max_hi yij &lt w, phi(xi,hi) &gt <br/>
	 * hj = max_hj - yij &lt w, phi(xj,hj) &gt<br/>
	 */
	protected Object[] prediction(ComputedScoresMinMax<H> xi, ComputedScoresMinMax<H> xj) {

		H hmaxi = xi.getHmax(0);
		double maxi = xi.getVmax(0);
		H hmini = xi.getHmin(0);
		double mini = xi.getVmin(0);

		H hmaxj = xj.getHmax(0);
		double maxj = xj.getVmax(0);
		H hminj = xj.getHmin(0);
		double minj = xj.getVmin(0);

		return prediction(hmaxi, maxi, hmini, mini, hmaxj, maxj, hminj, minj);
	}

	protected Object[] prediction(H hmaxi, double maxi, H hmini, double mini, 
			H hmaxj, double maxj, H hminj, double minj) {

		Object[] prediction = new Object[2];
		if(maxi - minj == maxj - mini) {
			prediction[0] = 0;
			prediction[1] = null;
		}
		else if(maxi - minj > maxj - mini) {
			prediction[0] = 1;
			prediction[1] = new LatentCouple<H>(hmaxi, hminj);
		}
		else {
			prediction[0] = -1;
			prediction[1] = new LatentCouple<H>(hmini, hmaxj);
		}

		return prediction;
	}


	/**
	 * max_y { delta(yij,y) + max_{hi,hj} y (&lt w, phi(xi,hi) &gt - &lt w, phi(xj,hj) &gt - 
	 * max_y'!=yij { min_{hi,hj} y'(&lt w, phi(xi,hi) &gt - &lt w, phi(xj,hj) &gt } }
	 * @param x
	 * @param yij
	 * @param w
	 * @return
	 * yij = argmax_yij { max_{hi,hj} yij (&lt w, phi(xi,hi) &gt - &lt w, phi(xj,hj) &gt - 
	 * max_y'!=yij { min_{hi,hj} y'(&lt w, phi(xi,hi) &gt - &lt w, phi(xj,hj) &gt } }<br/>
	 * hi = max_hi yij &lt w, phi(xi,hi) &gt <br/>
	 * hj = max_hj - yij &lt w, phi(xj,hj) &gt<br/>
	 */
	protected Object[] lossAugmentedInference(ComputedScoresMinMax<H> xi, ComputedScoresMinMax<H> xj, Integer yij) {

		H hmaxi = xi.getHmax(0);
		double maxi = xi.getVmax(0);
		H hmini = xi.getHmin(0);
		double mini = xi.getVmin(0);

		H hmaxj = xj.getHmax(0);
		double maxj = xj.getVmax(0);
		H hminj = xj.getHmin(0);
		double minj = xj.getVmin(0);

		return lossAugmentedInference(yij, hmaxi, maxi, hmini, mini, hmaxj, maxj, hminj, minj);
	}

	protected Object[] lossAugmentedInference(Integer yij, H hmaxi, double maxi, H hmini, double mini, 
			H hmaxj, double maxj, H hminj, double minj) {

		double score = maxi - minj + (yij == 1 ? 0.5*(yij - 1) : 0);
		Object[] prediction = new Object[4];
		prediction[0] = 1;	// yij = 1
		prediction[1] = new LatentCouple<H>(hmaxi, hminj);
		prediction[2] = 0;	// y'ij = 0
		prediction[3] = null;

		if(2*(maxi - minj) + (yij == 1 ? 0.5*(yij - 1) : 0) > score) {
			score = 2*(maxi - minj) + (yij == 1 ? 0.5*(yij - 1) : 0);
			prediction[0] = 1;
			prediction[1] = new LatentCouple<H>(hmaxi, hminj);
			prediction[2] = -1;	// y'ij = -1
			prediction[3] = new LatentCouple<H>(hmaxi, hminj);
		}

		if(maxj - mini + (yij == 1 ? 0.5*(yij + 1) : 0) > score) {
			score = maxj - mini + (yij == 1 ? 0.5*(yij + 1) : 0);
			prediction[0] = -1;
			prediction[1] = new LatentCouple<H>(hmini, hmaxj);
			prediction[2] = 0;	// y'ij = 0
			prediction[3] = null;
		}

		if(2*(maxj - mini) + (yij == 1 ? 0.5*(yij + 1) : 0) > score) {
			score = 2*(maxj - mini) + (yij == 1 ? 0.5*(yij + 1) : 0);
			prediction[0] = -1;
			prediction[1] = new LatentCouple<H>(hmini, hmaxj);
			prediction[2] = 1;	// y'ij = 1
			prediction[3] = new LatentCouple<H>(hmini, hmaxj);
		}

		return prediction;
	}

	/**
	 * max_y'!=yij { min_{hi,hj} y'(&lt w, phi(xi,hi) &gt - &lt w, phi(xj,hj) &gt }
	 * @param x
	 * @param yij
	 * @param w
	 * @return
	 * y'ij <br/>
	 * hi <br/>
	 * hj <br/>
	 */
	protected Object[] inferPrediction(ComputedScoresMinMax<H> xi, ComputedScoresMinMax<H> xj, Integer yij) {

		H hmaxi = xi.getHmax(0);
		double maxi = xi.getVmax(0);
		H hmini = xi.getHmin(0);
		double mini = xi.getVmin(0);

		H hmaxj = xj.getHmax(0);
		double maxj = xj.getVmax(0);
		H hminj = xj.getHmin(0);
		double minj = xj.getVmin(0);

		return inferPrediction(yij, hmaxi, maxi, hmini, mini, hmaxj, maxj, hminj, minj);
	}

	protected Object[] inferPrediction(int yij, H hmaxi, double maxi, H hmini, double mini, 
			H hmaxj, double maxj, H hminj, double minj) {
		Object[] prediction = new Object[2];
		if(yij == 1) {
			if(minj - maxi > 0) {
				// y'ij = -1
				prediction[0] = -1;
				LatentCouple<H> h = new LatentCouple<H>(hmaxi, hminj);
				prediction[1] = h; 
			}
			else {
				// y'ij = 0
				prediction[0] = 0;
				prediction[1] = null;
			}
		}
		else {
			if(mini > maxj) {
				// y'ij = 1
				prediction[0] = 1;
				LatentCouple<H> h = new LatentCouple<H>(hmini, hmaxj);
				prediction[1] = h; 
			}
			else {
				// y'ij = 0
				prediction[0] = 0;
				prediction[1] = null;
			}
		}

		return prediction;
	}


	/**
	 * Pre-compute the min and max score for each class 
	 * @param rep
	 * @param w
	 * @return
	 */
	protected ComputedScoresMinMax<H> precomputedScores(X x, double[] w) {
		ComputedScoresMinMax<H> score = new ComputedScoresMinMax<H>();
		Object[] or = valueOfHPlusMinus(x, w);
		score.add((H)or[0], (Double)or[1], (H)or[2], (Double)or[3]);
		return score;
	}

	protected List<ComputedScoresMinMax<H>> precomputedScores(LatentRankingInput<X, H> x, double[] w) {
		List<ComputedScoresMinMax<H>> scores = new ArrayList<ComputedScoresMinMax<H>>(x.getNumberOfExamples());
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			scores.add(i, precomputedScores(x.getExample(i), w));
		}
		return scores;
	}


	public double rocArea(List<STrainingSample<LatentRepresentation<LatentRankingInput<X, H>, List<List<LatentCouple<H>>>>, RankingOutput>> l) {
		RankingOutput prediction = prediction(l.get(0).input);
		return RankingOutput.rocAreaY(l.get(0).output, prediction);
	}

	public double averagePrecision(List<STrainingSample<LatentRepresentation<LatentRankingInput<X, H>, List<List<LatentCouple<H>>>>, RankingOutput>> l) {
		RankingOutput prediction = prediction(l.get(0).input);
		return RankingOutput.averagePrecisionY(l.get(0).output, prediction);
	}


	/**
	 * max_y'!=yij { min_{hi,hj} y'(&lt w, phi(xi,hi) &gt - &lt w, phi(xj,hj) &gt }
	 * @param x
	 * @param yij
	 * @param w
	 * @return
	 * y'ij <br/>
	 * hi <br/>
	 * hj <br/>
	 */
	/*protected Object[] inferPrediction(X xi, X xj, Integer yij, double[] w) {

		// compute the max/min for example i
		Object[] res = valueOfHPlusMinus(xi, w);
		H hmaxi = (H)res[0];
		double maxi = (double)res[1];
		H hmini = (H)res[2];
		double mini = (double)res[3];

		// compute the max/min for example j
		res = valueOfHPlusMinus(xj, w);
		H hmaxj = (H)res[0];
		double maxj = (double)res[1];
		H hminj = (H)res[2];
		double minj = (double)res[3];

		return inferPrediction(yij, hmaxi, maxi, hmini, mini, hmaxj, maxj, hminj, minj);
	}*/
	
	public String toString() {
		String s = super.toString();
		s = "ranking_rocarea_" + s;
		return s;
	}

}
