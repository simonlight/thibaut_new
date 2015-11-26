/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15.ranking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.Pair;


/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class RankingAPMantraM2CccpCuttingPlane1Slack<X,H> extends RankingMantraM2CccpCuttingPlane1Slack<LatentRankingInput<X, H>,List<LatentCoupleMinMax<H>>> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7166146352804336389L;

	/**
	 * 
	 * @param x
	 * @param w
	 * @return
	 * res[0] = hmax <br/>
	 * res[1] = max <br/>
	 * res[2] = hmin <br/>
	 * res[3] = min <br/>
	 */
	protected abstract Object[] valueOfHPlusMinus(X x, double[] w);

	@Override
	protected double valueOf(LatentRankingInput<X, H> x, RankingOutput y, List<LatentCoupleMinMax<H>> h) {
		return linear.valueOf(w, psi(x,y,h));
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingMantraM2CuttingPlane1Slack#psi(java.lang.Object, fr.durandt.jstruct.ssvm.ranking.RankingOutput, java.util.List)
	 */
	@Override
	protected double[] psi(LatentRankingInput<X, H> x, RankingOutput y, List<LatentCoupleMinMax<H>> h) {

		int[] count = new int[x.getNumberOfExamples()];
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			if(x.getLabel(i) == 1) {
				for(int j=0; j<x.getNumberOfExamples(); j++) {
					if(x.getLabel(j) != 1) {
						int yij = -1;
						if(y.getRanking(i) > y.getRanking(j)){
							yij = 1;
						}
						if(yij == 1) { // Yij = +1
							count[i]++;
							count[j]--;
						}
						else {	// Yij = -1
							count[i]--;
							count[j]++;
						}
					}
				}
			}
		}

		double[] psi = new double[dim];
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			double[] psiMax = x.getFeature(i, h.get(i).getHmax());
			double[] psiMin = x.getFeature(i, h.get(i).getHmin());
			for(int d=0; d<dim; d++) {
				psi[d] += (psiMax[d] + psiMin[d]) * count[i];
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
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingMantraM2#delta(fr.durandt.jstruct.ssvm.ranking.RankingOutput, fr.durandt.jstruct.ssvm.ranking.RankingOutput, java.lang.Object)
	 */
	@Override
	protected double delta(RankingOutput y, RankingOutput yp, List<LatentCoupleMinMax<H>> hp) {
		return 1 - RankingOutput.averagePrecision(y, yp);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingMantraM2#prediction(java.lang.Object, fr.durandt.jstruct.ssvm.ranking.RankingOutput, double[])
	 */
	@Override
	protected List<LatentCoupleMinMax<H>> prediction(LatentRankingInput<X, H> x, RankingOutput y, double[] w) {
		List<LatentCoupleMinMax<H>> latentPrediction = new ArrayList<LatentCoupleMinMax<H>>(x.getNumberOfExamples());
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			Object[] res = valueOfHPlusMinus(x.getExample(i), w);
			H hmax = (H)res[0];
			H hmin = (H)res[2];
			latentPrediction.add(i, new LatentCoupleMinMax<H>(hmax,hmin));
		}
		return latentPrediction;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingMantraM2#predictionOutputLatent(java.lang.Object, double[])
	 */
	@Override
	protected Object[] predictionOutputLatent(LatentRankingInput<X, H> x, double[] w) {
		List<Pair<Integer,Double>> examples = new ArrayList<Pair<Integer,Double>>(x.getNumberOfExamples());
		List<LatentCoupleMinMax<H>> latentPrediction = new ArrayList<LatentCoupleMinMax<H>>(x.getNumberOfExamples());

		for(int i=0; i<x.getNumberOfExamples(); i++) {
			Object[] res = valueOfHPlusMinus(x.getExample(i), w);
			H hmax = (H)res[0];
			H hmin = (H)res[2];
			latentPrediction.add(i, new LatentCoupleMinMax<H>(hmax,hmin));
			double score = linear.valueOf(w, x.getFeature(i, hmax)) + linear.valueOf(w, x.getFeature(i, hmin));
			examples.add(new Pair<Integer, Double>(i, score));
		}

		// Sort examples in descending order of score
		Collections.sort(examples, Collections.reverseOrder());
		Integer[] ranking = new Integer[examples.size()];

		for(int i=0; i<examples.size(); i++) {
			ranking[examples.get(i).getKey()] = x.getNumberOfExamples() - i;
		}

		RankingOutput prediction = new RankingOutput(ranking, x.getNpos(), x.getNneg());

		Object[] res = new Object[2];
		res[0] = prediction;
		res[1] = latentPrediction;
		return res;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingMantraM2#prediction(fr.durandt.jstruct.latent.LatentRepresentation, double[])
	 */
	@Override
	protected RankingOutput prediction(LatentRepresentation<LatentRankingInput<X, H>, List<LatentCoupleMinMax<H>>> x, 
			double[] w) {
		RankingOutput prediction = (RankingOutput)predictionOutputLatent(x.x, w)[0];
		return prediction;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingMantraM2#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Object[] lossAugmentedInference(
			STrainingSample<LatentRepresentation<LatentRankingInput<X, H>, List<LatentCoupleMinMax<H>>>, RankingOutput> ts,
			double[] w) {
		List<Pair<Integer,Double>> positiveExamples = new ArrayList<Pair<Integer,Double>>(ts.input.x.getNpos());
		List<Pair<Integer,Double>> negativeExamples = new ArrayList<Pair<Integer,Double>>(ts.input.x.getNneg());

		List<LatentCoupleMinMax<H>> latentPrediction = new ArrayList<LatentCoupleMinMax<H>>(ts.input.x.getNumberOfExamples());

		for(int i=0; i<ts.input.x.getNumberOfExamples(); i++) {
			Object[] res = valueOfHPlusMinus(ts.input.x.getExample(i), w);
			H hmax = (H)res[0];
			H hmin = (H)res[2];
			latentPrediction.add(i, new LatentCoupleMinMax<H>(hmax,hmin));

			// Compute the score with the predicted latent variable
			double score = linear.valueOf(w, ts.input.x.getFeature(i, hmax)) 
					+ linear.valueOf(w, ts.input.x.getFeature(i, hmin));

			if(ts.input.x.getLabel(i) == 1) {
				positiveExamples.add(new Pair<Integer, Double>(i, score));
			}
			else {
				negativeExamples.add(new Pair<Integer, Double>(i, score));
			}
		}

		// sort positiveExamples and negativeExamples in descending order of score
		Collections.sort(positiveExamples, Collections.reverseOrder());
		Collections.sort(negativeExamples, Collections.reverseOrder());

		int negativeId = 0;
		int positiveId = 0; 
		Integer[] imgIndexMap = new Integer[ts.input.x.getNumberOfExamples()];
		for(int i=0; i<ts.input.x.getNumberOfExamples(); i++){
			if(ts.input.x.getLabel(i) == 1){
				imgIndexMap[positiveExamples.get(positiveId).getKey()] = positiveId;
				positiveId++;
			}
			else{
				imgIndexMap[negativeExamples.get(negativeId).getKey()] = negativeId;
				negativeId++;
			}
		}    

		RankingOutput rankLAI = findOptimumNegLocations(ts.input.x, positiveExamples, negativeExamples, imgIndexMap);
		Object[] res = new Object[2];
		res[0] = rankLAI;
		res[1] = latentPrediction;
		return res;
	}

	protected RankingOutput findOptimumNegLocations(LatentRankingInput<X, H> x, List<Pair<Integer,Double>> positiveExamples, 
			List<Pair<Integer,Double>> negativeExamples, Integer[] imgIndexMap) {

		double maxValue = 0;
		double currentValue = 0;
		int maxIndex = -1;

		Integer[] optimumLocNegImg = new Integer[x.getNneg()];

		// for every jth negative image
		for(int j=1; j<=x.getNneg(); j++){
			maxValue = 0;
			maxIndex = x.getNpos()+1;
			// k is what we are maximising over. There would be one k_max for each negative image j
			currentValue = 0;
			for(int k=x.getNpos(); k>=1; k--){
				currentValue += (1/(double)x.getNpos())*((j/(double)(j+k))-((j-1)/(double)(j+k-1))) 
						- (2/(double)(x.getNpos()*x.getNneg()))*(positiveExamples.get(k-1).getValue() - negativeExamples.get(j-1).getValue());
				if(currentValue > maxValue){
					maxValue = currentValue;
					maxIndex = k;
				}
			}
			optimumLocNegImg[j-1] = maxIndex;
		}

		return encodeRanking(x, positiveExamples, negativeExamples, imgIndexMap, optimumLocNegImg);
	}

	protected RankingOutput encodeRanking(LatentRankingInput<X, H> x, List<Pair<Integer,Double>> positiveExamples, 
			List<Pair<Integer,Double>> negativeExamples, Integer[] imgIndexMap, Integer[] optimumLocNegImg){

		Integer[] ranking = new Integer[x.getNumberOfExamples()];
		for(int i=0; i<ranking.length; i++) {
			ranking[i] = 0;
		}

		for(int i=0; i<x.getNumberOfExamples(); i++){
			for(int j=i+1; j<x.getNumberOfExamples(); j++){        
				if(i == j){

				}
				else if(x.getLabel(i) == x.getLabel(j)){                
					if(x.getLabel(i) == 1){                                 
						if(positiveExamples.get(imgIndexMap[i]).getValue() > positiveExamples.get(imgIndexMap[j]).getValue()){
							ranking[i]++;
							ranking[j]--;
						}
						else if(positiveExamples.get(imgIndexMap[j]).getValue() > positiveExamples.get(imgIndexMap[i]).getValue()){
							ranking[i]--;
							ranking[j]++;
						}
						else{
							if(i < j){
								ranking[i]++;
								ranking[j]--;
							}
							else{
								ranking[i]--;
								ranking[j]++;
							}
						}
					}
					else{
						if(negativeExamples.get(imgIndexMap[i]).getValue() > negativeExamples.get(imgIndexMap[j]).getValue()){
							ranking[i]++;
							ranking[j]--;
						}
						else if(negativeExamples.get(imgIndexMap[j]).getValue() > negativeExamples.get(imgIndexMap[i]).getValue()){
							ranking[i]--;
							ranking[j]++;
						}
						else{
							if(i < j){
								ranking[i]++;
								ranking[j]--;
							}
							else{
								ranking[i]--;
								ranking[j]++;
							}
						}
					}        
				}
				else if((x.getLabel(i) == 1) && (x.getLabel(j) == 0)){
					int iPrime = imgIndexMap[i]+1;
					int jPrime = imgIndexMap[j]+1;
					int ojPrime = optimumLocNegImg[jPrime-1];

					if((ojPrime - iPrime - 0.5) > 0){
						ranking[i]++;
						ranking[j]--;
					}
					else{
						ranking[i]--;
						ranking[j]++;
					}
				}
				else if((x.getLabel(i) == 0) && (x.getLabel(j) == 1)){
					int iPrime = imgIndexMap[i]+1;
					int jPrime = imgIndexMap[j]+1;
					int oiPrime = optimumLocNegImg[iPrime-1];

					if((jPrime - oiPrime + 0.5) > 0){
						ranking[i]++;
						ranking[j]--;
					}
					else{
						ranking[i]--;
						ranking[j]++;
					}
				}                    
			}        
		}    

		return new RankingOutput(ranking, x.getNpos(), x.getNneg());
	}

	public double averagePrecision(List<STrainingSample<LatentRepresentation<LatentRankingInput<X, H>, List<LatentCoupleMinMax<H>>>, RankingOutput>> l) {
		RankingOutput prediction = prediction(l.get(0).input);
		return RankingOutput.averagePrecision(l.get(0).output, prediction);
	}
	
	public String toString() {
		String s = super.toString();
		s = "AveragePrecision_" + s;
		return s;
	}
}
