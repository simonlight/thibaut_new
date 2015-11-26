/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.ranking;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.LSSVMCuttingPlane1Slack;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.util.Pair;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public abstract class LAPSVMCuttingPlane1Slack<X,H> extends LSSVMCuttingPlane1Slack<LatentRankingInput<X,H>,RankingOutput, List<H>> {

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

		int[] count = new int[x.getNumberOfExamples()];
		for(int i=0; i<x.getNumberOfExamples(); i++) {
			if(x.getLabel(i) == 1) {
				for(int j=0; j<x.getNumberOfExamples(); j++) {
					if(x.getLabel(j) == 0) {
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
			double[] psii = x.getFeature(i, h.get(i));
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
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#delta(java.lang.Object, java.lang.Object, java.lang.Object)
	 */
	@Override
	protected double delta(RankingOutput yi, RankingOutput yp, List<H> hp) {
		// Stores rank of all images
		int[] ranking = new int[yi.getNumberOfExamples()]; 
		// Stores list of images sorted by rank. Higher rank to lower rank 
		int[] sortedExamples = new int[yi.getNumberOfExamples()]; 

		// convert rank matrix to rank list
		for(int i=0; i<yi.getNumberOfExamples(); i++){
			// start with lowest rank for each sample i.e 1 
			ranking[i] = 1; 
			for(int j=0; j<yi.getNumberOfExamples(); j++){
				if(yp.getRanking(i) > yp.getRanking(j)){
					ranking[i] = ranking[i] + 1;
				} 
			}
			sortedExamples[yi.getNumberOfExamples() - ranking[i]] = i;
		}  

		int posCount = 0;
		int totalCount = 0;
		double precisionAti = 0;
		for(int i=0; i<yi.getNumberOfExamples(); i++){
			int label = yi.getLabel(sortedExamples[i]);
			if(label == 1){
				posCount++;
				totalCount++;
			}
			else{
				totalCount++;
			}
			if(label == 1){
				precisionAti = precisionAti + (double)posCount/(double)totalCount;
			}
		}
		precisionAti = precisionAti/(double)posCount;

		double delta = 1 - precisionAti;

		return delta;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#lossAugmentedInference(fr.durandt.jstruct.struct.STrainingSample, double[])
	 */
	@Override
	protected Object[] lossAugmentedInference(
			STrainingSample<LatentRepresentation<LatentRankingInput<X, H>, List<H>>, RankingOutput> ts, double[] w) {
		
		List<Pair<Integer,Double>> positiveExamples = new ArrayList<Pair<Integer,Double>>(ts.input.x.getNpos());
		List<Pair<Integer,Double>> negativeExamples = new ArrayList<Pair<Integer,Double>>(ts.input.x.getNneg());
		
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

		List<Pair<Integer,Double>> examples = new ArrayList<Pair<Integer,Double>>(x.getNumberOfExamples());
		List<H> latentPrediction = new ArrayList<H>(x.getNumberOfExamples());

		for(int i=0; i<x.getNumberOfExamples(); i++) {
			H hpredict = latentPrediction(x.getExample(i));
			latentPrediction.add(i, hpredict);
			double score = linear.valueOf(w, x.getFeature(i, hpredict));
			examples.add(new Pair<Integer, Double>(i, score));
		}

		// sort examples in descending order of score
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
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#prediction(fr.durandt.jstruct.latent.LatentRepresentation, double[])
	 */
	@Override
	protected RankingOutput prediction(LatentRepresentation<LatentRankingInput<X, H>, List<H>> x, double[] w) {
		RankingOutput prediction = (RankingOutput)predictionOutputLatent(x.x, w)[0];
		return prediction;
	}
	
	public double averagePrecision(List<STrainingSample<LatentRepresentation<LatentRankingInput<X, H>, List<H>>, RankingOutput>> l) {
		RankingOutput prediction = prediction(l.get(0).input);
		return 1 - delta(l.get(0).output, prediction, null);
	}
	
	public String toString() {
		String s = "lapsvm_cuttingplane1slack_lambda_" + lambda + "_epsilon_" + epsilon 
				+ "_maxCCCPIter_" + maxCCCPIter + "_minCCCPIter_" + minCCCPIter
				+ "_cpmax_" + cpmax + "_cpmin_" + cpmin;
		return s;
	}
}
